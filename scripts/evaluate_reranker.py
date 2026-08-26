"""评测 Hybrid RRF + BGE Reranker。

完整流程：

PDF
→ 页面 Document
→ Chunk
→ Dense Top-N
→ BM25 Top-N
→ RRF 融合得到候选 Top-10
→ BGE Cross-Encoder 重排
→ 最终 Top-5
→ 计算 Hit@5、Recall@5、MRR@5
→ 对比 Dense、BM25、Hybrid、Reranker
→ 保存 reports/reranker_results.json

评测数据集继续使用 raglab.settings.EVAL_DATASET_PATH，
与 Dense、BM25 和 Hybrid 基线保持一致。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import fmean
from typing import Any, Sequence

from langchain_core.documents import Document

from evaluation.metrics import (
    hit_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from raglab.retrieval.hybrid import (
    RRFHybridRetriever,
)
from raglab.retrieval.reranker import (
    BGEReranker,
)
from raglab.settings import (
    CONFIG_DIR,
    EVAL_DATASET_PATH,
    PROJECT_ROOT,
)
from scripts.evaluate_hybrid import (
    build_bm25_index,
    build_bm25_search,
    build_chunks,
    build_dense_search,
    get_optional,
    get_required,
    load_report_metrics,
    load_yaml,
    resolve_project_path,
)


DEFAULT_CONFIG_PATH = (
    CONFIG_DIR
    / "reranker.yaml"
)


def parse_args() -> argparse.Namespace:
    """读取命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Hybrid RRF "
            "+ BGE Reranker."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="reranker.yaml 的路径。",
    )

    parser.add_argument(
        "--debug-query",
        type=str,
        default=None,
        help=(
            "可选。打印一个查询在 RRF 前后和 "
            "Reranker 重排后的排名。"
        ),
    )

    return parser.parse_args()


def load_eval_dataset() -> list[dict[str, Any]]:
    """读取与已有基线相同的评测数据集。"""

    if not EVAL_DATASET_PATH.exists():
        raise FileNotFoundError(
            "评测集不存在："
            f"{EVAL_DATASET_PATH}"
        )

    with EVAL_DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        dataset = json.load(file)

    if not isinstance(dataset, list):
        raise ValueError(
            "eval_dataset.json 根节点必须是列表。"
        )

    for index, row in enumerate(dataset):
        if not isinstance(row, dict):
            raise TypeError(
                f"评测项[{index}] 必须是字典，"
                f"实际类型：{type(row)!r}"
            )

    return dataset


def get_doc_id(
    document: Document,
) -> str:
    """读取 Document metadata 中的 doc_id。"""

    doc_id = document.metadata.get(
        "doc_id"
    )

    if not isinstance(doc_id, str):
        raise ValueError(
            "检索结果缺少有效 doc_id："
            f"{document.metadata}"
        )

    normalized = doc_id.strip()

    if not normalized:
        raise ValueError(
            "检索结果中的 doc_id 不能为空。"
        )

    return normalized


def average_or_none(
    values: Sequence[float],
) -> float | None:
    """计算平均值，空序列返回 None。"""

    if not values:
        return None

    return float(
        fmean(values)
    )


def serialize_document(
    document: Document,
    rank: int,
    *,
    save_text: bool,
) -> dict[str, Any]:
    """将单个 Document 转换成 JSON 数据。"""

    metadata = document.metadata

    result: dict[str, Any] = {
        "rank": rank,
        "chunk_id": metadata.get(
            "chunk_id"
        ),
        "doc_id": metadata.get(
            "doc_id"
        ),
        "title": metadata.get(
            "title"
        ),
        "page_number": metadata.get(
            "page_number"
        ),
        "section_id": metadata.get(
            "section_id"
        ),
        "source": metadata.get(
            "source"
        ),
        "retrieval_method": metadata.get(
            "retrieval_method"
        ),

        # RRF 阶段信息。
        "rrf_score": metadata.get(
            "rrf_score"
        ),
        "dense_rank": metadata.get(
            "dense_rank"
        ),
        "bm25_rank": metadata.get(
            "bm25_rank"
        ),
        "dense_raw_score": metadata.get(
            "dense_raw_score"
        ),
        "bm25_raw_score": metadata.get(
            "bm25_raw_score"
        ),

        # Reranker 阶段信息。
        "pre_rerank_rank": metadata.get(
            "pre_rerank_rank"
        ),
        "rerank_rank": metadata.get(
            "rerank_rank"
        ),
        "reranker_score": metadata.get(
            "reranker_score"
        ),
        "reranker_model": metadata.get(
            "reranker_model"
        ),
    }

    if save_text:
        result["text"] = (
            document.page_content
        )

    return result


def serialize_documents(
    documents: Sequence[Document],
    *,
    save_text: bool,
) -> list[dict[str, Any]]:
    """序列化一组 Document。"""

    return [
        serialize_document(
            document,
            rank,
            save_text=save_text,
        )
        for rank, document in enumerate(
            documents,
            start=1,
        )
    ]


def build_pipeline(
    reranker_config: dict[str, Any],
) -> tuple[
    RRFHybridRetriever,
    BGEReranker,
    dict[str, Any],
]:
    """构建 Hybrid Retriever 和 BGE Reranker。"""

    hybrid_config_path = resolve_project_path(
        get_required(
            reranker_config,
            "paths",
            "hybrid_config",
        )
    )

    hybrid_config = load_yaml(
        hybrid_config_path
    )

    dense_config_path = resolve_project_path(
        get_required(
            hybrid_config,
            "paths",
            "dense_config",
        )
    )

    bm25_config_path = resolve_project_path(
        get_required(
            hybrid_config,
            "paths",
            "bm25_config",
        )
    )

    dense_config = load_yaml(
        dense_config_path
    )

    bm25_config = load_yaml(
        bm25_config_path
    )

    dense_candidate_top_k = int(
        get_required(
            hybrid_config,
            "retrieval",
            "dense_candidate_top_k",
        )
    )

    bm25_candidate_top_k = int(
        get_required(
            hybrid_config,
            "retrieval",
            "bm25_candidate_top_k",
        )
    )

    rrf_k = int(
        get_required(
            hybrid_config,
            "fusion",
            "rrf_k",
        )
    )

    dense_weight = float(
        get_required(
            hybrid_config,
            "fusion",
            "dense_weight",
        )
    )

    bm25_weight = float(
        get_required(
            hybrid_config,
            "fusion",
            "bm25_weight",
        )
    )

    reranker_section = get_required(
        reranker_config,
        "reranker",
    )

    candidate_top_k = int(
        get_required(
            reranker_config,
            "reranker",
            "candidate_top_k",
        )
    )

    final_top_k = int(
        get_required(
            reranker_config,
            "reranker",
            "final_top_k",
        )
    )

    model_name = str(
        get_required(
            reranker_config,
            "reranker",
            "model_name",
        )
    )

    device_value = get_optional(
        reranker_config,
        "reranker",
        "device",
        default=None,
    )

    device = (
        None
        if device_value is None
        else str(device_value)
    )

    batch_size = int(
        get_optional(
            reranker_config,
            "reranker",
            "batch_size",
            default=8,
        )
    )

    max_length = int(
        get_optional(
            reranker_config,
            "reranker",
            "max_length",
            default=512,
        )
    )

    use_fp16 = bool(
        get_optional(
            reranker_config,
            "reranker",
            "use_fp16",
            default=False,
        )
    )

    if candidate_top_k <= final_top_k:
        print(
            "警告：candidate_top_k 应大于 final_top_k。"
        )
        print(
            "当前配置："
            f"candidate_top_k={candidate_top_k}, "
            f"final_top_k={final_top_k}"
        )

    chunks = build_chunks(
        bm25_config
    )

    dense_search = build_dense_search(
        dense_config
    )

    bm25_index = build_bm25_index(
        chunks,
        bm25_config,
    )

    bm25_search = build_bm25_search(
        bm25_index
    )

    hybrid_retriever = RRFHybridRetriever(
        dense_search=dense_search,
        bm25_search=bm25_search,
        dense_candidate_top_k=(
            dense_candidate_top_k
        ),
        bm25_candidate_top_k=(
            bm25_candidate_top_k
        ),

        # retrieve() 时还会显式传入 candidate_top_k。
        final_top_k=candidate_top_k,

        rrf_k=rrf_k,
        dense_weight=dense_weight,
        bm25_weight=bm25_weight,
    )

    reranker = BGEReranker(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        use_fp16=use_fp16,
    )

    pipeline_info = {
        "hybrid_config_path": str(
            hybrid_config_path
        ),
        "dense_config_path": str(
            dense_config_path
        ),
        "bm25_config_path": str(
            bm25_config_path
        ),
        "dense_candidate_top_k": (
            dense_candidate_top_k
        ),
        "bm25_candidate_top_k": (
            bm25_candidate_top_k
        ),
        "rrf_k": rrf_k,
        "dense_weight": dense_weight,
        "bm25_weight": bm25_weight,
        "candidate_top_k": candidate_top_k,
        "final_top_k": final_top_k,
        "reranker_model": model_name,
        "reranker_device": str(
            reranker.device
        ),
        "reranker_batch_size": batch_size,
        "reranker_max_length": max_length,
        "reranker_use_fp16": (
            reranker.use_fp16
        ),
    }

    return (
        hybrid_retriever,
        reranker,
        pipeline_info,
    )


def debug_pipeline(
    query: str,
    hybrid_retriever: RRFHybridRetriever,
    reranker: BGEReranker,
    *,
    candidate_top_k: int,
    final_top_k: int,
) -> None:
    """打印单个查询在重排前后的排名。"""

    print()
    print("=" * 100)
    print("RRF + Reranker 单查询调试")
    print("=" * 100)
    print(f"Query：{query}")

    hybrid_candidates = (
        hybrid_retriever.retrieve(
            query,
            top_k=candidate_top_k,
        )
    )

    reranked_documents = reranker.rerank(
        query,
        hybrid_candidates,
        top_k=final_top_k,
    )

    print()
    print(
        f"RRF 候选 Top-{candidate_top_k}"
    )
    print("-" * 100)

    for rank, document in enumerate(
        hybrid_candidates,
        start=1,
    ):
        metadata = document.metadata

        print(
            f"[{rank:02d}] "
            f"doc_id={metadata.get('doc_id')} | "
            f"chunk_id={metadata.get('chunk_id')} | "
            f"rrf={float(metadata.get('rrf_score', 0.0)):.8f} | "
            f"dense_rank={metadata.get('dense_rank')} | "
            f"bm25_rank={metadata.get('bm25_rank')}"
        )

    print()
    print(
        f"Reranker 最终 Top-{final_top_k}"
    )
    print("-" * 100)

    for rank, document in enumerate(
        reranked_documents,
        start=1,
    ):
        metadata = document.metadata

        print(
            f"[{rank:02d}] "
            f"doc_id={metadata.get('doc_id')} | "
            f"chunk_id={metadata.get('chunk_id')} | "
            f"reranker_score="
            f"{float(metadata.get('reranker_score', 0.0)):.8f} | "
            f"pre_rerank_rank="
            f"{metadata.get('pre_rerank_rank')}"
        )

        preview = (
            document.page_content[:180]
            .replace("\n", " ")
        )

        print(
            f"     {preview}"
        )

    print("=" * 100)
    print()


def evaluate(
    dataset: Sequence[dict[str, Any]],
    hybrid_retriever: RRFHybridRetriever,
    reranker: BGEReranker,
    *,
    candidate_top_k: int,
    final_top_k: int,
    save_retrieved_chunks: bool,
) -> dict[str, Any]:
    """执行完整 Reranker 评测。"""

    question_results: list[
        dict[str, Any]
    ] = []

    hit_values: list[float] = []
    recall_values: list[float] = []
    reciprocal_rank_values: list[
        float
    ] = []

    hybrid_latency_values: list[
        float
    ] = []
    reranker_latency_values: list[
        float
    ] = []
    total_latency_values: list[
        float
    ] = []

    for index, row in enumerate(
        dataset,
        start=1,
    ):
        question_id = str(
            row["id"]
        )

        question = str(
            row["question"]
        )

        answerable = bool(
            row["answerable"]
        )

        relevant_doc_ids = [
            str(doc_id)
            for doc_id in row[
                "relevant_doc_ids"
            ]
        ]

        print(
            f"[{index:02d}/{len(dataset):02d}] "
            f"{question_id}：{question}"
        )

        total_start = (
            time.perf_counter()
        )

        hybrid_start = (
            time.perf_counter()
        )

        hybrid_candidates = (
            hybrid_retriever.retrieve(
                question,
                top_k=candidate_top_k,
            )
        )

        hybrid_latency_ms = (
            time.perf_counter()
            - hybrid_start
        ) * 1000.0

        reranker_start = (
            time.perf_counter()
        )

        reranked_documents = (
            reranker.rerank(
                question,
                hybrid_candidates,
                top_k=final_top_k,
            )
        )

        reranker_latency_ms = (
            time.perf_counter()
            - reranker_start
        ) * 1000.0

        total_latency_ms = (
            time.perf_counter()
            - total_start
        ) * 1000.0

        hybrid_latency_values.append(
            hybrid_latency_ms
        )

        reranker_latency_values.append(
            reranker_latency_ms
        )

        total_latency_values.append(
            total_latency_ms
        )

        hybrid_doc_ids = [
            get_doc_id(document)
            for document in hybrid_candidates
        ]

        reranked_doc_ids = [
            get_doc_id(document)
            for document in reranked_documents
        ]

        record: dict[str, Any] = {
            "id": question_id,
            "question": question,
            "answerable": answerable,
            "category": row.get(
                "category"
            ),
            "difficulty": row.get(
                "difficulty"
            ),
            "relevant_doc_ids": (
                relevant_doc_ids
            ),
            "hybrid_candidate_doc_ids": (
                hybrid_doc_ids
            ),
            "reranked_doc_ids": (
                reranked_doc_ids
            ),
            "hybrid_latency_ms": (
                hybrid_latency_ms
            ),
            "reranker_latency_ms": (
                reranker_latency_ms
            ),
            "total_latency_ms": (
                total_latency_ms
            ),
        }

        if answerable:
            current_hit = hit_at_k(
                retrieved_doc_ids=(
                    reranked_doc_ids
                ),
                relevant_doc_ids=(
                    relevant_doc_ids
                ),
                top_k=final_top_k,
            )

            current_recall = recall_at_k(
                retrieved_doc_ids=(
                    reranked_doc_ids
                ),
                relevant_doc_ids=(
                    relevant_doc_ids
                ),
                top_k=final_top_k,
            )

            current_rr = (
                reciprocal_rank_at_k(
                    retrieved_doc_ids=(
                        reranked_doc_ids
                    ),
                    relevant_doc_ids=(
                        relevant_doc_ids
                    ),
                    top_k=final_top_k,
                )
            )

            record.update(
                {
                    f"hit@{final_top_k}": (
                        current_hit
                    ),
                    f"recall@{final_top_k}": (
                        current_recall
                    ),
                    f"rr@{final_top_k}": (
                        current_rr
                    ),
                }
            )

            hit_values.append(
                current_hit
            )

            recall_values.append(
                current_recall
            )

            reciprocal_rank_values.append(
                current_rr
            )

        else:
            record.update(
                {
                    f"hit@{final_top_k}": None,
                    f"recall@{final_top_k}": None,
                    f"rr@{final_top_k}": None,
                }
            )

        if save_retrieved_chunks:
            record[
                "hybrid_candidates"
            ] = serialize_documents(
                hybrid_candidates,
                save_text=True,
            )

            record[
                "reranked_chunks"
            ] = serialize_documents(
                reranked_documents,
                save_text=True,
            )

        question_results.append(
            record
        )

        print(
            "    "
            f"Hybrid={hybrid_latency_ms:.2f} ms | "
            f"Reranker={reranker_latency_ms:.2f} ms | "
            f"Total={total_latency_ms:.2f} ms"
        )

    answerable_count = sum(
        1
        for row in dataset
        if bool(row["answerable"])
    )

    summary = {
        "question_count": len(dataset),
        "answerable_count": (
            answerable_count
        ),
        "unanswerable_count": (
            len(dataset)
            - answerable_count
        ),
        "candidate_top_k": (
            candidate_top_k
        ),
        "top_k": final_top_k,
        f"hit@{final_top_k}": (
            average_or_none(
                hit_values
            )
        ),
        f"recall@{final_top_k}": (
            average_or_none(
                recall_values
            )
        ),
        f"mrr@{final_top_k}": (
            average_or_none(
                reciprocal_rank_values
            )
        ),
        "mean_hybrid_latency_ms": (
            average_or_none(
                hybrid_latency_values
            )
        ),
        "mean_reranker_latency_ms": (
            average_or_none(
                reranker_latency_values
            )
        ),
        "mean_total_latency_ms": (
            average_or_none(
                total_latency_values
            )
        ),
    }

    return {
        "summary": summary,
        "questions": question_results,
    }


def save_report(
    report: dict[str, Any],
    report_path: Path,
) -> None:
    """保存评测报告。"""

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )


def print_comparison(
    *,
    top_k: int,
    reranker_summary: dict[str, Any],
    dense_report_path: Path,
    bm25_report_path: Path,
    hybrid_report_path: Path,
) -> None:
    """打印四种检索方法的指标对比。"""

    dense_metrics = load_report_metrics(
        dense_report_path,
        top_k,
    )

    bm25_metrics = load_report_metrics(
        bm25_report_path,
        top_k,
    )

    hybrid_metrics = load_report_metrics(
        hybrid_report_path,
        top_k,
    )

    reranker_metrics = {
        "hit": reranker_summary[
            f"hit@{top_k}"
        ],
        "recall": reranker_summary[
            f"recall@{top_k}"
        ],
        "mrr": reranker_summary[
            f"mrr@{top_k}"
        ],
    }

    rows = [
        (
            "Dense",
            dense_metrics,
        ),
        (
            "BM25",
            bm25_metrics,
        ),
        (
            "Hybrid RRF",
            hybrid_metrics,
        ),
        (
            "RRF + Reranker",
            reranker_metrics,
        ),
    ]

    def format_value(
        value: float | None,
    ) -> str:
        if value is None:
            return "N/A"

        return f"{value:.4f}"

    print()
    print("=" * 82)
    print(
        "Dense、BM25、Hybrid 与 Reranker 对比"
    )
    print("=" * 82)

    print(
        f"{'Method':<22}"
        f"{f'Hit@{top_k}':>14}"
        f"{f'Recall@{top_k}':>14}"
        f"{f'MRR@{top_k}':>14}"
    )

    print("-" * 82)

    for method_name, metrics in rows:
        if metrics is None:
            hit_value = None
            recall_value = None
            mrr_value = None
        else:
            hit_value = metrics["hit"]
            recall_value = metrics["recall"]
            mrr_value = metrics["mrr"]

        print(
            f"{method_name:<22}"
            f"{format_value(hit_value):>14}"
            f"{format_value(recall_value):>14}"
            f"{format_value(mrr_value):>14}"
        )

    print("=" * 82)


def main() -> None:
    """程序入口。"""

    args = parse_args()

    reranker_config_path = Path(
        args.config
    ).resolve()

    reranker_config = load_yaml(
        reranker_config_path
    )

    experiment_name = str(
        reranker_config.get(
            "experiment_name",
            "hybrid_rrf_bge_reranker",
        )
    )

    candidate_top_k = int(
        get_required(
            reranker_config,
            "reranker",
            "candidate_top_k",
        )
    )

    final_top_k = int(
        get_required(
            reranker_config,
            "reranker",
            "final_top_k",
        )
    )

    report_path = resolve_project_path(
        get_required(
            reranker_config,
            "outputs",
            "report_path",
        )
    )

    save_retrieved_chunks = bool(
        get_optional(
            reranker_config,
            "outputs",
            "save_retrieved_chunks",
            default=True,
        )
    )

    dense_report_path = resolve_project_path(
        get_optional(
            reranker_config,
            "comparison",
            "dense_report_path",
            default=(
                "reports/"
                "baseline_results.json"
            ),
        )
    )

    bm25_report_path = resolve_project_path(
        get_optional(
            reranker_config,
            "comparison",
            "bm25_report_path",
            default=(
                "reports/"
                "bm25_results.json"
            ),
        )
    )

    hybrid_report_path = resolve_project_path(
        get_optional(
            reranker_config,
            "comparison",
            "hybrid_report_path",
            default=(
                "reports/"
                "hybrid_results.json"
            ),
        )
    )

    print("=" * 82)
    print(
        "RAGLab Hybrid RRF + BGE Reranker 评测"
    )
    print("=" * 82)
    print(
        f"实验名称：{experiment_name}"
    )
    print(
        f"配置文件：{reranker_config_path}"
    )
    print(
        f"评测数据：{EVAL_DATASET_PATH}"
    )
    print(
        f"Hybrid 候选数：{candidate_top_k}"
    )
    print(
        f"最终 Top-K：{final_top_k}"
    )
    print("=" * 82)

    dataset = load_eval_dataset()

    print(
        f"评测问题数：{len(dataset)}"
    )

    (
        hybrid_retriever,
        reranker,
        pipeline_info,
    ) = build_pipeline(
        reranker_config
    )

    if args.debug_query:
        debug_pipeline(
            args.debug_query,
            hybrid_retriever,
            reranker,
            candidate_top_k=(
                candidate_top_k
            ),
            final_top_k=final_top_k,
        )

    evaluation_result = evaluate(
        dataset,
        hybrid_retriever,
        reranker,
        candidate_top_k=(
            candidate_top_k
        ),
        final_top_k=final_top_k,
        save_retrieved_chunks=(
            save_retrieved_chunks
        ),
    )

    report = {
        "experiment_name": (
            experiment_name
        ),
        "config": {
            "reranker_config_path": str(
                reranker_config_path
            ),
            "evaluation_dataset_path": str(
                EVAL_DATASET_PATH
            ),
            **pipeline_info,
        },
        **evaluation_result,
    }

    save_report(
        report,
        report_path,
    )

    summary = report["summary"]

    print()
    print("=" * 82)
    print(
        "RRF + Reranker 评测完成"
    )
    print("=" * 82)

    print(
        f"Hit@{final_top_k}："
        f"{summary[f'hit@{final_top_k}']:.4f}"
    )

    print(
        f"Recall@{final_top_k}："
        f"{summary[f'recall@{final_top_k}']:.4f}"
    )

    print(
        f"MRR@{final_top_k}："
        f"{summary[f'mrr@{final_top_k}']:.4f}"
    )

    print(
        "Hybrid 平均耗时："
        f"{summary['mean_hybrid_latency_ms']:.2f} ms"
    )

    print(
        "Reranker 平均耗时："
        f"{summary['mean_reranker_latency_ms']:.2f} ms"
    )

    print(
        "总平均耗时："
        f"{summary['mean_total_latency_ms']:.2f} ms"
    )

    print(
        f"报告位置：{report_path}"
    )

    print_comparison(
        top_k=final_top_k,
        reranker_summary=summary,
        dense_report_path=(
            dense_report_path
        ),
        bm25_report_path=(
            bm25_report_path
        ),
        hybrid_report_path=(
            hybrid_report_path
        ),
    )


if __name__ == "__main__":
    main()