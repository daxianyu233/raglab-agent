"""批量评测 BM25 Retrieval。

流程：

PDF
→ 页面 Document
→ Chunk
→ BM25 索引
→ 逐条检索评测问题
→ 计算 Hit@K、Recall@K、MRR@K
→ 保存评测报告
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean
from typing import Any

import yaml
from langchain_core.documents import Document

from evaluation.metrics import (
    hit_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from raglab.ingestion.loaders import (
    load_pdf_corpus,
)
from raglab.ingestion.splitters import (
    create_recursive_splitter,
    split_page_documents,
)
from raglab.retrieval.bm25 import (
    BM25SearchIndex,
)
from raglab.settings import (
    CONFIG_DIR,
    EVAL_DATASET_PATH,
    PROJECT_ROOT,
)


BM25_CONFIG_PATH = CONFIG_DIR / "bm25.yaml"


def load_yaml_config(
    config_path: Path,
) -> dict[str, Any]:
    """读取 YAML 配置。"""
    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"配置文件根节点必须是字典：{config_path}"
        )

    return config


def load_eval_dataset() -> list[dict[str, Any]]:
    """读取评测数据集。"""
    with EVAL_DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        dataset = json.load(file)

    if not isinstance(dataset, list):
        raise ValueError(
            "eval_dataset.json 根节点必须是列表。"
        )

    return dataset


def resolve_project_path(
    path_value: str,
) -> Path:
    """将项目相对路径转换为绝对路径。"""
    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def get_doc_id(
    document: Document,
) -> str:
    """读取检索结果中的 doc_id。"""
    doc_id = document.metadata.get("doc_id")

    if not isinstance(doc_id, str):
        raise ValueError(
            "检索结果缺少有效 doc_id："
            f"{document.metadata}"
        )

    return doc_id


def serialize_results(
    results: list[tuple[Document, float]],
) -> list[dict[str, Any]]:
    """将 BM25 结果转换成 JSON 数据。"""
    serialized: list[dict[str, Any]] = []

    for rank, (
        document,
        score,
    ) in enumerate(
        results,
        start=1,
    ):
        serialized.append(
            {
                "rank": rank,
                "bm25_score": float(score),
                "chunk_id": document.metadata.get(
                    "chunk_id"
                ),
                "doc_id": document.metadata.get(
                    "doc_id"
                ),
                "title": document.metadata.get(
                    "title"
                ),
                "page_number": document.metadata.get(
                    "page_number"
                ),
                "text": document.page_content,
            }
        )

    return serialized


def average_or_none(
    values: list[float],
) -> float | None:
    """计算平均值。"""
    if not values:
        return None

    return float(fmean(values))


def main() -> None:
    config = load_yaml_config(
        BM25_CONFIG_PATH
    )
    dataset = load_eval_dataset()

    splitter_config = config["splitter"]
    retrieval_config = config["retrieval"]
    outputs_config = config["outputs"]

    experiment_name = str(
        config["experiment_name"]
    )

    chunk_size = int(
        splitter_config["chunk_size"]
    )
    chunk_overlap = int(
        splitter_config["chunk_overlap"]
    )

    top_k = int(
        retrieval_config["top_k"]
    )
    k1 = float(
        retrieval_config["k1"]
    )
    b = float(
        retrieval_config["b"]
    )

    report_path = resolve_project_path(
        str(outputs_config["report_path"])
    )

    save_retrieved_chunks = bool(
        outputs_config.get(
            "save_retrieved_chunks",
            True,
        )
    )

    print("=" * 70)
    print("RAGLab BM25 Retrieval 评测")
    print("=" * 70)
    print(f"实验名称：{experiment_name}")
    print(f"评测问题数：{len(dataset)}")
    print(f"Top-K：{top_k}")
    print(f"k1：{k1}")
    print(f"b：{b}")

    # 1. 使用和 Dense 基线相同的 PDF 与切块参数。
    print("\n正在加载并切分语料……")

    page_documents = load_pdf_corpus()

    splitter = create_recursive_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = split_page_documents(
        documents=page_documents,
        text_splitter=splitter,
    )

    print(f"页面数量：{len(page_documents)}")
    print(f"Chunk 数量：{len(chunks)}")

    # 2. 建立 BM25 内存索引。
    print("正在建立 BM25 索引……")

    bm25_index = BM25SearchIndex(
        documents=chunks,
        k1=k1,
        b=b,
    )

    question_results: list[
        dict[str, Any]
    ] = []

    hit_values: list[float] = []
    recall_values: list[float] = []
    reciprocal_rank_values: list[float] = []

    answerable_top1_scores: list[
        float
    ] = []
    unanswerable_top1_scores: list[
        float
    ] = []

    # 3. 逐条执行检索。
    for index, row in enumerate(
        dataset,
        start=1,
    ):
        question_id = str(row["id"])
        question = str(row["question"])
        answerable = bool(row["answerable"])

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

        results = bm25_index.search(
            query=question,
            top_k=top_k,
        )

        retrieved_doc_ids = [
            get_doc_id(document)
            for document, _ in results
        ]

        top1_score: float | None = None

        if results:
            top1_score = float(
                results[0][1]
            )

        record: dict[str, Any] = {
            "id": question_id,
            "question": question,
            "answerable": answerable,
            "category": row.get("category"),
            "difficulty": row.get(
                "difficulty"
            ),
            "relevant_doc_ids": (
                relevant_doc_ids
            ),
            "retrieved_doc_ids": (
                retrieved_doc_ids
            ),
            "top1_bm25_score": top1_score,
        }

        if answerable:
            current_hit = hit_at_k(
                retrieved_doc_ids=(
                    retrieved_doc_ids
                ),
                relevant_doc_ids=(
                    relevant_doc_ids
                ),
                top_k=top_k,
            )

            current_recall = recall_at_k(
                retrieved_doc_ids=(
                    retrieved_doc_ids
                ),
                relevant_doc_ids=(
                    relevant_doc_ids
                ),
                top_k=top_k,
            )

            current_rr = reciprocal_rank_at_k(
                retrieved_doc_ids=(
                    retrieved_doc_ids
                ),
                relevant_doc_ids=(
                    relevant_doc_ids
                ),
                top_k=top_k,
            )

            record.update(
                {
                    f"hit@{top_k}": current_hit,
                    f"recall@{top_k}": (
                        current_recall
                    ),
                    f"rr@{top_k}": current_rr,
                }
            )

            hit_values.append(current_hit)
            recall_values.append(
                current_recall
            )
            reciprocal_rank_values.append(
                current_rr
            )

            if top1_score is not None:
                answerable_top1_scores.append(
                    top1_score
                )

        else:
            record.update(
                {
                    f"hit@{top_k}": None,
                    f"recall@{top_k}": None,
                    f"rr@{top_k}": None,
                }
            )

            if top1_score is not None:
                unanswerable_top1_scores.append(
                    top1_score
                )

        if save_retrieved_chunks:
            record["retrieved_chunks"] = (
                serialize_results(results)
            )

        question_results.append(record)

    answerable_count = sum(
        1
        for row in dataset
        if bool(row["answerable"])
    )

    summary = {
        "experiment_name": experiment_name,
        "question_count": len(dataset),
        "answerable_count": answerable_count,
        "unanswerable_count": (
            len(dataset) - answerable_count
        ),
        "top_k": top_k,
        f"hit@{top_k}": average_or_none(
            hit_values
        ),
        f"recall@{top_k}": average_or_none(
            recall_values
        ),
        f"mrr@{top_k}": average_or_none(
            reciprocal_rank_values
        ),
        "mean_top1_score_answerable": (
            average_or_none(
                answerable_top1_scores
            )
        ),
        "mean_top1_score_unanswerable": (
            average_or_none(
                unanswerable_top1_scores
            )
        ),
    }

    report = {
        "summary": summary,
        "questions": question_results,
    }

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

    print("\n" + "=" * 70)
    print("BM25 评测完成")
    print("=" * 70)

    print(
        f"Hit@{top_k}："
        f"{summary[f'hit@{top_k}']:.4f}"
    )
    print(
        f"Recall@{top_k}："
        f"{summary[f'recall@{top_k}']:.4f}"
    )
    print(
        f"MRR@{top_k}："
        f"{summary[f'mrr@{top_k}']:.4f}"
    )

    answerable_score = summary[
        "mean_top1_score_answerable"
    ]
    unanswerable_score = summary[
        "mean_top1_score_unanswerable"
    ]

    if answerable_score is not None:
        print(
            "可回答问题平均 Top-1 BM25 score："
            f"{answerable_score:.6f}"
        )

    if unanswerable_score is not None:
        print(
            "不可回答问题平均 Top-1 BM25 score："
            f"{unanswerable_score:.6f}"
        )

    print(f"报告位置：{report_path}")


if __name__ == "__main__":
    main()