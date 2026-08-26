"""批量评测 RAGLab 基线 Dense Retrieval。

执行流程：

eval_dataset.json
→ 逐条读取问题
→ 查询持久化 Chroma
→ 获取 Top-K Chunk
→ 提取 retrieved doc_id
→ 计算 Hit@K、Recall@K、MRR@K
→ 保存 JSON 报告

当前只评测检索，不调用大语言模型。
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
from raglab.embeddings.factory import (
    create_huggingface_embeddings,
)
from raglab.retrieval.dense import (
    dense_similarity_search_with_score,
)
from raglab.settings import (
    BASELINE_CONFIG_PATH,
    EVAL_DATASET_PATH,
    PROJECT_ROOT,
)
from raglab.vectorstores.chroma_store import (
    open_chroma_store,
)


def load_baseline_config() -> dict[str, Any]:
    """读取 baseline.yaml。"""
    with BASELINE_CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "baseline.yaml 的根节点必须是字典。"
        )

    return config


def load_eval_dataset() -> list[dict[str, Any]]:
    """读取检索评测数据集。"""
    if not EVAL_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"评测集不存在：{EVAL_DATASET_PATH}"
        )

    with EVAL_DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        dataset = json.load(file)

    if not isinstance(dataset, list):
        raise ValueError(
            "eval_dataset.json 的根节点必须是列表。"
        )

    for index, row in enumerate(dataset):
        if not isinstance(row, dict):
            raise ValueError(
                "评测集中的每条记录必须是字典，"
                f"错误位置：{index}"
            )

        required_fields = {
            "id",
            "question",
            "relevant_doc_ids",
            "answerable",
        }

        missing_fields = (
            required_fields - row.keys()
        )

        if missing_fields:
            raise ValueError(
                f"评测记录 {row.get('id', index)} "
                f"缺少字段：{sorted(missing_fields)}"
            )

    return dataset


def resolve_project_path(
    path_value: str,
) -> Path:
    """将配置中的相对路径转成项目绝对路径。"""
    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def require_document_metadata(
    document: Document,
    key: str,
) -> str:
    """读取 Document Metadata 中的必需字符串字段。"""
    value = document.metadata.get(key)

    if not isinstance(value, str):
        raise ValueError(
            f"检索结果缺少有效的 {key}："
            f"{document.metadata}"
        )

    cleaned_value = value.strip()

    if not cleaned_value:
        raise ValueError(
            f"检索结果中的 {key} 不能为空。"
        )

    return cleaned_value


def serialize_retrieved_chunks(
    results: list[
        tuple[Document, float]
    ],
) -> list[dict[str, Any]]:
    """将检索结果转换成可写入 JSON 的结构。"""
    serialized_results: list[
        dict[str, Any]
    ] = []

    for rank, (
        document,
        distance,
    ) in enumerate(
        results,
        start=1,
    ):
        serialized_results.append(
            {
                "rank": rank,
                "distance": float(distance),
                "chunk_id": (
                    document.metadata.get(
                        "chunk_id"
                    )
                ),
                "doc_id": (
                    document.metadata.get(
                        "doc_id"
                    )
                ),
                "title": (
                    document.metadata.get(
                        "title"
                    )
                ),
                "page_number": (
                    document.metadata.get(
                        "page_number"
                    )
                ),
                "text": document.page_content,
            }
        )

    return serialized_results


def average_or_none(
    values: list[float],
) -> float | None:
    """计算平均值；空列表返回 None。"""
    if not values:
        return None

    return float(fmean(values))


def main() -> None:
    """执行完整的批量检索评测。"""
    config = load_baseline_config()
    dataset = load_eval_dataset()

    embedding_config = config["embedding"]
    vector_store_config = config[
        "vector_store"
    ]
    retrieval_config = config["retrieval"]
    outputs_config = config["outputs"]

    experiment_name = str(
        config["experiment_name"]
    )

    model_name = str(
        embedding_config["model"]
    )

    normalize_embeddings = bool(
        embedding_config[
            "normalize_embeddings"
        ]
    )

    collection_name = str(
        vector_store_config[
            "collection_name"
        ]
    )

    persist_directory = resolve_project_path(
        str(
            vector_store_config[
                "persist_directory"
            ]
        )
    )

    top_k = int(
        retrieval_config["top_k"]
    )

    metadata_filter = retrieval_config.get(
        "metadata_filter"
    )

    report_path = resolve_project_path(
        str(
            outputs_config["report_path"]
        )
    )

    save_retrieved_chunks = bool(
        outputs_config.get(
            "save_retrieved_chunks",
            True,
        )
    )

    if (
        not persist_directory.exists()
        or not any(persist_directory.iterdir())
    ):
        raise FileNotFoundError(
            "没有找到 Chroma 索引。\n"
            "请先运行：\n"
            "python -m scripts.build_index\n"
            f"目录：{persist_directory}"
        )

    print("=" * 70)
    print("RAGLab Dense Retrieval 基线评测")
    print("=" * 70)

    print(f"实验名称：{experiment_name}")
    print(f"评测问题数：{len(dataset)}")
    print(f"Top-K：{top_k}")
    print(f"Embedding 模型：{model_name}")
    print(f"Collection：{collection_name}")

    print("\n正在加载 Embedding 模型……")

    embeddings = create_huggingface_embeddings(
        model_name=model_name,
        normalize_embeddings=normalize_embeddings,
        device="cpu",
    )

    vector_store = open_chroma_store(
        embeddings=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    question_results: list[
        dict[str, Any]
    ] = []

    hit_values: list[float] = []
    recall_values: list[float] = []
    reciprocal_rank_values: list[float] = []

    answerable_top1_distances: list[
        float
    ] = []

    unanswerable_top1_distances: list[
        float
    ] = []

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

        results = (
            dense_similarity_search_with_score(
                vector_store=vector_store,
                query=question,
                top_k=top_k,
                metadata_filter=metadata_filter,
            )
        )

        retrieved_doc_ids = [
            require_document_metadata(
                document=document,
                key="doc_id",
            )
            for document, _ in results
        ]

        top1_distance: float | None = None

        if results:
            top1_distance = float(
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
            "relevant_section_ids": (
                row.get(
                    "relevant_section_ids",
                    [],
                )
            ),
            "retrieved_doc_ids": (
                retrieved_doc_ids
            ),
            "top1_distance": top1_distance,
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

            if top1_distance is not None:
                answerable_top1_distances.append(
                    top1_distance
                )

        else:
            # 无答案问题没有 relevant_doc_ids，
            # 因此不计算 Hit、Recall 和 RR。
            #
            # 但保留 Top-1 距离，
            # 后面用于分析相关性阈值。
            record.update(
                {
                    f"hit@{top_k}": None,
                    f"recall@{top_k}": None,
                    f"rr@{top_k}": None,
                }
            )

            if top1_distance is not None:
                unanswerable_top1_distances.append(
                    top1_distance
                )

        if save_retrieved_chunks:
            record["retrieved_chunks"] = (
                serialize_retrieved_chunks(
                    results
                )
            )

        question_results.append(record)

    answerable_count = sum(
        1
        for row in dataset
        if bool(row["answerable"])
    )

    unanswerable_count = (
        len(dataset) - answerable_count
    )

    summary = {
        "experiment_name": experiment_name,
        "question_count": len(dataset),
        "answerable_count": answerable_count,
        "unanswerable_count": (
            unanswerable_count
        ),
        "top_k": top_k,
        f"hit@{top_k}": (
            average_or_none(hit_values)
        ),
        f"recall@{top_k}": (
            average_or_none(recall_values)
        ),
        f"mrr@{top_k}": (
            average_or_none(
                reciprocal_rank_values
            )
        ),
        "mean_top1_distance_answerable": (
            average_or_none(
                answerable_top1_distances
            )
        ),
        "mean_top1_distance_unanswerable": (
            average_or_none(
                unanswerable_top1_distances
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
    print("评测完成")
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

    answerable_distance = summary[
        "mean_top1_distance_answerable"
    ]

    unanswerable_distance = summary[
        "mean_top1_distance_unanswerable"
    ]

    if answerable_distance is not None:
        print(
            "可回答问题平均 Top-1 distance："
            f"{answerable_distance:.6f}"
        )

    if unanswerable_distance is not None:
        print(
            "不可回答问题平均 Top-1 distance："
            f"{unanswerable_distance:.6f}"
        )

    print(f"报告位置：{report_path}")


if __name__ == "__main__":
    main()