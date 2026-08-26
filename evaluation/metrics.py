"""RAG 检索评测指标。

当前指标基于 doc_id 计算：

1. Hit@K
2. Recall@K
3. MRR@K

输入示例：

retrieved_doc_ids = [
    "DOC-LC-002",
    "DOC-LC-001",
    "DOC-RAG-001",
]

relevant_doc_ids = [
    "DOC-LC-001",
]
"""

from __future__ import annotations

from collections.abc import Sequence


def validate_top_k(top_k: int) -> None:
    """检查 Top-K 参数。"""
    if top_k <= 0:
        raise ValueError(
            f"top_k 必须大于 0，当前值：{top_k}"
        )


def normalize_ids(
    ids: Sequence[str],
) -> list[str]:
    """清理 ID，去除空字符串。

    注意：
        这里不去重，因为检索结果中的排名顺序
        和重复结果本身也可能有分析价值。
    """
    normalized_ids: list[str] = []

    for item in ids:
        cleaned_id = str(item).strip()

        if cleaned_id:
            normalized_ids.append(cleaned_id)

    return normalized_ids


def hit_at_k(
    retrieved_doc_ids: Sequence[str],
    relevant_doc_ids: Sequence[str],
    top_k: int,
) -> float:
    """计算 Hit@K。

    只要前 K 个检索结果中至少出现一个相关文档，
    就记为 1，否则记为 0。

    示例：

        检索结果：
            DOC-A
            DOC-B
            DOC-C

        相关文档：
            DOC-B

        Hit@3 = 1

    Returns:
        float:
            1.0 或 0.0。
    """
    validate_top_k(top_k)

    retrieved = normalize_ids(
        retrieved_doc_ids
    )[:top_k]

    relevant = set(
        normalize_ids(relevant_doc_ids)
    )

    if not relevant:
        raise ValueError(
            "相关文档列表为空，无法计算 Hit@K。"
        )

    return float(
        any(
            doc_id in relevant
            for doc_id in retrieved
        )
    )


def recall_at_k(
    retrieved_doc_ids: Sequence[str],
    relevant_doc_ids: Sequence[str],
    top_k: int,
) -> float:
    """计算 Recall@K。

    计算前 K 个结果覆盖了多少相关文档。

    公式可以理解为：

        被召回的相关文档数量
        --------------------
        全部相关文档数量

    示例：

        相关文档：
            DOC-A
            DOC-B

        Top-5 中只出现 DOC-A：

            Recall@5 = 1 / 2 = 0.5

    注意：
        这里按 doc_id 去重。

        同一篇文档的多个 Chunk 被检索出来，
        只算召回了一个相关文档。
    """
    validate_top_k(top_k)

    retrieved = set(
        normalize_ids(
            retrieved_doc_ids
        )[:top_k]
    )

    relevant = set(
        normalize_ids(relevant_doc_ids)
    )

    if not relevant:
        raise ValueError(
            "相关文档列表为空，无法计算 Recall@K。"
        )

    matched = retrieved.intersection(
        relevant
    )

    return len(matched) / len(relevant)


def reciprocal_rank_at_k(
    retrieved_doc_ids: Sequence[str],
    relevant_doc_ids: Sequence[str],
    top_k: int,
) -> float:
    """计算 Reciprocal Rank@K。

    找到第一个相关结果所在的排名，
    然后计算其倒数。

    示例：

        第一名就是相关文档：
            RR = 1 / 1 = 1.0

        第二名才是相关文档：
            RR = 1 / 2 = 0.5

        第五名才是相关文档：
            RR = 1 / 5 = 0.2

        Top-K 中没有相关文档：
            RR = 0.0

    多个问题的 RR 平均值就是 MRR。
    """
    validate_top_k(top_k)

    retrieved = normalize_ids(
        retrieved_doc_ids
    )[:top_k]

    relevant = set(
        normalize_ids(relevant_doc_ids)
    )

    if not relevant:
        raise ValueError(
            "相关文档列表为空，无法计算 RR@K。"
        )

    for rank, doc_id in enumerate(
        retrieved,
        start=1,
    ):
        if doc_id in relevant:
            return 1.0 / rank

    return 0.0