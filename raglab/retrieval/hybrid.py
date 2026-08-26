from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from langchain_core.documents import Document


# 检索函数统一接口：
# 输入 query 和 top_k，返回 Document、(Document, score) 或类似结果。
SearchFunction = Callable[[str, int], Sequence[Any]]


@dataclass(frozen=True)
class NormalizedResult:
    """统一 Dense、BM25 检索结果的内部表示。"""

    document: Document
    raw_score: float | None = None


def _safe_float(value: Any) -> float | None:
    """尽可能将原始分数转换为 float。"""

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_result(item: Any) -> NormalizedResult:
    """
    将不同检索器的返回结果转换为 NormalizedResult。

    当前支持：
    1. Document
    2. (Document, score)
    3. {"document": Document, "score": ...}
    4. {"doc": Document, "score": ...}
    5. 具有 document/doc 和 score 属性的对象
    """

    if isinstance(item, Document):
        return NormalizedResult(document=item)

    if isinstance(item, tuple) and len(item) >= 2:
        document, score = item[0], item[1]

        if not isinstance(document, Document):
            raise TypeError(
                "元组型检索结果的第一个元素必须是 Document，"
                f"实际得到：{type(document)!r}"
            )

        return NormalizedResult(
            document=document,
            raw_score=_safe_float(score),
        )

    if isinstance(item, dict):
        document = item.get("document")
        if document is None:
            document = item.get("doc")

        if isinstance(document, Document):
            raw_score = item.get("score")

            if raw_score is None:
                raw_score = item.get("distance")

            return NormalizedResult(
                document=document,
                raw_score=_safe_float(raw_score),
            )

    document = getattr(item, "document", None)
    if document is None:
        document = getattr(item, "doc", None)

    if isinstance(document, Document):
        raw_score = getattr(item, "score", None)

        if raw_score is None:
            raw_score = getattr(item, "distance", None)

        return NormalizedResult(
            document=document,
            raw_score=_safe_float(raw_score),
        )

    raise TypeError(
        "无法识别检索结果类型。期望 Document、(Document, score)、"
        f"字典或带 document 属性的对象，实际得到：{type(item)!r}"
    )


def get_chunk_id(document: Document) -> str:
    """读取并校验稳定的 chunk_id。"""

    chunk_id = document.metadata.get("chunk_id")

    if chunk_id is None:
        raise KeyError(
            "Document metadata 中缺少 chunk_id。"
            "Dense 与 BM25 必须使用同一批 Chunk 和相同的 chunk_id。"
        )

    chunk_id = str(chunk_id).strip()

    if not chunk_id:
        raise ValueError("Document metadata 中的 chunk_id 不能为空。")

    return chunk_id


def make_search_function(
    retriever: Any,
    method_name: str | None = None,
) -> SearchFunction:
    """
    将已有检索器对象包装为统一的 search(query, top_k) 接口。

    自动尝试的方法顺序：
    retrieve -> search -> similarity_search -> invoke

    也可以通过 method_name 显式指定。
    """

    candidate_names = (
        [method_name]
        if method_name is not None
        else ["retrieve", "search", "similarity_search", "invoke"]
    )

    method = None

    for name in candidate_names:
        candidate = getattr(retriever, name, None)

        if callable(candidate):
            method = candidate
            break

    if method is None:
        if callable(retriever):
            method = retriever
        else:
            raise TypeError(
                "检索器必须是可调用对象，或实现 retrieve/search/"
                "similarity_search/invoke 方法。"
            )

    def search(query: str, top_k: int) -> Sequence[Any]:
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0。")

        parameters = inspect.signature(method).parameters

        if "top_k" in parameters:
            return method(query, top_k=top_k)

        if "k" in parameters:
            return method(query, k=top_k)

        if "n_results" in parameters:
            return method(query, n_results=top_k)

        # 某些 LangChain Retriever 的 k 已经在初始化时配置，
        # invoke 只接收 query。
        return method(query)

    return search


def rrf_fuse(
    dense_results: Sequence[Any],
    bm25_results: Sequence[Any],
    *,
    top_k: int,
    rrf_k: int = 60,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
) -> list[Document]:
    """
    使用 Reciprocal Rank Fusion 融合 Dense 与 BM25 结果。

    原始 Dense distance 和 BM25 score 仅保留为调试信息，
    不参与 RRF 分数计算。
    """

    if top_k <= 0:
        raise ValueError("top_k 必须大于 0。")

    if rrf_k < 0:
        raise ValueError("rrf_k 不能小于 0。")

    fused: dict[str, dict[str, Any]] = {}

    sources = (
        ("dense", dense_results, float(dense_weight)),
        ("bm25", bm25_results, float(bm25_weight)),
    )

    for source_name, source_results, source_weight in sources:
        # 防止同一个检索器内部意外返回重复 chunk。
        seen_in_source: set[str] = set()

        for rank, raw_item in enumerate(source_results, start=1):
            item = normalize_result(raw_item)
            document = item.document
            chunk_id = get_chunk_id(document)

            if chunk_id in seen_in_source:
                continue

            seen_in_source.add(chunk_id)

            if chunk_id not in fused:
                fused[chunk_id] = {
                    "document": document,
                    "chunk_id": chunk_id,
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "bm25_rank": None,
                    "dense_raw_score": None,
                    "bm25_raw_score": None,
                }

            candidate = fused[chunk_id]

            candidate["rrf_score"] += source_weight / (rrf_k + rank)
            candidate[f"{source_name}_rank"] = rank
            candidate[f"{source_name}_raw_score"] = item.raw_score

    def sort_key(candidate: dict[str, Any]) -> tuple[float, int, str]:
        dense_rank = candidate["dense_rank"]
        bm25_rank = candidate["bm25_rank"]

        valid_ranks = [
            rank
            for rank in (dense_rank, bm25_rank)
            if rank is not None
        ]

        best_source_rank = min(valid_ranks) if valid_ranks else 10**9

        # RRF 分数降序；
        # 分数相同时，单路最好名次升序；
        # 仍相同时，以 chunk_id 保证结果稳定。
        return (
            -candidate["rrf_score"],
            best_source_rank,
            candidate["chunk_id"],
        )

    ranked_candidates = sorted(fused.values(), key=sort_key)

    output: list[Document] = []

    for candidate in ranked_candidates[:top_k]:
        original_document: Document = candidate["document"]

        metadata = dict(original_document.metadata)
        metadata.update(
            {
                "retrieval_method": "hybrid_rrf",
                "rrf_score": candidate["rrf_score"],
                "dense_rank": candidate["dense_rank"],
                "bm25_rank": candidate["bm25_rank"],
                "dense_raw_score": candidate["dense_raw_score"],
                "bm25_raw_score": candidate["bm25_raw_score"],
            }
        )

        output.append(
            Document(
                page_content=original_document.page_content,
                metadata=metadata,
            )
        )

    return output


class RRFHybridRetriever:
    """Dense + BM25 的 RRF 混合检索器。"""

    def __init__(
        self,
        dense_search: SearchFunction,
        bm25_search: SearchFunction,
        *,
        dense_candidate_top_k: int = 20,
        bm25_candidate_top_k: int = 20,
        final_top_k: int = 5,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ) -> None:
        if dense_candidate_top_k <= 0:
            raise ValueError("dense_candidate_top_k 必须大于 0。")

        if bm25_candidate_top_k <= 0:
            raise ValueError("bm25_candidate_top_k 必须大于 0。")

        if final_top_k <= 0:
            raise ValueError("final_top_k 必须大于 0。")

        self.dense_search = dense_search
        self.bm25_search = bm25_search

        self.dense_candidate_top_k = dense_candidate_top_k
        self.bm25_candidate_top_k = bm25_candidate_top_k
        self.final_top_k = final_top_k

        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[Document]:
        query = query.strip()

        if not query:
            return []

        dense_results = self.dense_search(
            query,
            self.dense_candidate_top_k,
        )

        bm25_results = self.bm25_search(
            query,
            self.bm25_candidate_top_k,
        )

        return rrf_fuse(
            dense_results=dense_results,
            bm25_results=bm25_results,
            top_k=top_k or self.final_top_k,
            rrf_k=self.rrf_k,
            dense_weight=self.dense_weight,
            bm25_weight=self.bm25_weight,
        )

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[Document]:
        """retrieve 的别名，方便兼容已有工程接口。"""

        return self.retrieve(query=query, top_k=top_k)