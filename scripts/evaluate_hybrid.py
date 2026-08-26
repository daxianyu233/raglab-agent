"""评测 Dense + BM25 + RRF Hybrid Retrieval。

执行流程：

hybrid.yaml
→ 读取 Dense 与 BM25 配置
→ 打开持久化 Chroma
→ 重新加载 PDF 并按原参数构建 Chunk
→ 构建 BM25 内存索引
→ Dense Top-N + BM25 Top-N
→ 按 chunk_id 去重
→ RRF 融合
→ 计算 Hit@5、Recall@5、MRR@5
→ 保存 reports/hybrid_results.json
→ 与 Dense、BM25 报告进行对比
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

import yaml
from langchain_core.documents import Document

from raglab.embeddings.factory import (
    create_huggingface_embeddings,
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
from raglab.retrieval.hybrid import (
    RRFHybridRetriever,
    make_search_function,
)
from raglab.settings import (
    EVAL_DATASET_PATH,
    PROJECT_ROOT,
)
from raglab.vectorstores.chroma_store import (
    open_chroma_store,
)


DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "hybrid.yaml"
)


def parse_args() -> argparse.Namespace:
    """读取命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Dense + BM25 "
            "RRF Hybrid Retrieval."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="hybrid.yaml 的路径。",
    )

    parser.add_argument(
        "--debug-query",
        type=str,
        default=None,
        help=(
            "可选。打印一个查询的 Dense、BM25 "
            "和 RRF 排名信息，然后继续完整评测。"
        ),
    )

    return parser.parse_args()


def load_yaml(
    path: Path,
) -> dict[str, Any]:
    """读取 YAML 配置。"""

    if not path.exists():
        raise FileNotFoundError(
            f"配置文件不存在：{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"YAML 根节点必须是字典：{path}"
        )

    return data


def resolve_project_path(
    path_value: str | Path,
) -> Path:
    """将相对路径解析为相对于项目根目录的绝对路径。"""

    path = Path(path_value)

    if path.is_absolute():
        return path

    return (
        PROJECT_ROOT
        / path
    ).resolve()


def get_required(
    mapping: dict[str, Any],
    *keys: str,
) -> Any:
    """读取必需配置项。"""

    current: Any = mapping

    for key in keys:
        if (
            not isinstance(current, dict)
            or key not in current
        ):
            dotted_key = ".".join(keys)

            raise KeyError(
                f"配置中缺少必需字段：{dotted_key}"
            )

        current = current[key]

    return current


def get_optional(
    mapping: dict[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    """读取可选配置项。"""

    current: Any = mapping

    for key in keys:
        if (
            not isinstance(current, dict)
            or key not in current
        ):
            return default

        current = current[key]

    return current


def resolve_eval_dataset_path(
    hybrid_config: dict[str, Any],
) -> Path:
    """确定评测集路径。

    优先读取：

    paths.evaluation_dataset

    如果该字段为 null，则从 raglab.settings 中读取：

    EVAL_DATASET_PATH
    """

    configured_path = get_optional(
        hybrid_config,
        "paths",
        "evaluation_dataset",
        default=None,
    )

    if configured_path:
        return resolve_project_path(
            configured_path
        )

    try:
        from raglab.settings import (
            EVAL_DATASET_PATH,
        )
    except ImportError as exc:
        raise RuntimeError(
            "hybrid.yaml 中 "
            "paths.evaluation_dataset 为 null，"
            "但是 raglab.settings 中没有 "
            "EVAL_DATASET_PATH。\n"
            "请在 hybrid.yaml 中填写现有评测集的路径。"
        ) from exc

    return Path(
        EVAL_DATASET_PATH
    ).resolve()


def load_evaluation_items(
    path: Path,
) -> list[dict[str, Any]]:
    """加载检索评测集。"""

    if not path.exists():
        raise FileNotFoundError(
            f"评测集不存在：{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if isinstance(data, list):
        items = data

    elif isinstance(data, dict):
        items = None

        for key in (
            "items",
            "questions",
            "samples",
            "data",
            "cases",
        ):
            candidate = data.get(key)

            if isinstance(candidate, list):
                items = candidate
                break

        if items is None:
            raise ValueError(
                "评测集 JSON 根节点是字典，"
                "但是没有找到 items、questions、"
                "samples、data 或 cases 列表。"
            )

    else:
        raise ValueError(
            "评测集 JSON 根节点必须是列表或字典。"
        )

    normalized: list[dict[str, Any]] = []

    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise TypeError(
                f"评测项[{index}] 必须是字典，"
                f"实际得到：{type(item)!r}"
            )

        normalized.append(item)

    if not normalized:
        raise ValueError(
            "评测集不能为空。"
        )

    return normalized


def extract_query(
    item: dict[str, Any],
) -> str:
    """从评测项中读取查询文本。"""

    for key in (
        "query",
        "question",
        "text",
    ):
        value = item.get(key)

        if (
            isinstance(value, str)
            and value.strip()
        ):
            return value.strip()

    raise KeyError(
        "评测项缺少 query、question 或 text 字段。"
    )


def normalize_id_list(
    value: Any,
) -> list[str]:
    """将单个 ID 或 ID 序列统一转换为字符串列表。"""

    if value is None:
        return []

    if isinstance(value, str):
        value = [value]

    if not isinstance(value, Sequence):
        raise TypeError(
            "相关文档 ID 字段必须是字符串或序列。"
        )

    output: list[str] = []

    for item in value:
        normalized = str(item).strip()

        if (
            normalized
            and normalized not in output
        ):
            output.append(normalized)

    return output


def extract_relevant_doc_ids(
    item: dict[str, Any],
) -> list[str]:
    """读取一个查询对应的相关文档 ID。"""

    for key in (
        "relevant_doc_ids",
        "gold_doc_ids",
        "expected_doc_ids",
        "answer_doc_ids",
        "relevant_ids",
    ):
        if key in item:
            return normalize_id_list(
                item.get(key)
            )

    if "doc_id" in item:
        return normalize_id_list(
            item.get("doc_id")
        )

    return []


def extract_answerable(
    item: dict[str, Any],
    relevant_doc_ids: Sequence[str],
) -> bool:
    """判断评测问题是否可回答。"""

    for key in (
        "answerable",
        "is_answerable",
    ):
        value = item.get(key)

        if isinstance(value, bool):
            return value

    return bool(relevant_doc_ids)


def extract_case_id(
    item: dict[str, Any],
    index: int,
) -> str:
    """读取评测项 ID。"""

    for key in (
        "id",
        "case_id",
        "question_id",
        "query_id",
    ):
        value = item.get(key)

        if value is not None:
            normalized = str(value).strip()

            if normalized:
                return normalized

    return f"case_{index:04d}"


def document_doc_id(
    document: Document,
) -> str:
    """读取检索结果中的 doc_id。"""

    value = document.metadata.get(
        "doc_id"
    )

    if value is None:
        raise KeyError(
            "检索结果 metadata 中缺少 doc_id。"
            "当前评测按照 doc_id 与 "
            "relevant_doc_ids 进行比较。"
        )

    normalized = str(value).strip()

    if not normalized:
        raise ValueError(
            "检索结果 metadata 中的 doc_id 不能为空。"
        )

    return normalized


def normalize_ids(
    ids: Iterable[str],
) -> list[str]:
    """去除 ID 中的空值和重复值。"""

    output: list[str] = []

    for value in ids:
        normalized = str(value).strip()

        if (
            normalized
            and normalized not in output
        ):
            output.append(normalized)

    return output


def hit_at_k(
    retrieved_doc_ids: Sequence[str],
    relevant_doc_ids: Sequence[str],
    top_k: int,
) -> float:
    """计算 Hit@K。

    只要 Top-K 中至少存在一个相关文档，
    该查询的 Hit@K 就是 1。
    """

    relevant = set(
        normalize_ids(relevant_doc_ids)
    )

    if not relevant:
        return 0.0

    retrieved = normalize_ids(
        retrieved_doc_ids[:top_k]
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
    """计算 Recall@K。"""

    relevant = set(
        normalize_ids(relevant_doc_ids)
    )

    if not relevant:
        return 0.0

    retrieved = set(
        normalize_ids(
            retrieved_doc_ids[:top_k]
        )
    )

    return (
        len(retrieved & relevant)
        / len(relevant)
    )


def reciprocal_rank_at_k(
    retrieved_doc_ids: Sequence[str],
    relevant_doc_ids: Sequence[str],
    top_k: int,
) -> float:
    """计算当前查询的 Reciprocal Rank@K。"""

    relevant = set(
        normalize_ids(relevant_doc_ids)
    )

    if not relevant:
        return 0.0

    retrieved = normalize_ids(
        retrieved_doc_ids[:top_k]
    )

    for rank, doc_id in enumerate(
        retrieved,
        start=1,
    ):
        if doc_id in relevant:
            return 1.0 / rank

    return 0.0


def build_chunks(
    bm25_config: dict[str, Any],
) -> list[Document]:
    """根据 BM25 配置重新加载语料并构建 Chunk。"""

    input_dir = resolve_project_path(
        get_required(
            bm25_config,
            "corpus",
            "input_dir",
        )
    )

    chunk_size = int(
        get_required(
            bm25_config,
            "splitter",
            "chunk_size",
        )
    )

    chunk_overlap = int(
        get_required(
            bm25_config,
            "splitter",
            "chunk_overlap",
        )
    )

    print(
        f"[1/6] 加载 PDF：{input_dir}"
    )

    page_documents = load_pdf_corpus(
        input_dir
    )

    print(
        "[2/6] 构建 Chunk："
        f"chunk_size={chunk_size}, "
        f"chunk_overlap={chunk_overlap}"
    )

    splitter = create_recursive_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = split_page_documents(
        page_documents,
        splitter,
    )

    if not chunks:
        raise RuntimeError(
            "分块结果为空。"
        )

    missing_chunk_id = [
        index
        for index, document in enumerate(chunks)
        if not str(
            document.metadata.get(
                "chunk_id",
                "",
            )
        ).strip()
    ]

    if missing_chunk_id:
        raise RuntimeError(
            "部分 Chunk 缺少 chunk_id，"
            "无法执行 Hybrid 去重。\n"
            f"示例 Chunk 索引：{missing_chunk_id[:10]}"
        )

    print(
        f"      页面数：{len(page_documents)}"
    )
    print(
        f"      Chunk 数：{len(chunks)}"
    )

    return chunks


def create_embeddings_from_config(
    dense_config: dict[str, Any],
):
    """根据 Dense 配置创建 Embedding。"""

    model_name = str(
        get_required(
            dense_config,
            "embedding",
            "model",
        )
    )

    normalize_embeddings = bool(
        get_optional(
            dense_config,
            "embedding",
            "normalize_embeddings",
            default=True,
        )
    )

    # 兼容 create_huggingface_embeddings
    # 可能采用的不同参数形式。
    attempts = [
        {
            "model_name": model_name,
            "normalize_embeddings": (
                normalize_embeddings
            ),
            "device": "cpu",
        },
        {
            "model_name": model_name,
            "normalize_embeddings": (
                normalize_embeddings
            ),
        },
        {
            "model": model_name,
            "normalize_embeddings": (
                normalize_embeddings
            ),
            "device": "cpu",
        },
        {
            "model": model_name,
            "normalize_embeddings": (
                normalize_embeddings
            ),
        },
    ]

    errors: list[Exception] = []

    for kwargs in attempts:
        try:
            return create_huggingface_embeddings(
                **kwargs
            )
        except TypeError as exc:
            errors.append(exc)

    try:
        return create_huggingface_embeddings(
            model_name
        )
    except TypeError as exc:
        errors.append(exc)

    error_text = "\n".join(
        f"  - {error}"
        for error in errors
    )

    raise TypeError(
        "无法调用 create_huggingface_embeddings。\n"
        f"{error_text}"
    )


def open_vector_store_from_config(
    dense_config: dict[str, Any],
    embeddings: Any,
):
    """根据 Dense 配置打开持久化 Chroma。"""

    persist_directory = resolve_project_path(
        get_required(
            dense_config,
            "vector_store",
            "persist_directory",
        )
    )

    collection_name = str(
        get_optional(
            dense_config,
            "vector_store",
            "collection_name",
            default="langchain",
        )
    )

    print(
        "[3/6] 打开 Chroma："
        f"{persist_directory}"
    )

    attempts = [
        {
            "embeddings": embeddings,
            "persist_directory": (
                persist_directory
            ),
            "collection_name": (
                collection_name
            ),
        },
        {
            "embedding": embeddings,
            "persist_directory": (
                persist_directory
            ),
            "collection_name": (
                collection_name
            ),
        },
        {
            "embedding_function": embeddings,
            "persist_directory": (
                persist_directory
            ),
            "collection_name": (
                collection_name
            ),
        },
        {
            "embeddings": embeddings,
            "persist_directory": (
                persist_directory
            ),
        },
        {
            "embedding": embeddings,
            "persist_directory": (
                persist_directory
            ),
        },
    ]

    errors: list[Exception] = []

    for kwargs in attempts:
        try:
            return open_chroma_store(
                **kwargs
            )
        except TypeError as exc:
            errors.append(exc)

    try:
        return open_chroma_store(
            persist_directory,
            embeddings,
        )
    except TypeError as exc:
        errors.append(exc)

    error_text = "\n".join(
        f"  - {error}"
        for error in errors
    )

    raise TypeError(
        "无法调用 open_chroma_store。\n"
        f"{error_text}"
    )


def build_dense_search(
    dense_config: dict[str, Any],
):
    """构建 Dense 检索函数。"""

    embeddings = create_embeddings_from_config(
        dense_config
    )

    vector_store = open_vector_store_from_config(
        dense_config,
        embeddings,
    )

    def dense_search(
        query: str,
        top_k: int,
    ):
        """返回 (Document, distance)。"""

        return (
            vector_store
            .similarity_search_with_score(
                query,
                k=top_k,
            )
        )

    return dense_search


def build_bm25_index(
    chunks: Sequence[Document],
    bm25_config: dict[str, Any],
) -> BM25SearchIndex:
    """创建 BM25 内存索引。"""

    k1 = float(
        get_optional(
            bm25_config,
            "retrieval",
            "k1",
            default=1.5,
        )
    )

    b = float(
        get_optional(
            bm25_config,
            "retrieval",
            "b",
            default=0.75,
        )
    )

    print(
        "[4/6] 构建 BM25 内存索引："
        f"k1={k1}, b={b}"
    )

    errors: list[Exception] = []

    # 形式一：
    # BM25SearchIndex(documents=..., k1=..., b=...)
    try:
        return BM25SearchIndex(
            documents=list(chunks),
            k1=k1,
            b=b,
        )
    except TypeError as exc:
        errors.append(exc)

    # 形式二：
    # BM25SearchIndex(chunks=..., k1=..., b=...)
    try:
        return BM25SearchIndex(
            chunks=list(chunks),
            k1=k1,
            b=b,
        )
    except TypeError as exc:
        errors.append(exc)

    # 形式三：
    # BM25SearchIndex(chunks, k1=..., b=...)
    try:
        return BM25SearchIndex(
            list(chunks),
            k1=k1,
            b=b,
        )
    except TypeError as exc:
        errors.append(exc)

    error_text = "\n".join(
        f"  - {error}"
        for error in errors
    )

    raise TypeError(
        "无法初始化 BM25SearchIndex。\n"
        "已经尝试 documents=、chunks= 和位置参数。\n"
        f"{error_text}"
    )


def build_bm25_search(
    bm25_index: BM25SearchIndex,
):
    """将 BM25 索引包装为统一检索接口。"""

    return make_search_function(
        bm25_index,
        method_name="search",
    )


def result_to_json(
    document: Document,
    rank: int,
    save_text: bool,
) -> dict[str, Any]:
    """将 Hybrid 检索结果转换为可保存的字典。"""

    metadata = document.metadata

    item: dict[str, Any] = {
        "rank": rank,
        "chunk_id": metadata.get(
            "chunk_id"
        ),
        "doc_id": metadata.get(
            "doc_id"
        ),
        "section_id": metadata.get(
            "section_id"
        ),
        "source": metadata.get(
            "source"
        ),
        "page": metadata.get(
            "page"
        ),
        "retrieval_method": metadata.get(
            "retrieval_method"
        ),
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
    }

    if save_text:
        item["page_content"] = (
            document.page_content
        )

    return item


def debug_retrieval(
    retriever: RRFHybridRetriever,
    query: str,
    top_k: int,
) -> None:
    """打印一个查询的融合过程。"""

    print()
    print("=" * 100)
    print("Hybrid Retrieval 单查询调试")
    print("=" * 100)
    print(f"Query：{query}")

    documents = retriever.retrieve(
        query,
        top_k=top_k,
    )

    seen_chunk_ids: set[str] = set()

    for rank, document in enumerate(
        documents,
        start=1,
    ):
        metadata = document.metadata

        chunk_id = str(
            metadata.get("chunk_id")
        )

        if chunk_id in seen_chunk_ids:
            raise RuntimeError(
                "Hybrid 最终结果出现重复 chunk_id："
                f"{chunk_id}"
            )

        seen_chunk_ids.add(chunk_id)

        print("-" * 100)

        print(
            f"[{rank}] "
            f"chunk_id={chunk_id} | "
            f"doc_id={metadata.get('doc_id')} | "
            f"rrf="
            f"{float(metadata.get('rrf_score', 0.0)):.8f} | "
            f"dense_rank={metadata.get('dense_rank')} | "
            f"bm25_rank={metadata.get('bm25_rank')} | "
            f"dense_raw="
            f"{metadata.get('dense_raw_score')} | "
            f"bm25_raw="
            f"{metadata.get('bm25_raw_score')}"
        )

        preview = (
            document.page_content[:240]
            .replace("\n", " ")
        )

        print(preview)

    print("=" * 100)
    print()


def evaluate(
    retriever: RRFHybridRetriever,
    evaluation_items: Sequence[
        dict[str, Any]
    ],
    *,
    top_k: int,
    save_retrieved_chunks: bool,
) -> dict[str, Any]:
    """执行完整评测。"""

    hit_values: list[float] = []
    recall_values: list[float] = []
    reciprocal_rank_values: list[float] = []

    per_case: list[dict[str, Any]] = []

    total_start = time.perf_counter()

    for index, item in enumerate(
        evaluation_items,
        start=1,
    ):
        case_id = extract_case_id(
            item,
            index,
        )

        query = extract_query(item)

        relevant_doc_ids = (
            extract_relevant_doc_ids(item)
        )

        answerable = extract_answerable(
            item,
            relevant_doc_ids,
        )

        start = time.perf_counter()

        documents = retriever.retrieve(
            query,
            top_k=top_k,
        )

        latency_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        retrieved_doc_ids = [
            document_doc_id(document)
            for document in documents
        ]

        if answerable:
            hit_value = hit_at_k(
                retrieved_doc_ids,
                relevant_doc_ids,
                top_k,
            )

            recall_value = recall_at_k(
                retrieved_doc_ids,
                relevant_doc_ids,
                top_k,
            )

            reciprocal_rank_value = (
                reciprocal_rank_at_k(
                    retrieved_doc_ids,
                    relevant_doc_ids,
                    top_k,
                )
            )

            hit_values.append(
                hit_value
            )

            recall_values.append(
                recall_value
            )

            reciprocal_rank_values.append(
                reciprocal_rank_value
            )

        else:
            hit_value = None
            recall_value = None
            reciprocal_rank_value = None

        case_result = {
            "case_id": case_id,
            "query": query,
            "answerable": answerable,
            "relevant_doc_ids": (
                relevant_doc_ids
            ),
            "retrieved_doc_ids": (
                retrieved_doc_ids
            ),
            "hit_at_k": hit_value,
            "recall_at_k": recall_value,
            "reciprocal_rank_at_k": (
                reciprocal_rank_value
            ),
            "latency_ms": latency_ms,
            "retrieved_chunks": [
                result_to_json(
                    document,
                    rank,
                    save_retrieved_chunks,
                )
                for rank, document in enumerate(
                    documents,
                    start=1,
                )
            ],
        }

        per_case.append(
            case_result
        )

        print(
            f"[{index:03d}/"
            f"{len(evaluation_items):03d}] "
            f"{case_id} | "
            f"answerable={answerable} | "
            f"hit={hit_value} | "
            f"recall={recall_value} | "
            f"rr={reciprocal_rank_value} | "
            f"{latency_ms:.2f} ms"
        )

    total_latency_ms = (
        time.perf_counter()
        - total_start
    ) * 1000.0

    if not hit_values:
        raise RuntimeError(
            "评测集中没有可回答问题，"
            "无法计算检索指标。"
        )

    summary = {
        f"hit_at_{top_k}": mean(
            hit_values
        ),
        f"recall_at_{top_k}": mean(
            recall_values
        ),
        f"mrr_at_{top_k}": mean(
            reciprocal_rank_values
        ),
        "total_cases": len(
            evaluation_items
        ),
        "answerable_cases": len(
            hit_values
        ),
        "unanswerable_cases": (
            len(evaluation_items)
            - len(hit_values)
        ),
        "mean_latency_ms": mean(
            case["latency_ms"]
            for case in per_case
        ),
        "total_latency_ms": (
            total_latency_ms
        ),
    }

    return {
        "summary": summary,
        "cases": per_case,
    }


def save_report(
    report: dict[str, Any],
    output_path: Path,
) -> None:
    """保存 JSON 报告。"""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )


def find_metric_value(
    data: Any,
    candidate_keys: Sequence[str],
) -> float | None:
    """递归查找旧报告中的指标值。"""

    if isinstance(data, dict):
        for key in candidate_keys:
            value = data.get(key)

            if isinstance(
                value,
                (int, float),
            ):
                numeric = float(value)

                if math.isfinite(numeric):
                    return numeric

        for value in data.values():
            result = find_metric_value(
                value,
                candidate_keys,
            )

            if result is not None:
                return result

    return None


def load_report_metrics(
    path: Path,
    top_k: int,
) -> dict[str, float] | None:
    """从 Dense 或 BM25 的评测报告中读取检索指标。

    兼容以下常见字段命名：

    hit@5
    Hit@5
    hit_at_5
    hit_at_k
    hit

    recall@5
    Recall@5
    recall_at_5
    recall_at_k
    recall

    mrr@5
    MRR@5
    mrr_at_5
    mrr_at_k
    mrr
    """

    if not path.exists():
        return None

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    hit = find_metric_value(
        data,
        (
            f"hit@{top_k}",
            f"Hit@{top_k}",
            f"hit_at_{top_k}",
            "hit_at_k",
            "hit",
        ),
    )

    recall = find_metric_value(
        data,
        (
            f"recall@{top_k}",
            f"Recall@{top_k}",
            f"recall_at_{top_k}",
            "recall_at_k",
            "recall",
        ),
    )

    mrr = find_metric_value(
        data,
        (
            f"mrr@{top_k}",
            f"MRR@{top_k}",
            f"mrr_at_{top_k}",
            "mrr_at_k",
            "mrr",
        ),
    )

    if (
        hit is None
        or recall is None
        or mrr is None
    ):
        return None

    return {
        "hit": hit,
        "recall": recall,
        "mrr": mrr,
    }

def print_comparison(
    *,
    top_k: int,
    hybrid_metrics: dict[str, float],
    dense_report_path: Path,
    bm25_report_path: Path,
) -> None:
    """打印 Dense、BM25、Hybrid 的对比表。"""

    dense = load_report_metrics(
        dense_report_path,
        top_k,
    )

    bm25 = load_report_metrics(
        bm25_report_path,
        top_k,
    )

    rows: list[
        tuple[
            str,
            float | None,
            float | None,
            float | None,
        ]
    ] = []

    rows.append(
        (
            "Dense",
            (
                None
                if dense is None
                else dense["hit"]
            ),
            (
                None
                if dense is None
                else dense["recall"]
            ),
            (
                None
                if dense is None
                else dense["mrr"]
            ),
        )
    )

    rows.append(
        (
            "BM25",
            (
                None
                if bm25 is None
                else bm25["hit"]
            ),
            (
                None
                if bm25 is None
                else bm25["recall"]
            ),
            (
                None
                if bm25 is None
                else bm25["mrr"]
            ),
        )
    )

    rows.append(
        (
            "Hybrid RRF",
            hybrid_metrics[
                f"hit_at_{top_k}"
            ],
            hybrid_metrics[
                f"recall_at_{top_k}"
            ],
            hybrid_metrics[
                f"mrr_at_{top_k}"
            ],
        )
    )

    def format_value(
        value: float | None,
    ) -> str:
        if value is None:
            return "N/A"

        return f"{value:.4f}"

    print()
    print("=" * 72)
    print(
        "Dense、BM25 与 Hybrid Retrieval 对比"
    )
    print("=" * 72)

    print(
        f"{'Method':<18}"
        f"{f'Hit@{top_k}':>14}"
        f"{f'Recall@{top_k}':>14}"
        f"{f'MRR@{top_k}':>14}"
    )

    print("-" * 72)

    for method, hit, recall, mrr in rows:
        print(
            f"{method:<18}"
            f"{format_value(hit):>14}"
            f"{format_value(recall):>14}"
            f"{format_value(mrr):>14}"
        )

    print("=" * 72)

    if dense is None:
        print(
            "提示：未能从 Dense 报告读取三个指标："
            f"{dense_report_path}"
        )

    if bm25 is None:
        print(
            "提示：未能从 BM25 报告读取三个指标："
            f"{bm25_report_path}"
        )


def main() -> None:
    """程序入口。"""

    args = parse_args()

    hybrid_config_path = Path(
        args.config
    ).resolve()

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

# 与 evaluate_bm25.py 使用完全相同的评测数据集。
    evaluation_dataset_path = (
        EVAL_DATASET_PATH.resolve()
    )

    final_top_k = int(
        get_required(
            hybrid_config,
            "retrieval",
            "final_top_k",
        )
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

    output_path = resolve_project_path(
        get_required(
            hybrid_config,
            "outputs",
            "report_path",
        )
    )

    save_retrieved_chunks = bool(
        get_optional(
            hybrid_config,
            "outputs",
            "save_retrieved_chunks",
            default=True,
        )
    )

    dense_report_path = resolve_project_path(
        get_optional(
            hybrid_config,
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
            hybrid_config,
            "comparison",
            "bm25_report_path",
            default=(
                "reports/"
                "bm25_results.json"
            ),
        )
    )

    print("=" * 80)
    print(
        "RAGLab Hybrid Retrieval Evaluation"
    )
    print("=" * 80)

    print(
        f"Hybrid config："
        f"{hybrid_config_path}"
    )

    print(
        f"Dense config ："
        f"{dense_config_path}"
    )

    print(
        f"BM25 config  ："
        f"{bm25_config_path}"
    )

    print(
        f"Eval dataset ："
        f"{evaluation_dataset_path}"
    )

    print(
        "Candidates   ："
        f"Dense Top-{dense_candidate_top_k} "
        f"+ BM25 Top-{bm25_candidate_top_k}"
    )

    print(
        "Fusion       ："
        f"RRF(k={rrf_k}, "
        f"dense_weight={dense_weight}, "
        f"bm25_weight={bm25_weight})"
    )

    print(
        f"Final Top-K  ："
        f"{final_top_k}"
    )

    print("=" * 80)

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

    retriever = RRFHybridRetriever(
        dense_search=dense_search,
        bm25_search=bm25_search,
        dense_candidate_top_k=(
            dense_candidate_top_k
        ),
        bm25_candidate_top_k=(
            bm25_candidate_top_k
        ),
        final_top_k=final_top_k,
        rrf_k=rrf_k,
        dense_weight=dense_weight,
        bm25_weight=bm25_weight,
    )

    if args.debug_query:
        debug_retrieval(
            retriever,
            args.debug_query,
            final_top_k,
        )

    print(
        "[5/6] 加载评测集："
        f"{evaluation_dataset_path}"
    )

    evaluation_items = (
        load_evaluation_items(
            evaluation_dataset_path
        )
    )

    print(
        f"      评测项数："
        f"{len(evaluation_items)}"
    )

    print(
        "[6/6] 执行 Hybrid Retrieval 评测"
    )

    evaluation_result = evaluate(
        retriever,
        evaluation_items,
        top_k=final_top_k,
        save_retrieved_chunks=(
            save_retrieved_chunks
        ),
    )

    report = {
        "experiment_name": str(
            hybrid_config.get(
                "experiment_name",
                "hybrid_dense_bm25_rrf",
            )
        ),
        "config": {
            "hybrid_config_path": str(
                hybrid_config_path
            ),
            "dense_config_path": str(
                dense_config_path
            ),
            "bm25_config_path": str(
                bm25_config_path
            ),
            "evaluation_dataset_path": str(
                evaluation_dataset_path
            ),
            "dense_candidate_top_k": (
                dense_candidate_top_k
            ),
            "bm25_candidate_top_k": (
                bm25_candidate_top_k
            ),
            "final_top_k": (
                final_top_k
            ),
            "rrf_k": rrf_k,
            "dense_weight": dense_weight,
            "bm25_weight": bm25_weight,
            "deduplicate_by": (
                "chunk_id"
            ),
        },
        **evaluation_result,
    }

    save_report(
        report,
        output_path,
    )

    summary = report["summary"]

    print()
    print("=" * 80)
    print(
        "Hybrid Retrieval 评测完成"
    )
    print("=" * 80)

    print(
        f"Hit@{final_top_k}："
        f"{summary[f'hit_at_{final_top_k}']:.4f}"
    )

    print(
        f"Recall@{final_top_k}："
        f"{summary[f'recall_at_{final_top_k}']:.4f}"
    )

    print(
        f"MRR@{final_top_k}："
        f"{summary[f'mrr_at_{final_top_k}']:.4f}"
    )

    print(
        "平均检索耗时："
        f"{summary['mean_latency_ms']:.2f} ms"
    )

    print(
        f"报告位置：{output_path}"
    )

    print_comparison(
        top_k=final_top_k,
        hybrid_metrics=summary,
        dense_report_path=(
            dense_report_path
        ),
        bm25_report_path=(
            bm25_report_path
        ),
    )


if __name__ == "__main__":
    main()