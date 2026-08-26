from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any, Literal, Sequence

from langchain_core.documents import Document


# ============================================================
# 项目路径
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# 复用 RAGLab 已有模块
# ============================================================

from raglab.embeddings.factory import (
    create_huggingface_embeddings,
)

from raglab.retrieval.bm25 import (
    BM25SearchIndex,
)

from raglab.retrieval.hybrid import (
    RRFHybridRetriever,
    make_search_function,
    normalize_result,
)

from raglab.vectorstores.chroma_store import (
    open_chroma_store,
)


# ============================================================
# 类型
# ============================================================

RetrievalMode = Literal[
    "bm25",
    "dense",
    "hybrid",
]


# ============================================================
# 默认路径
# ============================================================

DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT
    / "storage"
    / "intelligence"
    / "rag_index_manifest.json"
)

DEFAULT_CHROMA_DIRECTORY = (
    PROJECT_ROOT
    / "storage"
    / "chroma"
    / "intelligence"
)

DEFAULT_CHROMA_COLLECTION_NAME = (
    "github_intelligence"
)

DEFAULT_BM25_DIRECTORY = (
    PROJECT_ROOT
    / "storage"
    / "bm25"
    / "intelligence"
)

DEFAULT_BM25_DOCUMENTS_PATH = (
    DEFAULT_BM25_DIRECTORY
    / "documents.jsonl"
)

DEFAULT_BM25_PICKLE_PATH = (
    DEFAULT_BM25_DIRECTORY
    / "bm25_index.pkl"
)


# ============================================================
# 默认模型与检索参数
# ============================================================

DEFAULT_EMBEDDING_MODEL = (
    "BAAI/bge-small-zh-v1.5"
)

DEFAULT_EMBEDDING_DEVICE = "cpu"

DEFAULT_NORMALIZE_EMBEDDINGS = True

DEFAULT_BM25_K1 = 1.5
DEFAULT_BM25_B = 0.75

DEFAULT_DENSE_CANDIDATE_TOP_K = 20
DEFAULT_BM25_CANDIDATE_TOP_K = 20
DEFAULT_FINAL_TOP_K = 5

DEFAULT_RRF_K = 60
DEFAULT_DENSE_WEIGHT = 1.0
DEFAULT_BM25_WEIGHT = 1.0


# ============================================================
# 基础工具
# ============================================================


def compact_text(
    value: Any,
) -> str:
    """
    压缩连续空白。
    """
    if value is None:
        return ""

    return " ".join(
        str(
            value
        ).split()
    ).strip()


def safe_int(
    value: Any,
    default: int,
) -> int:
    """
    安全转换整数。
    """
    try:
        return int(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        return default


def safe_float(
    value: Any,
    default: float,
) -> float:
    """
    安全转换浮点数。
    """
    try:
        return float(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        return default


def read_json(
    path: Path,
) -> dict[str, Any]:
    """
    读取 JSON 对象。

    文件不存在或格式错误时返回空字典。
    """
    path = Path(
        path
    )

    if not path.exists():
        return {}

    try:
        data = json.loads(
            path.read_text(
                encoding="utf-8",
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
        UnicodeDecodeError,
    ):
        return {}

    if not isinstance(
        data,
        dict,
    ):
        return {}

    return data


def resolve_path(
    value: Any,
    default: Path,
) -> Path:
    """
    将配置路径转换为绝对路径。
    """
    text = compact_text(
        value
    )

    if not text:
        return Path(
            default
        ).resolve()

    path = Path(
        text
    )

    if path.is_absolute():
        return path.resolve()

    return (
        PROJECT_ROOT
        / path
    ).resolve()


def normalize_mode(
    mode: str,
) -> RetrievalMode:
    """
    校验检索模式。
    """
    normalized = compact_text(
        mode
    ).lower()

    if normalized not in {
        "bm25",
        "dense",
        "hybrid",
    }:
        raise ValueError(
            "mode 只能是 bm25、dense 或 hybrid，"
            f"当前值：{mode}"
        )

    return normalized  # type: ignore[return-value]


# ============================================================
# BM25 文档加载
# ============================================================


def load_bm25_documents(
    path: Path,
) -> list[Document]:
    """
    从 documents.jsonl 读取用于 BM25 的 Chunk 文档。
    """
    path = Path(
        path
    )

    if not path.exists():
        raise FileNotFoundError(
            "BM25 Chunk 文档不存在："
            f"{path}\n"
            "请先运行 scripts\\build_intelligence_indexes.py"
        )

    documents: list[Document] = []

    for line_number, line in enumerate(
        path.read_text(
            encoding="utf-8",
        ).splitlines(),
        start=1,
    ):
        normalized_line = line.strip()

        if not normalized_line:
            continue

        try:
            record = json.loads(
                normalized_line
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                "BM25 documents.jsonl 格式错误："
                f"{path} 第 {line_number} 行\n"
                f"{exc}"
            ) from exc

        if not isinstance(
            record,
            dict,
        ):
            continue

        page_content = compact_text(
            record.get(
                "page_content"
            )
        )

        metadata = record.get(
            "metadata"
        )

        if not isinstance(
            metadata,
            dict,
        ):
            metadata = {}

        if not page_content:
            continue

        chunk_id = compact_text(
            metadata.get(
                "chunk_id"
            )
        )

        if not chunk_id:
            raise ValueError(
                "BM25 文档缺少 chunk_id："
                f"{path} 第 {line_number} 行"
            )

        documents.append(
            Document(
                page_content=(
                    page_content
                ),
                metadata=dict(
                    metadata
                ),
            )
        )

    if not documents:
        raise RuntimeError(
            "BM25 documents.jsonl 中没有有效文档："
            f"{path}"
        )

    return documents


def create_bm25_index(
    documents: Sequence[Document],
    *,
    k1: float,
    b: float,
) -> BM25SearchIndex:
    """
    兼容 BM25SearchIndex 的多种构造接口。
    """
    errors: list[Exception] = []

    constructor_options = (
        {
            "documents": list(
                documents
            ),
            "k1": k1,
            "b": b,
        },
        {
            "chunks": list(
                documents
            ),
            "k1": k1,
            "b": b,
        },
    )

    for kwargs in constructor_options:
        try:
            return BM25SearchIndex(
                **kwargs
            )

        except TypeError as exc:
            errors.append(
                exc
            )

    try:
        return BM25SearchIndex(
            list(
                documents
            ),
            k1=k1,
            b=b,
        )

    except TypeError as exc:
        errors.append(
            exc
        )

    error_text = "\n".join(
        f"- {error}"
        for error in errors
    )

    raise TypeError(
        "无法初始化 BM25SearchIndex。\n"
        "已经尝试 documents=、chunks= 和位置参数。\n"
        f"{error_text}"
    )


def load_or_rebuild_bm25_index(
    *,
    pickle_path: Path,
    documents_path: Path,
    k1: float,
    b: float,
) -> tuple[
    BM25SearchIndex,
    str,
    int,
]:
    """
    优先从 pickle 加载 BM25。

    pickle 不存在或加载失败时，
    从 documents.jsonl 重建。
    """
    pickle_path = Path(
        pickle_path
    )

    documents_path = Path(
        documents_path
    )

    if pickle_path.exists():
        try:
            with pickle_path.open(
                "rb"
            ) as file:
                index = pickle.load(
                    file
                )

            if not callable(
                getattr(
                    index,
                    "search",
                    None,
                )
            ):
                raise TypeError(
                    "pickle 中的对象没有 search 方法。"
                )

            document_count = 0

            for attribute_name in (
                "documents",
                "chunks",
                "corpus",
            ):
                value = getattr(
                    index,
                    attribute_name,
                    None,
                )

                if isinstance(
                    value,
                    Sequence,
                ):
                    document_count = len(
                        value
                    )
                    break

            return (
                index,
                "pickle",
                document_count,
            )

        except Exception as exc:
            print(
                "警告：BM25 pickle 加载失败，"
                "将从 documents.jsonl 重建。"
            )

            print(
                f"原因：{exc}"
            )

    documents = load_bm25_documents(
        documents_path
    )

    index = create_bm25_index(
        documents,
        k1=k1,
        b=b,
    )

    return (
        index,
        "documents_jsonl",
        len(
            documents
        ),
    )


# ============================================================
# Metadata-aware BM25 辅助
# ============================================================


def extract_bm25_documents(
    index: Any,
) -> list[Document]:
    """从已加载的 BM25 索引中提取原始 Document 列表。

    当前 RAGLab 的 BM25SearchIndex 在不同阶段可能使用
    documents、chunks 或 corpus 保存原始文档，因此这里
    统一兼容三种属性名。
    """

    for attribute_name in (
        "documents",
        "chunks",
        "corpus",
    ):
        value = getattr(
            index,
            attribute_name,
            None,
        )

        if isinstance(
            value,
            Sequence,
        ) and not isinstance(
            value,
            (
                str,
                bytes,
                bytearray,
            ),
        ):
            documents = [
                item
                for item in value
                if isinstance(
                    item,
                    Document,
                )
            ]

            if documents:
                return documents

    raise TypeError(
        "无法从 BM25 索引中提取原始 Document。"
        "期望索引具有 documents、chunks 或 corpus 属性。"
    )


def filter_bm25_documents_by_metadata(
    documents: Sequence[Document],
    *,
    snapshot_date: str | None = None,
    doc_types: Sequence[str] | None = None,
    repository: str | None = None,
    topic_name: str | None = None,
) -> list[Document]:
    """在语义检索前按 GitHub RAG metadata 做硬约束过滤。

    这些条件属于确定性的业务约束，不应交给 BM25 通过
    文本相关性去“猜”。只有通过过滤的文档才进入后续
    BM25 排序。
    """

    normalized_snapshot_date = compact_text(
        snapshot_date
    )

    normalized_doc_types = {
        compact_text(
            value
        ).casefold()
        for value in (
            doc_types
            or []
        )
        if compact_text(
            value
        )
    }

    normalized_repository = compact_text(
        repository
    ).casefold()

    normalized_topic_name = compact_text(
        topic_name
    ).casefold()

    filtered_documents: list[Document] = []

    for document in documents:
        metadata = dict(
            document.metadata
            or {}
        )

        if normalized_snapshot_date:
            document_date = compact_text(
                metadata.get(
                    "snapshot_date"
                )
            )

            if (
                document_date
                != normalized_snapshot_date
            ):
                continue

        if normalized_doc_types:
            document_type = compact_text(
                metadata.get(
                    "doc_type"
                )
            ).casefold()

            if (
                document_type
                not in normalized_doc_types
            ):
                continue

        if normalized_repository:
            document_repository = compact_text(
                metadata.get(
                    "repository"
                )
            ).casefold()

            if (
                document_repository
                != normalized_repository
            ):
                continue

        if normalized_topic_name:
            document_topic_name = compact_text(
                metadata.get(
                    "topic_name"
                )
            ).casefold()

            if (
                document_topic_name
                != normalized_topic_name
            ):
                continue

        filtered_documents.append(
            document
        )

    return filtered_documents


def _document_chunk_order(
    document: Document,
) -> tuple[int, int, str]:
    """为同一逻辑文档的 Chunk 生成稳定顺序。"""

    metadata = dict(
        document.metadata
        or {}
    )

    return (
        safe_int(
            metadata.get(
                "source_line"
            ),
            0,
        ),
        safe_int(
            metadata.get(
                "chunk_index"
            ),
            0,
        ),
        compact_text(
            metadata.get(
                "chunk_id"
            )
        ),
    )


def search_bm25_with_metadata_filters(
    index: Any,
    query: str,
    *,
    top_k: int,
    k1: float,
    b: float,
    snapshot_date: str | None = None,
    doc_types: Sequence[str] | None = None,
    repository: str | None = None,
    topic_name: str | None = None,
) -> tuple[Sequence[Any], int]:
    """先做 metadata pre-filter，再在候选集内部执行 BM25。

    返回：
        (检索结果, 过滤后的候选 Chunk 数)

    如果过滤后所有 Chunk 都属于同一个 document_id，说明
    用户已经通过 metadata 精确定位到单一逻辑文档。例如：

        snapshot_date=2026-08-17
        doc_types=["daily_brief"]

    此时不再让同一日报的 Chunk 互相竞争 BM25 分数，而是
    直接按原始 Chunk 顺序返回完整文档内容。
    """

    normalized_query = compact_text(
        query
    )

    if not normalized_query:
        return (
            [],
            0,
        )

    requested_top_k = int(
        top_k
    )

    if requested_top_k <= 0:
        raise ValueError(
            "top_k 必须大于 0。"
        )

    all_documents = extract_bm25_documents(
        index
    )

    filtered_documents = (
        filter_bm25_documents_by_metadata(
            all_documents,
            snapshot_date=(
                snapshot_date
            ),
            doc_types=doc_types,
            repository=repository,
            topic_name=topic_name,
        )
    )

    candidate_count = len(
        filtered_documents
    )

    if candidate_count == 0:
        return (
            [],
            0,
        )

    document_ids = {
        compact_text(
            document.metadata.get(
                "document_id"
            )
            or document.metadata.get(
                "doc_id"
            )
        )
        for document
        in filtered_documents
    }

    document_ids.discard(
        ""
    )

    # metadata 已经精确定位到同一逻辑文档时，
    # 直接按 Chunk 顺序返回，避免日报正文被 Top-K 截断或乱序。
    if len(document_ids) == 1:
        ordered_documents = sorted(
            filtered_documents,
            key=_document_chunk_order,
        )

        return (
            ordered_documents,
            candidate_count,
        )

    filtered_index = create_bm25_index(
        filtered_documents,
        k1=float(
            k1
        ),
        b=float(
            b
        ),
    )

    effective_top_k = min(
        requested_top_k,
        candidate_count,
    )

    search_method = getattr(
        filtered_index,
        "search",
        None,
    )

    if not callable(
        search_method
    ):
        raise TypeError(
            "过滤后的 BM25 索引没有可调用的 search() 方法。"
        )

    try:
        results = search_method(
            query=normalized_query,
            top_k=effective_top_k,
        )

    except TypeError:
        results = search_method(
            normalized_query,
            effective_top_k,
        )

    return (
        results,
        candidate_count,
    )


# ============================================================
# 检索结果处理
# ============================================================


def copy_document_with_metadata(
    document: Document,
    extra_metadata: dict[str, Any],
) -> Document:
    """
    复制 Document，并补充检索元数据。
    """
    metadata = dict(
        document.metadata
    )

    metadata.update(
        extra_metadata
    )

    return Document(
        page_content=(
            document.page_content
        ),
        metadata=metadata,
    )


def normalize_search_results(
    results: Sequence[Any],
    *,
    retrieval_method: str,
) -> list[Document]:
    """
    将检索结果统一转换为 Document 列表。
    """
    documents: list[Document] = []

    for rank, raw_result in enumerate(
        results,
        start=1,
    ):
        normalized = normalize_result(
            raw_result
        )

        document = normalized.document

        documents.append(
            copy_document_with_metadata(
                document,
                {
                    "retrieval_method": (
                        retrieval_method
                    ),

                    "retrieval_rank": (
                        rank
                    ),

                    "retrieval_raw_score": (
                        normalized.raw_score
                    ),
                },
            )
        )

    return documents


def document_to_dict(
    document: Document,
    *,
    include_text: bool = True,
) -> dict[str, Any]:
    """
    将检索结果转换为普通字典。
    """
    metadata = dict(
        document.metadata
    )

    result: dict[str, Any] = {
        "document_id": (
            metadata.get(
                "document_id"
            )
            or metadata.get(
                "doc_id"
            )
        ),

        "chunk_id": (
            metadata.get(
                "chunk_id"
            )
        ),

        "doc_type": (
            metadata.get(
                "doc_type"
            )
        ),

        "title": (
            metadata.get(
                "title"
            )
        ),

        "snapshot_date": (
            metadata.get(
                "snapshot_date"
            )
        ),

        "repository": (
            metadata.get(
                "repository"
            )
        ),

        "topic_name": (
            metadata.get(
                "topic_name"
            )
        ),

        "source": (
            metadata.get(
                "source"
            )
        ),

        "retrieval_method": (
            metadata.get(
                "retrieval_method"
            )
        ),

        "retrieval_rank": (
            metadata.get(
                "retrieval_rank"
                )
                or metadata.get(
                    "hybrid_rank"
                )
        ),

        "retrieval_raw_score": (
            metadata.get(
                "retrieval_raw_score"
            )
        ),

        "rrf_score": (
            metadata.get(
                "rrf_score"
            )
        ),

        "dense_rank": (
            metadata.get(
                "dense_rank"
            )
        ),

        "bm25_rank": (
            metadata.get(
                "bm25_rank"
            )
        ),

        "dense_raw_score": (
            metadata.get(
                "dense_raw_score"
            )
        ),

        "bm25_raw_score": (
            metadata.get(
                "bm25_raw_score"
            )
        ),
    }

    if include_text:
        result[
            "page_content"
        ] = document.page_content

    return result


# ============================================================
# 持久化技术情报检索器
# ============================================================


class IntelligenceRetriever:
    """
    GitHub 技术情报持久化检索器。

    支持：

    - BM25；
    - Dense Chroma；
    - Dense + BM25 RRF Hybrid。
    """

    def __init__(
        self,
        *,
        manifest_path: Path = (
            DEFAULT_MANIFEST_PATH
        ),
        default_mode: RetrievalMode = (
            "bm25"
        ),
        default_top_k: int = (
            DEFAULT_FINAL_TOP_K
        ),
        dense_candidate_top_k: int = (
            DEFAULT_DENSE_CANDIDATE_TOP_K
        ),
        bm25_candidate_top_k: int = (
            DEFAULT_BM25_CANDIDATE_TOP_K
        ),
        rrf_k: int = DEFAULT_RRF_K,
        dense_weight: float = (
            DEFAULT_DENSE_WEIGHT
        ),
        bm25_weight: float = (
            DEFAULT_BM25_WEIGHT
        ),
    ) -> None:
        self.manifest_path = Path(
            manifest_path
        ).resolve()

        self.manifest = read_json(
            self.manifest_path
        )

        self.default_mode = (
            normalize_mode(
                default_mode
            )
        )

        self.default_top_k = int(
            default_top_k
        )

        if self.default_top_k <= 0:
            raise ValueError(
                "default_top_k 必须大于 0。"
            )

        self.dense_candidate_top_k = int(
            dense_candidate_top_k
        )

        self.bm25_candidate_top_k = int(
            bm25_candidate_top_k
        )

        if (
            self.dense_candidate_top_k
            <= 0
        ):
            raise ValueError(
                "dense_candidate_top_k "
                "必须大于 0。"
            )

        if (
            self.bm25_candidate_top_k
            <= 0
        ):
            raise ValueError(
                "bm25_candidate_top_k "
                "必须大于 0。"
            )

        self.rrf_k = int(
            rrf_k
        )

        self.dense_weight = float(
            dense_weight
        )

        self.bm25_weight = float(
            bm25_weight
        )

        self._load_configuration()
        self._load_dense_store()
        self._load_bm25()
        self._build_hybrid()

    # --------------------------------------------------------
    # 配置解析
    # --------------------------------------------------------

    def _load_configuration(
        self,
    ) -> None:
        """
        从 manifest 读取索引配置。

        manifest 不存在时使用默认值。
        """
        embedding_config = (
            self.manifest.get(
                "embedding"
            )
        )

        if not isinstance(
            embedding_config,
            dict,
        ):
            embedding_config = {}

        chroma_config = (
            self.manifest.get(
                "chroma"
            )
        )

        if not isinstance(
            chroma_config,
            dict,
        ):
            chroma_config = {}

        bm25_config = (
            self.manifest.get(
                "bm25"
            )
        )

        if not isinstance(
            bm25_config,
            dict,
        ):
            bm25_config = {}

        self.embedding_model = (
            compact_text(
                embedding_config.get(
                    "model"
                )
            )
            or DEFAULT_EMBEDDING_MODEL
        )

        self.embedding_device = (
            compact_text(
                embedding_config.get(
                    "device"
                )
            )
            or DEFAULT_EMBEDDING_DEVICE
        )

        self.normalize_embeddings = bool(
            embedding_config.get(
                "normalize_embeddings",
                DEFAULT_NORMALIZE_EMBEDDINGS,
            )
        )

        self.chroma_directory = resolve_path(
            chroma_config.get(
                "persist_directory"
            ),
            DEFAULT_CHROMA_DIRECTORY,
        )

        self.chroma_collection_name = (
            compact_text(
                chroma_config.get(
                    "collection_name"
                )
            )
            or DEFAULT_CHROMA_COLLECTION_NAME
        )

        self.bm25_directory = resolve_path(
            bm25_config.get(
                "directory"
            ),
            DEFAULT_BM25_DIRECTORY,
        )

        self.bm25_documents_path = (
            resolve_path(
                bm25_config.get(
                    "documents_path"
                ),
                DEFAULT_BM25_DOCUMENTS_PATH,
            )
        )

        configured_pickle_path = (
            bm25_config.get(
                "pickle_path"
            )
        )

        if configured_pickle_path:
            self.bm25_pickle_path = (
                resolve_path(
                    configured_pickle_path,
                    DEFAULT_BM25_PICKLE_PATH,
                )
            )
        else:
            self.bm25_pickle_path = (
                DEFAULT_BM25_PICKLE_PATH
            )

        self.bm25_k1 = safe_float(
            bm25_config.get(
                "k1"
            ),
            DEFAULT_BM25_K1,
        )

        self.bm25_b = safe_float(
            bm25_config.get(
                "b"
            ),
            DEFAULT_BM25_B,
        )

    # --------------------------------------------------------
    # Dense
    # --------------------------------------------------------

    def _load_dense_store(
        self,
    ) -> None:
        """
        打开持久化 Chroma。
        """
        if not self.chroma_directory.exists():
            raise FileNotFoundError(
                "Chroma 持久化目录不存在："
                f"{self.chroma_directory}\n"
                "请先运行：\n"
                "python scripts\\build_intelligence_indexes.py"
            )

        self.embeddings = (
            create_huggingface_embeddings(
                model_name=(
                    self.embedding_model
                ),

                normalize_embeddings=(
                    self.normalize_embeddings
                ),

                device=(
                    self.embedding_device
                ),
            )
        )

        self.vector_store = (
            open_chroma_store(
                embeddings=(
                    self.embeddings
                ),

                persist_directory=(
                    self.chroma_directory
                ),

                collection_name=(
                    self.chroma_collection_name
                ),
            )
        )

    def _dense_raw_search(
        self,
        query: str,
        top_k: int,
    ) -> Sequence[Any]:
        """
        执行 Chroma Dense 检索。
        """
        return (
            self.vector_store
            .similarity_search_with_score(
                query,
                k=top_k,
            )
        )

    def search_dense(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> list[Document]:
        """
        执行 Dense 检索。
        """
        normalized_query = compact_text(
            query
        )

        if not normalized_query:
            return []

        requested_top_k = (
            self.default_top_k
            if top_k is None
            else int(
                top_k
            )
        )

        if requested_top_k <= 0:
            raise ValueError(
                "top_k 必须大于 0。"
            )

        results = self._dense_raw_search(
            normalized_query,
            requested_top_k,
        )

        return normalize_search_results(
            results,
            retrieval_method=(
                "dense"
            ),
        )

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    def _load_bm25(
        self,
    ) -> None:
        """
        加载或重建 BM25。
        """
        (
            self.bm25_index,
            self.bm25_load_method,
            self.bm25_document_count,
        ) = load_or_rebuild_bm25_index(
            pickle_path=(
                self.bm25_pickle_path
            ),

            documents_path=(
                self.bm25_documents_path
            ),

            k1=self.bm25_k1,

            b=self.bm25_b,
        )

        self._bm25_search_function = (
            make_search_function(
                self.bm25_index,
                method_name="search",
            )
        )

    def _bm25_raw_search(
        self,
        query: str,
        top_k: int,
    ) -> Sequence[Any]:
        """
        执行 BM25 原始检索。
        """
        return self._bm25_search_function(
            query,
            top_k,
        )

    def search_bm25(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> list[Document]:
        """
        执行 BM25 检索。
        """
        normalized_query = compact_text(
            query
        )

        if not normalized_query:
            return []

        requested_top_k = (
            self.default_top_k
            if top_k is None
            else int(
                top_k
            )
        )

        if requested_top_k <= 0:
            raise ValueError(
                "top_k 必须大于 0。"
            )

        results = self._bm25_raw_search(
            normalized_query,
            requested_top_k,
        )

        return normalize_search_results(
            results,
            retrieval_method=(
                "bm25"
            ),
        )

    # --------------------------------------------------------
    # Hybrid
    # --------------------------------------------------------

    def _build_hybrid(
        self,
    ) -> None:
        """
        构建 RRF Hybrid 检索器。
        """
        self.hybrid_retriever = (
            RRFHybridRetriever(
                dense_search=(
                    self._dense_raw_search
                ),

                bm25_search=(
                    self._bm25_raw_search
                ),

                dense_candidate_top_k=(
                    self.dense_candidate_top_k
                ),

                bm25_candidate_top_k=(
                    self.bm25_candidate_top_k
                ),

                final_top_k=(
                    self.default_top_k
                ),

                rrf_k=(
                    self.rrf_k
                ),

                dense_weight=(
                    self.dense_weight
                ),

                bm25_weight=(
                    self.bm25_weight
                ),
            )
        )

    def search_hybrid(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> list[Document]:
        """
        执行 Dense + BM25 RRF 检索。
        """
        normalized_query = compact_text(
            query
        )

        if not normalized_query:
            return []

        requested_top_k = (
            self.default_top_k
            if top_k is None
            else int(
                top_k
            )
        )

        if requested_top_k <= 0:
            raise ValueError(
                "top_k 必须大于 0。"
            )

        return self.hybrid_retriever.retrieve(
            normalized_query,
            top_k=requested_top_k,
        )

    # --------------------------------------------------------
    # 统一入口
    # --------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        mode: RetrievalMode | str | None = None,
        top_k: int | None = None,
    ) -> list[Document]:
        """
        统一检索入口。
        """
        selected_mode = (
            self.default_mode
            if mode is None
            else normalize_mode(
                mode
            )
        )

        if selected_mode == "bm25":
            return self.search_bm25(
                query,
                top_k=top_k,
            )

        if selected_mode == "dense":
            return self.search_dense(
                query,
                top_k=top_k,
            )

        return self.search_hybrid(
            query,
            top_k=top_k,
        )

    def search_as_dicts(
        self,
        query: str,
        *,
        mode: RetrievalMode | str | None = None,
        top_k: int | None = None,
        include_text: bool = True,
    ) -> list[dict[str, Any]]:
        """
        执行检索并转换为字典。
        """
        documents = self.search(
            query,
            mode=mode,
            top_k=top_k,
        )

        return [
            document_to_dict(
                document,
                include_text=(
                    include_text
                ),
            )
            for document in documents
        ]

    # --------------------------------------------------------
    # 状态
    # --------------------------------------------------------

    def chroma_document_count(
        self,
    ) -> int | None:
        """
        尝试读取 Chroma Collection 中的记录数。
        """
        collection = getattr(
            self.vector_store,
            "_collection",
            None,
        )

        count_method = getattr(
            collection,
            "count",
            None,
        )

        if not callable(
            count_method
        ):
            return None

        try:
            return int(
                count_method()
            )
        except Exception:
            return None

    def status(
        self,
    ) -> dict[str, Any]:
        """
        返回当前检索器状态。
        """
        return {
            "manifest_path": str(
                self.manifest_path
            ),

            "manifest_exists": (
                self.manifest_path.exists()
            ),

            "default_mode": (
                self.default_mode
            ),

            "default_top_k": (
                self.default_top_k
            ),

            "embedding_model": (
                self.embedding_model
            ),

            "embedding_device": (
                self.embedding_device
            ),

            "normalize_embeddings": (
                self.normalize_embeddings
            ),

            "chroma_directory": str(
                self.chroma_directory
            ),

            "chroma_collection_name": (
                self.chroma_collection_name
            ),

            "chroma_document_count": (
                self.chroma_document_count()
            ),

            "bm25_directory": str(
                self.bm25_directory
            ),

            "bm25_pickle_path": str(
                self.bm25_pickle_path
            ),

            "bm25_documents_path": str(
                self.bm25_documents_path
            ),

            "bm25_load_method": (
                self.bm25_load_method
            ),

            "bm25_document_count": (
                self.bm25_document_count
            ),

            "bm25_k1": (
                self.bm25_k1
            ),

            "bm25_b": (
                self.bm25_b
            ),

            "dense_candidate_top_k": (
                self.dense_candidate_top_k
            ),

            "bm25_candidate_top_k": (
                self.bm25_candidate_top_k
            ),

            "rrf_k": (
                self.rrf_k
            ),

            "dense_weight": (
                self.dense_weight
            ),

            "bm25_weight": (
                self.bm25_weight
            ),
        }


# ============================================================
# 便捷创建函数
# ============================================================


def create_intelligence_retriever(
    *,
    default_mode: RetrievalMode = (
        "bm25"
    ),
    default_top_k: int = (
        DEFAULT_FINAL_TOP_K
    ),
) -> IntelligenceRetriever:
    """
    创建默认技术情报检索器。
    """
    return IntelligenceRetriever(
        default_mode=default_mode,
        default_top_k=default_top_k,
    )