from __future__ import annotations

import json
import pickle
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

from raglab.ingestion.splitters import (
    create_recursive_splitter,
)

from raglab.retrieval.bm25 import (
    BM25SearchIndex,
)

from raglab.vectorstores.chroma_store import (
    open_chroma_store,
)


# ============================================================
# 配置区
# ============================================================

TIMEZONE = "Asia/Shanghai"

# 持久化模块已经生成的统一 RAG 文档源。
RAG_DOCUMENTS_ROOT = (
    PROJECT_ROOT
    / "data"
    / "intelligence"
    / "rag_documents"
)

# Chroma 向量库。
CHROMA_DIRECTORY = (
    PROJECT_ROOT
    / "storage"
    / "chroma"
    / "intelligence"
)

CHROMA_COLLECTION_NAME = (
    "github_intelligence"
)

# BM25 索引和 Chunk 文档。
BM25_DIRECTORY = (
    PROJECT_ROOT
    / "storage"
    / "bm25"
    / "intelligence"
)

BM25_DOCUMENTS_PATH = (
    BM25_DIRECTORY
    / "documents.jsonl"
)

BM25_PICKLE_PATH = (
    BM25_DIRECTORY
    / "bm25_index.pkl"
)

# 本次索引构建记录。
MANIFEST_PATH = (
    PROJECT_ROOT
    / "storage"
    / "intelligence"
    / "rag_index_manifest.json"
)

# 沿用原 RAGLab 的 Embedding 配置。
EMBEDDING_MODEL = (
    "BAAI/bge-small-zh-v1.5"
)

EMBEDDING_DEVICE = "cpu"

NORMALIZE_EMBEDDINGS = True

# 沿用之前验证过的递归切分参数。
CHUNK_SIZE = 600
CHUNK_OVERLAP = 80

# 沿用之前 BM25 参数。
BM25_K1 = 1.5
BM25_B = 0.75

# 索引构建完成后的连通测试。
TEST_QUERY = (
    "AI Agent RAG MCP 技术热点"
)

TEST_TOP_K = 3


# ============================================================
# 文件工具
# ============================================================


def write_json(
    path: Path,
    data: Any,
) -> None:
    """
    原子化保存 JSON。
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    temporary_path.replace(
        path
    )


def write_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """
    原子化保存 JSONL。
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    content = "\n".join(
        json.dumps(
            row,
            ensure_ascii=False,
        )
        for row in rows
    )

    temporary_path.write_text(
        content
        + (
            "\n"
            if content
            else ""
        ),
        encoding="utf-8",
    )

    temporary_path.replace(
        path
    )


def compact_text(
    value: Any,
) -> str:
    """
    压缩连续空白。
    """
    return " ".join(
        str(
            value
            or ""
        ).split()
    ).strip()


def to_chroma_metadata(
    value: Any,
) -> str | int | float | bool:
    """
    将元数据转换为 Chroma 支持的标量类型。

    Chroma 元数据不直接接受列表和字典，
    因此将它们转换为 JSON 字符串。
    """
    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    if value is None:
        return ""

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(
            ",",
            ":",
        ),
    )


# ============================================================
# 读取统一 RAG 文档源
# ============================================================


def load_source_documents() -> tuple[
    list[Document],
    list[Path],
]:
    """
    读取所有日期目录中的 JSONL 文档。

    当前目录示例：

    data/intelligence/rag_documents/
    └── 2026-08-02/
        ├── repository_summaries.jsonl
        ├── daily_hotspots.jsonl
        └── daily_brief.jsonl

    使用 document_id 去重。
    """
    if not RAG_DOCUMENTS_ROOT.exists():
        raise FileNotFoundError(
            "RAG 文档源目录不存在："
            f"{RAG_DOCUMENTS_ROOT}"
        )

    jsonl_paths = sorted(
        RAG_DOCUMENTS_ROOT.rglob(
            "*.jsonl"
        )
    )

    if not jsonl_paths:
        raise RuntimeError(
            "没有找到 JSONL 文档："
            f"{RAG_DOCUMENTS_ROOT}"
        )

    records: dict[
        str,
        dict[str, Any],
    ] = {}

    for jsonl_path in jsonl_paths:
        lines = jsonl_path.read_text(
            encoding="utf-8"
        ).splitlines()

        for line_number, line in enumerate(
            lines,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:
                raise ValueError(
                    "JSONL 解析失败："
                    f"{jsonl_path} "
                    f"第 {line_number} 行\n"
                    f"{exc}"
                ) from exc

            if not isinstance(
                record,
                dict,
            ):
                continue

            document_id = compact_text(
                record.get(
                    "document_id"
                )
            )

            text = compact_text(
                record.get(
                    "text"
                )
            )

            if not document_id:
                continue

            if not text:
                continue

            copied = dict(
                record
            )

            copied[
                "_source_jsonl"
            ] = str(
                jsonl_path.relative_to(
                    PROJECT_ROOT
                )
            )

            copied[
                "_source_line"
            ] = line_number

            # 相同 document_id 出现多次时，
            # 后读取的记录覆盖旧记录。
            records[
                document_id
            ] = copied

    documents: list[
        Document
    ] = []

    for document_id in sorted(
        records
    ):
        record = records[
            document_id
        ]

        title = compact_text(
            record.get(
                "title"
            )
        )

        text = compact_text(
            record.get(
                "text"
            )
        )

        raw_metadata = record.get(
            "metadata"
        )

        if not isinstance(
            raw_metadata,
            dict,
        ):
            raw_metadata = {}

        metadata = {
            str(
                key
            ): to_chroma_metadata(
                value
            )
            for key, value
            in raw_metadata.items()
        }

        metadata.update(
            {
                "document_id": (
                    document_id
                ),

                "doc_id": (
                    document_id
                ),

                "doc_type": compact_text(
                    record.get(
                        "doc_type"
                    )
                ),

                "title": (
                    title
                ),

                "source": compact_text(
                    record.get(
                        "_source_jsonl"
                    )
                ),

                "source_line": int(
                    record.get(
                        "_source_line"
                    )
                    or 0
                ),
            }
        )

        # 标题同时进入正文，有利于项目名和主题名检索。
        page_content = (
            f"{title}\n\n{text}"
            if title
            else text
        )

        documents.append(
            Document(
                page_content=(
                    page_content
                ),
                metadata=metadata,
            )
        )

    if not documents:
        raise RuntimeError(
            "没有读取到有效的 RAG 文档。"
        )

    return (
        documents,
        jsonl_paths,
    )


# ============================================================
# 文档切分
# ============================================================


def build_splitter():
    """
    创建原 RAGLab 的递归切分器。

    同时兼容两种可能的参数接口。
    """
    try:
        return create_recursive_splitter(
            chunk_size=(
                CHUNK_SIZE
            ),
            chunk_overlap=(
                CHUNK_OVERLAP
            ),
        )

    except TypeError:
        return create_recursive_splitter(
            {
                "chunk_size": (
                    CHUNK_SIZE
                ),

                "chunk_overlap": (
                    CHUNK_OVERLAP
                ),
            }
        )


def split_source_documents(
    source_documents: list[
        Document
    ],
) -> list[Document]:
    """
    切分源文档并生成稳定的 chunk_id。
    """
    splitter = build_splitter()

    if not hasattr(
        splitter,
        "split_documents",
    ):
        raise TypeError(
            "create_recursive_splitter "
            "返回对象没有 split_documents 方法。"
        )

    raw_chunks = (
        splitter.split_documents(
            source_documents
        )
    )

    counters: Counter[
        str
    ] = Counter()

    chunks: list[
        Document
    ] = []

    for raw_chunk in raw_chunks:
        if not isinstance(
            raw_chunk,
            Document,
        ):
            continue

        metadata = dict(
            raw_chunk.metadata
        )

        document_id = compact_text(
            metadata.get(
                "document_id"
            )
            or metadata.get(
                "doc_id"
            )
        )

        if not document_id:
            continue

        chunk_index = counters[
            document_id
        ]

        counters[
            document_id
        ] += 1

        chunk_id = (
            f"{document_id}"
            f"::chunk-{chunk_index:03d}"
        )

        metadata.update(
            {
                "document_id": (
                    document_id
                ),

                "doc_id": (
                    document_id
                ),

                "chunk_id": (
                    chunk_id
                ),

                "chunk_index": (
                    chunk_index
                ),

                "section_id": (
                    chunk_id
                ),
            }
        )

        chunks.append(
            Document(
                page_content=(
                    raw_chunk.page_content
                ),
                metadata=metadata,
            )
        )

    if not chunks:
        raise RuntimeError(
            "切分后没有产生有效 Chunk。"
        )

    return chunks


# ============================================================
# BM25
# ============================================================


def build_bm25_index(
    chunks: list[Document],
) -> BM25SearchIndex:
    """
    复用原来的 BM25SearchIndex。

    兼容 documents=、chunks= 和位置参数三种接口。
    """
    errors: list[
        Exception
    ] = []

    for arguments in (
        {
            "documents": (
                chunks
            ),
            "k1": BM25_K1,
            "b": BM25_B,
        },
        {
            "chunks": (
                chunks
            ),
            "k1": BM25_K1,
            "b": BM25_B,
        },
    ):
        try:
            return BM25SearchIndex(
                **arguments
            )

        except TypeError as exc:
            errors.append(
                exc
            )

    try:
        return BM25SearchIndex(
            chunks,
            k1=BM25_K1,
            b=BM25_B,
        )

    except TypeError as exc:
        errors.append(
            exc
        )

    raise TypeError(
        "无法初始化 BM25SearchIndex：\n"
        + "\n".join(
            f"- {error}"
            for error in errors
        )
    )


def save_bm25_assets(
    chunks: list[Document],
    bm25_index: BM25SearchIndex,
) -> dict[str, Any]:
    """
    保存 BM25 Chunk 文档，并尝试序列化索引。

    如果 BM25 对象无法 pickle，
    后续查询仍然可以读取 documents.jsonl
    快速重建索引。
    """
    BM25_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_jsonl(
        BM25_DOCUMENTS_PATH,
        [
            {
                "page_content": (
                    chunk.page_content
                ),

                "metadata": dict(
                    chunk.metadata
                ),
            }
            for chunk in chunks
        ],
    )

    temporary_pickle = (
        BM25_PICKLE_PATH.with_suffix(
            ".pkl.tmp"
        )
    )

    try:
        with temporary_pickle.open(
            "wb"
        ) as file:
            pickle.dump(
                bm25_index,
                file,
                protocol=(
                    pickle.HIGHEST_PROTOCOL
                ),
            )

        temporary_pickle.replace(
            BM25_PICKLE_PATH
        )

        return {
            "pickle_status": (
                "success"
            ),

            "pickle_error": (
                None
            ),

            "pickle_path": str(
                BM25_PICKLE_PATH
            ),
        }

    except Exception as exc:
        if temporary_pickle.exists():
            temporary_pickle.unlink()

        if BM25_PICKLE_PATH.exists():
            BM25_PICKLE_PATH.unlink()

        return {
            "pickle_status": (
                "failed"
            ),

            "pickle_error": str(
                exc
            ),

            "pickle_path": (
                None
            ),
        }


# ============================================================
# 检索结果兼容
# ============================================================


def normalize_results(
    results: Any,
) -> list[Document]:
    """
    将不同检索结果统一转换为 Document 列表。

    支持：

    - Document；
    - (Document, score)；
    - 带 document 属性的结果对象。
    """
    if results is None:
        return []

    if not isinstance(
        results,
        list,
    ):
        results = list(
            results
        )

    documents: list[
        Document
    ] = []

    for item in results:
        if isinstance(
            item,
            Document,
        ):
            documents.append(
                item
            )

        elif (
            isinstance(
                item,
                tuple,
            )
            and item
            and isinstance(
                item[0],
                Document,
            )
        ):
            documents.append(
                item[0]
            )

        elif isinstance(
            getattr(
                item,
                "document",
                None,
            ),
            Document,
        ):
            documents.append(
                item.document
            )

    return documents


def test_bm25(
    bm25_index: BM25SearchIndex,
) -> list[Document]:
    """
    兼容 BM25 search 的不同参数名。
    """
    search_method = getattr(
        bm25_index,
        "search",
        None,
    )

    if not callable(
        search_method
    ):
        raise TypeError(
            "BM25SearchIndex "
            "没有 search 方法。"
        )

    for keyword_arguments in (
        {
            "top_k": TEST_TOP_K,
        },
        {
            "k": TEST_TOP_K,
        },
        {
            "n_results": (
                TEST_TOP_K
            ),
        },
    ):
        try:
            return normalize_results(
                search_method(
                    TEST_QUERY,
                    **keyword_arguments,
                )
            )

        except TypeError:
            continue

    return normalize_results(
        search_method(
            TEST_QUERY,
            TEST_TOP_K,
        )
    )


# ============================================================
# Chroma Dense 索引
# ============================================================


def build_chroma_index(
    chunks: list[Document],
):
    """
    创建 Chroma Dense 索引。

    当前数据量较小，每次完整重建更简单，
    也能够避免已删除文档残留。
    """
    if CHROMA_DIRECTORY.exists():
        shutil.rmtree(
            CHROMA_DIRECTORY
        )

    CHROMA_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    embeddings = (
        create_huggingface_embeddings(
            model_name=(
                EMBEDDING_MODEL
            ),

            normalize_embeddings=(
                NORMALIZE_EMBEDDINGS
            ),

            device=(
                EMBEDDING_DEVICE
            ),
        )
    )

    vector_store = open_chroma_store(
        embeddings=embeddings,

        persist_directory=(
            CHROMA_DIRECTORY
        ),

        collection_name=(
            CHROMA_COLLECTION_NAME
        ),
    )

    vector_store.add_documents(
        documents=chunks,

        ids=[
            str(
                chunk.metadata[
                    "chunk_id"
                ]
            )
            for chunk in chunks
        ],
    )

    # 兼容旧版需要手动 persist 的 Chroma。
    persist_method = getattr(
        vector_store,
        "persist",
        None,
    )

    if callable(
        persist_method
    ):
        persist_method()

    return vector_store


# ============================================================
# 输出测试结果
# ============================================================


def print_results(
    title: str,
    documents: list[Document],
) -> None:
    """
    打印检索测试结果。
    """
    print()
    print(
        title
    )

    if not documents:
        print(
            "  未返回结果"
        )

        return

    for rank, document in enumerate(
        documents,
        start=1,
    ):
        metadata = (
            document.metadata
        )

        display_title = (
            metadata.get("title")
            or metadata.get("document_id")
        )

        print(
            f"  {rank}. {display_title}"
        )

        print(
            "     "
            f"doc_type="
            f"{metadata.get('doc_type')}，"
            f"chunk_id="
            f"{metadata.get('chunk_id')}"
        )


# ============================================================
# 主流程
# ============================================================


def main() -> int:
    """
    构建技术情报 RAG 索引。
    """
    started_at = datetime.now(
        ZoneInfo(
            TIMEZONE
        )
    )

    print(
        "=" * 78
    )

    print(
        "GitHub 技术情报 RAG 索引构建"
    )

    print(
        "=" * 78
    )

    print(
        f"文档源：{RAG_DOCUMENTS_ROOT}"
    )

    print(
        f"Embedding：{EMBEDDING_MODEL}"
    )

    print(
        f"Chunk：size={CHUNK_SIZE}，"
        f"overlap={CHUNK_OVERLAP}"
    )

    print(
        f"BM25：k1={BM25_K1}，"
        f"b={BM25_B}"
    )

    # --------------------------------------------------------
    # 1. 读取文档源
    # --------------------------------------------------------

    print(
        "\n[1/5] 读取 JSONL 文档源……"
    )

    (
        source_documents,
        jsonl_paths,
    ) = load_source_documents()

    print(
        "      JSONL 文件数："
        f"{len(jsonl_paths)}"
    )

    print(
        "      去重后文档数："
        f"{len(source_documents)}"
    )

    # --------------------------------------------------------
    # 2. 文档切分
    # --------------------------------------------------------

    print(
        "\n[2/5] 切分文档……"
    )

    chunks = split_source_documents(
        source_documents
    )

    print(
        f"      Chunk 数：{len(chunks)}"
    )

    # --------------------------------------------------------
    # 3. BM25
    # --------------------------------------------------------

    print(
        "\n[3/5] 构建 BM25 索引……"
    )

    bm25_index = build_bm25_index(
        chunks
    )

    bm25_assets = save_bm25_assets(
        chunks,
        bm25_index,
    )

    print(
        "      Chunk 文档："
        f"{BM25_DOCUMENTS_PATH}"
    )

    print(
        "      Pickle 状态："
        f"{bm25_assets['pickle_status']}"
    )

    if bm25_assets[
        "pickle_error"
    ]:
        print(
            "      提示：BM25 对象无法序列化，"
            "后续查询时将从 documents.jsonl 重建。"
        )

        print(
            "      原因："
            f"{bm25_assets['pickle_error']}"
        )

    # --------------------------------------------------------
    # 4. Chroma
    # --------------------------------------------------------

    print(
        "\n[4/5] 构建 Chroma 向量库……"
    )

    vector_store = build_chroma_index(
        chunks
    )

    print(
        "      Chroma 目录："
        f"{CHROMA_DIRECTORY}"
    )

    print(
        "      Collection："
        f"{CHROMA_COLLECTION_NAME}"
    )

    # --------------------------------------------------------
    # 5. 连通测试
    # --------------------------------------------------------

    print(
        "\n[5/5] 检索连通测试……"
    )

    bm25_results = test_bm25(
        bm25_index
    )

    dense_results = normalize_results(
        vector_store
        .similarity_search_with_score(
            TEST_QUERY,

            k=min(
                TEST_TOP_K,
                len(
                    chunks
                ),
            ),
        )
    )

    print_results(
        "BM25 测试结果：",
        bm25_results,
    )

    print_results(
        "Dense 测试结果：",
        dense_results,
    )

    finished_at = datetime.now(
        ZoneInfo(
            TIMEZONE
        )
    )

    doc_type_counts = Counter(
        str(
            document.metadata.get(
                "doc_type"
            )
            or "unknown"
        )
        for document
        in source_documents
    )

    date_counts = Counter(
        str(
            document.metadata.get(
                "snapshot_date"
            )
            or "unknown"
        )
        for document
        in source_documents
    )

    manifest = {
        "status": "success",

        "started_at": (
            started_at.isoformat()
        ),

        "finished_at": (
            finished_at.isoformat()
        ),

        "elapsed_seconds": round(
            (
                finished_at
                - started_at
            ).total_seconds(),
            3,
        ),

        "source": {
            "root": str(
                RAG_DOCUMENTS_ROOT
            ),

            "jsonl_files": [
                str(
                    path
                )
                for path
                in jsonl_paths
            ],

            "jsonl_file_count": len(
                jsonl_paths
            ),

            "document_count": len(
                source_documents
            ),

            "chunk_count": len(
                chunks
            ),

            "doc_type_counts": dict(
                doc_type_counts
            ),

            "snapshot_date_counts": dict(
                date_counts
            ),
        },

        "splitter": {
            "type": (
                "recursive_character"
            ),

            "chunk_size": (
                CHUNK_SIZE
            ),

            "chunk_overlap": (
                CHUNK_OVERLAP
            ),
        },

        "embedding": {
            "provider": (
                "huggingface"
            ),

            "model": (
                EMBEDDING_MODEL
            ),

            "device": (
                EMBEDDING_DEVICE
            ),

            "normalize_embeddings": (
                NORMALIZE_EMBEDDINGS
            ),
        },

        "chroma": {
            "persist_directory": str(
                CHROMA_DIRECTORY
            ),

            "collection_name": (
                CHROMA_COLLECTION_NAME
            ),

            "indexed_chunk_count": len(
                chunks
            ),
        },

        "bm25": {
            "directory": str(
                BM25_DIRECTORY
            ),

            "documents_path": str(
                BM25_DOCUMENTS_PATH
            ),

            "pickle_path": (
                bm25_assets[
                    "pickle_path"
                ]
            ),

            "pickle_status": (
                bm25_assets[
                    "pickle_status"
                ]
            ),

            "pickle_error": (
                bm25_assets[
                    "pickle_error"
                ]
            ),

            "k1": (
                BM25_K1
            ),

            "b": (
                BM25_B
            ),

            "indexed_chunk_count": len(
                chunks
            ),
        },

        "test": {
            "query": (
                TEST_QUERY
            ),

            "top_k": (
                TEST_TOP_K
            ),

            "bm25_result_count": len(
                bm25_results
            ),

            "dense_result_count": len(
                dense_results
            ),

            "bm25_chunk_ids": [
                document.metadata.get(
                    "chunk_id"
                )
                for document
                in bm25_results
            ],

            "dense_chunk_ids": [
                document.metadata.get(
                    "chunk_id"
                )
                for document
                in dense_results
            ],
        },
    }

    write_json(
        MANIFEST_PATH,
        manifest,
    )

    print(
        "\n" + "=" * 78
    )

    print(
        "RAG 索引构建完成"
    )

    print(
        "=" * 78
    )

    print(
        f"文档数：{len(source_documents)}"
    )

    print(
        f"Chunk 数：{len(chunks)}"
    )

    print(
        f"Chroma：{CHROMA_DIRECTORY}"
    )

    print(
        f"BM25：{BM25_DIRECTORY}"
    )

    print(
        f"Manifest：{MANIFEST_PATH}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )