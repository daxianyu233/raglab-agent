"""文档加载模块。

当前阶段只负责：
1. 读取语料清单；
2. 加载 PDF；
3. 为每一页补充统一 metadata。

暂时不负责文本分块。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from raglab.settings import CORPUS_MANIFEST_PATH, PDF_CORPUS_DIR


def load_corpus_manifest(
    manifest_path: Path = CORPUS_MANIFEST_PATH,
) -> dict[str, dict[str, Any]]:
    """读取语料清单，并按 PDF 文件名建立索引。

    返回示例：
    {
        "01_langchain_runnable_chain.pdf": {
            "doc_id": "DOC-LC-001",
            "title": "LangChain Runnable 与链式编排",
            ...
        }
    }
    """
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"语料清单不存在：{manifest_path}"
        )

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        manifest_rows = json.load(file)

    manifest_by_filename: dict[str, dict[str, Any]] = {}

    for row in manifest_rows:
        pdf_file = row.get("pdf_file")

        if not pdf_file:
            raise ValueError(
                f"语料清单记录缺少 pdf_file：{row}"
            )

        filename = Path(pdf_file).name
        manifest_by_filename[filename] = row

    return manifest_by_filename


def load_single_pdf(
    pdf_path: Path,
    manifest_metadata: dict[str, Any],
) -> list[Document]:
    """加载一个 PDF，并给每一页补充统一元数据。"""
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF 文件不存在：{pdf_path}"
        )

    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    for fallback_page_index, document in enumerate(
        documents,
        start=1,
    ):
        raw_page_index = document.metadata.get("page")

        if isinstance(raw_page_index, int):
            page_number = raw_page_index + 1
        else:
            page_number = fallback_page_index

        document.metadata.update(
            {
                "doc_id": manifest_metadata["doc_id"],
                "title": manifest_metadata["title"],
                "corpus": manifest_metadata["corpus"],
                "language": manifest_metadata["language"],
                "source_type": manifest_metadata[
                    "source_type"
                ],
                "source_file": pdf_path.name,
                "source_path": str(pdf_path.resolve()),
                "page_number": page_number,
            }
        )

    return documents


def load_pdf_corpus(
    pdf_dir: Path = PDF_CORPUS_DIR,
    manifest_path: Path = CORPUS_MANIFEST_PATH,
) -> list[Document]:
    """加载目录中的全部 PDF。

    一个 PDF 有多少页，通常就会产生多少个原始 Document。
    """
    if not pdf_dir.exists():
        raise FileNotFoundError(
            f"PDF 语料目录不存在：{pdf_dir}"
        )

    manifest = load_corpus_manifest(manifest_path)
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_paths:
        raise FileNotFoundError(
            f"PDF 目录中没有找到文件：{pdf_dir}"
        )

    all_documents: list[Document] = []

    for pdf_path in pdf_paths:
        metadata = manifest.get(pdf_path.name)

        if metadata is None:
            raise KeyError(
                "PDF 未在语料清单中登记："
                f"{pdf_path.name}"
            )

        documents = load_single_pdf(
            pdf_path=pdf_path,
            manifest_metadata=metadata,
        )

        all_documents.extend(documents)

    return all_documents