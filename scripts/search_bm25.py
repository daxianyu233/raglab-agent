"""手动检查 BM25 检索结果。"""

from __future__ import annotations

import yaml

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
)


BM25_CONFIG_PATH = (
    CONFIG_DIR / "bm25.yaml"
)


def main() -> None:
    with BM25_CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    splitter_config = config["splitter"]
    retrieval_config = config["retrieval"]

    page_documents = load_pdf_corpus()

    splitter = create_recursive_splitter(
        chunk_size=int(
            splitter_config["chunk_size"]
        ),
        chunk_overlap=int(
            splitter_config["chunk_overlap"]
        ),
    )

    chunks = split_page_documents(
        documents=page_documents,
        text_splitter=splitter,
    )

    print("正在建立 BM25 索引……")

    bm25_index = BM25SearchIndex(
        documents=chunks,
        k1=float(
            retrieval_config["k1"]
        ),
        b=float(
            retrieval_config["b"]
        ),
    )

    top_k = int(
        retrieval_config["top_k"]
    )

    print(f"Chunk 数量：{len(chunks)}")
    print("输入 q 或直接回车退出。")

    while True:
        query = input(
            "\n请输入问题："
        ).strip()

        if not query or query.lower() == "q":
            break

        results = bm25_index.search(
            query=query,
            top_k=top_k,
        )

        for rank, (
            document,
            score,
        ) in enumerate(
            results,
            start=1,
        ):
            preview = " ".join(
                document.page_content.split()
            )

            print("\n" + "-" * 70)
            print(f"排名：{rank}")
            print(
                f"BM25 score：{score:.6f}"
            )
            print(
                "chunk_id："
                f"{document.metadata.get('chunk_id')}"
            )
            print(
                "title："
                f"{document.metadata.get('title')}"
            )
            print(
                "文本："
                f"{preview[:300]}"
            )


if __name__ == "__main__":
    main()