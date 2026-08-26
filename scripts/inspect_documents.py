"""检查 PDF Loader 的输出结果。"""

from pprint import pprint

from raglab.ingestion.loaders import load_pdf_corpus
from raglab.settings import PDF_CORPUS_DIR


def main() -> None:
    documents = load_pdf_corpus()

    pdf_count = len(
        list(PDF_CORPUS_DIR.glob("*.pdf"))
    )

    print("=" * 70)
    print("RAGLab PDF 加载结果")
    print("=" * 70)
    print(f"PDF 文件数量：{pdf_count}")
    print(f"Document 数量：{len(documents)}")

    print("\n前 3 个 Document：")

    for index, document in enumerate(
        documents[:3],
        start=1,
    ):
        print("\n" + "-" * 70)
        print(f"Document {index}")
        print("-" * 70)

        print("metadata：")
        pprint(document.metadata)

        content_preview = (
            document.page_content
            .strip()
            .replace("\n", " ")
        )

        print("\npage_content 前 300 个字符：")
        print(content_preview[:300])

        print(
            "\n当前页面字符数："
            f"{len(document.page_content)}"
        )


if __name__ == "__main__":
    main()