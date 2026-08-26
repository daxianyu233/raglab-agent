"""查看文本分块结果。

这个脚本本身不负责实现 PDF 加载或文本分块算法，
而是调用项目中已经写好的功能，并把中间结果打印出来。

整体执行流程：

baseline.yaml
    ↓ 读取 chunk_size 和 chunk_overlap
PDF 文件
    ↓ load_pdf_corpus()
页面级 Document
    ↓ split_page_documents()
Chunk 级 Document
    ↓
打印并检查分块结果
"""

# Counter 用于统计每个 doc_id 一共产生了多少个 Chunk。
from collections import Counter

# pprint 可以将字典等复杂对象以更清晰的格式打印出来。
from pprint import pprint

# 用于读取 YAML 配置文件。
import yaml

# 导入 PDF 语料加载函数。
# 该函数会把所有 PDF 按页读取为 list[Document]。
from raglab.ingestion.loaders import load_pdf_corpus

# 导入文本切分相关函数。
from raglab.ingestion.splitters import (
    # 根据 chunk_size 和 chunk_overlap 创建 LangChain 切分器。
    create_recursive_splitter,

    # 使用切分器，将页面级 Document 切成 Chunk 级 Document。
    split_page_documents,
)

# baseline.yaml 配置文件的统一路径。
from raglab.settings import BASELINE_CONFIG_PATH


def load_splitter_config() -> tuple[int, int]:
    """从 baseline.yaml 中读取文本分块参数。

    Returns:
        tuple[int, int]:
            第一个值为 chunk_size；
            第二个值为 chunk_overlap。

    例如：
        (600, 80)
    """

    # 使用 utf-8 编码打开 baseline.yaml。
    with BASELINE_CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        # yaml.safe_load() 会把 YAML 内容转成 Python 字典。
        config = yaml.safe_load(file)

    # 获取配置文件中的 splitter 部分。
    #
    # 假设 YAML 内容是：
    #
    # splitter:
    #   chunk_size: 600
    #   chunk_overlap: 80
    #
    # 那么 splitter_config 得到：
    #
    # {
    #     "chunk_size": 600,
    #     "chunk_overlap": 80
    # }
    splitter_config = config["splitter"]

    # 读取 chunk_size，并确保转换为整数。
    chunk_size = int(
        splitter_config["chunk_size"]
    )

    # 读取 chunk_overlap，并确保转换为整数。
    chunk_overlap = int(
        splitter_config["chunk_overlap"]
    )

    # 将两个参数作为元组返回。
    return chunk_size, chunk_overlap


def normalize_preview(text: str) -> str:
    """整理文本格式，方便在终端中查看。

    原始 PDF 文本中可能包含：

    - 多个连续换行；
    - 制表符；
    - 多余空格；
    - 页内排版造成的断行。

    这里使用 split() 将各种空白字符拆开，
    再使用单个空格重新连接。

    例如：

        "第一段\\n\\n第二段    内容"

    会变成：

        "第一段 第二段 内容"

    注意：
        这个函数只用于打印预览，
        不会修改真正用于分块和检索的原始文本。
    """

    return " ".join(
        text.strip().split()
    )


def main() -> None:
    """执行文档加载、文本分块和结果检查。"""

    # ---------------------------------------------------------
    # 第一步：读取文本切分配置
    # ---------------------------------------------------------

    # load_splitter_config() 返回一个元组：
    #
    # (chunk_size, chunk_overlap)
    #
    # 这里使用元组解包，分别赋值给两个变量。
    chunk_size, chunk_overlap = (
        load_splitter_config()
    )

    # ---------------------------------------------------------
    # 第二步：加载 PDF，得到页面级 Document
    # ---------------------------------------------------------

    # load_pdf_corpus() 会：
    #
    # 1. 查找 data/corpus/pdf 下的所有 PDF；
    # 2. 使用 PyPDFLoader 逐页读取 PDF；
    # 3. 每一页转换为一个 LangChain Document；
    # 4. 为每页补充 doc_id、page_number 等 metadata；
    # 5. 返回 list[Document]。
    #
    # 此时一个 Document 通常代表一个 PDF 页面。
    page_documents = load_pdf_corpus()

    # ---------------------------------------------------------
    # 第三步：创建文本切分器
    # ---------------------------------------------------------

    # 根据配置文件中的参数创建
    # RecursiveCharacterTextSplitter。
    #
    # 例如：
    #
    # chunk_size = 600
    # chunk_overlap = 80
    #
    # 表示：
    # 每个 Chunk 目标最大长度约为 600 个字符，
    # 相邻 Chunk 尽量保留约 80 个字符的重叠内容。
    text_splitter = create_recursive_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # ---------------------------------------------------------
    # 第四步：将页面级 Document 切成 Chunk 级 Document
    # ---------------------------------------------------------

    # 输入：
    #     list[页面级 Document]
    #
    # 输出：
    #     list[Chunk 级 Document]
    #
    # Chunk 仍然是 LangChain Document 对象，
    # 只是 page_content 变成了较短的一段文本。
    #
    # 原页面的 metadata 会被继承，
    # 同时会增加：
    #
    # chunk_id
    # chunk_index
    # chunk_char_count
    # start_index
    chunks = split_page_documents(
        documents=page_documents,
        text_splitter=text_splitter,
    )

    # ---------------------------------------------------------
    # 第五步：打印整体统计信息
    # ---------------------------------------------------------

    print("=" * 70)
    print("RAGLab 文本分块结果")
    print("=" * 70)

    # 页面级 Document 数量通常等于所有 PDF 的总页数。
    print(
        f"页面级 Document 数量："
        f"{len(page_documents)}"
    )

    # Chunk 数量通常大于页面数量，
    # 因为一个较长页面可能会被切成多个 Chunk。
    print(
        f"Chunk 数量：{len(chunks)}"
    )

    # 打印本次实验实际使用的切分参数。
    print(
        f"chunk_size：{chunk_size}"
    )

    print(
        f"chunk_overlap：{chunk_overlap}"
    )

    # ---------------------------------------------------------
    # 第六步：统计每份文档产生了多少个 Chunk
    # ---------------------------------------------------------

    # Counter 会统计每一个 doc_id 出现的次数。
    #
    # 这里每一个 Chunk 都有一个 doc_id，
    # 因此某个 doc_id 出现多少次，
    # 就表示该文档被切成了多少个 Chunk。
    chunk_counts = Counter(
        chunk.metadata["doc_id"]
        for chunk in chunks
    )

    print(
        "\n每份文档的 Chunk 数量："
    )

    # sorted() 按 doc_id 排序，
    # 让终端输出顺序更加稳定和清晰。
    for doc_id, count in sorted(
        chunk_counts.items()
    ):
        print(
            f"  {doc_id}: {count}"
        )

    # ---------------------------------------------------------
    # 第七步：选择第一个页面作为观察样本
    # ---------------------------------------------------------

    # 从所有页面级 Document 中取第一个页面。
    first_page = page_documents[0]

    # 读取这个页面所属的文档编号。
    first_doc_id = first_page.metadata[
        "doc_id"
    ]

    # 读取自然页码。
    first_page_number = first_page.metadata[
        "page_number"
    ]

    # 从全部 Chunk 中筛选：
    #
    # 1. doc_id 与第一个页面相同；
    # 2. page_number 与第一个页面相同。
    #
    # 得到的就是：
    # 第一个页面被切分后产生的所有 Chunk。
    first_page_chunks = [
        chunk
        for chunk in chunks
        if (
            chunk.metadata["doc_id"]
            == first_doc_id

            and

            chunk.metadata["page_number"]
            == first_page_number
        )
    ]

    # ---------------------------------------------------------
    # 第八步：打印第一个页面的分块详情
    # ---------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "第一个页面的分块详情"
    )

    print("=" * 70)

    # 当前页面属于哪份文档。
    print(
        f"doc_id：{first_doc_id}"
    )

    # 当前是该 PDF 的第几页。
    print(
        f"page_number："
        f"{first_page_number}"
    )

    # 原始页面文本的字符数量。
    print(
        "原始页面字符数："
        f"{len(first_page.page_content)}"
    )

    # 当前页面最终被切成多少个 Chunk。
    print(
        "当前页面 Chunk 数量："
        f"{len(first_page_chunks)}"
    )

    # ---------------------------------------------------------
    # 第九步：逐个打印第一个页面的 Chunk
    # ---------------------------------------------------------

    for chunk in first_page_chunks:
        print(
            "\n" + "-" * 70
        )

        # chunk_id 是我们自己生成的唯一编号。
        #
        # 例如：
        #
        # DOC-LC-001-P001-C000
        #
        # 表示：
        # DOC-LC-001：文档编号
        # P001：第 1 页
        # C000：该页第 1 个 Chunk
        print(
            "chunk_id："
            f"{chunk.metadata['chunk_id']}"
        )

        # start_index 表示：
        # 当前 Chunk 在原始页面字符串中的起始字符位置。
        #
        # 该字段由：
        #
        # add_start_index=True
        #
        # 产生。
        print(
            "start_index："
            f"{chunk.metadata.get('start_index')}"
        )

        # 当前 Chunk 的实际字符数量。
        print(
            "chunk_char_count："
            f"{chunk.metadata['chunk_char_count']}"
        )

        # 打印当前 Chunk 的完整 metadata。
        print("metadata：")
        pprint(
            chunk.metadata
        )

        # 打印当前 Chunk 的文本内容。
        print("\nChunk 内容：")

        # normalize_preview() 只整理显示格式，
        # 不会改变原始 Chunk 对象。
        print(
            normalize_preview(
                chunk.page_content
            )
        )

    # ---------------------------------------------------------
    # 第十步：检查相邻 Chunk 的边界和重叠内容
    # ---------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "相邻 Chunk 边界观察"
    )

    print("=" * 70)

    # 假设第一个页面有 3 个 Chunk：
    #
    # Chunk 0
    # Chunk 1
    # Chunk 2
    #
    # 那么这里会依次比较：
    #
    # Chunk 0 与 Chunk 1
    # Chunk 1 与 Chunk 2
    #
    # 因此循环次数是：
    #
    # len(first_page_chunks) - 1
    for index in range(
        len(first_page_chunks) - 1
    ):
        # 当前 Chunk。
        current_chunk = (
            first_page_chunks[index]
        )

        # 下一个相邻 Chunk。
        next_chunk = (
            first_page_chunks[index + 1]
        )

        # 取当前 Chunk 最后 100 个字符，
        # 用于观察它在什么位置结束。
        current_tail = normalize_preview(
            current_chunk.page_content[-100:]
        )

        # 取下一个 Chunk 最前面 100 个字符，
        # 用于观察它从什么内容开始。
        next_head = normalize_preview(
            next_chunk.page_content[:100]
        )

        print(
            f"\nChunk {index} 尾部："
        )

        print(
            current_tail
        )

        print(
            f"Chunk {index + 1} 头部："
        )

        print(
            next_head
        )


# 只有直接运行当前模块时，才执行 main()。
#
# 例如：
#
# python -m scripts.inspect_chunks
#
# 如果该文件被其他模块 import，
# main() 不会自动执行。
if __name__ == "__main__":
    main()