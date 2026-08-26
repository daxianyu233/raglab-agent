
"""BM25 稀疏检索模块。

处理流程：

Chunk Document
→ 中英文混合分词
→ 构建 BM25 索引
→ 查询分词
→ 计算 BM25 分数
→ 返回 Top-K Chunk

BM25 与 Dense Retrieval 的主要区别：

Dense：
    文本 → Embedding 向量 → 向量相似度

BM25：
    文本 → 词项序列 → 词频和逆文档频率评分
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import jieba
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


# 保留以下内容：
#
# 1. 中文字符；
# 2. 英文字母；
# 3. 数字；
# 4. 下划线；
# 5. 技术词中常见的点号、加号和减号。
#
# 例如：
#
# chunk_id
# top_k
# langchain-core
# gpt-4.1
TOKEN_PATTERN = re.compile(
    r"[\u4e00-\u9fff]+|"
    r"[a-zA-Z0-9_][a-zA-Z0-9_.+\-]*"
)


def tokenize_mixed_text(
    text: str,
) -> list[str]:
    """对中英文混合技术文本进行分词。

    中文由 jieba 分词；
    英文统一转换成小写；
    标点符号和空白字符被过滤。
    """
    normalized_text = text.lower().strip()

    if not normalized_text:
        return []

    tokens: list[str] = []

    # jieba 负责先处理中文词语边界。
    for segment in jieba.lcut(
        normalized_text,
        cut_all=False,
    ):
        cleaned_segment = segment.strip()

        if not cleaned_segment:
            continue

        # 再通过正则保留中文、英文和技术标识符。
        matched_tokens = TOKEN_PATTERN.findall(
            cleaned_segment
        )

        tokens.extend(matched_tokens)

    return tokens


class BM25SearchIndex:
    """基于 Chunk Document 的内存 BM25 索引。"""

    def __init__(
        self,
        documents: Sequence[Document],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not documents:
            raise ValueError(
                "Document 列表为空，不能构建 BM25 索引。"
            )

        if k1 <= 0:
            raise ValueError(
                f"k1 必须大于 0，当前值：{k1}"
            )

        if not 0 <= b <= 1:
            raise ValueError(
                f"b 必须位于 [0, 1]，当前值：{b}"
            )

        # 保存 Document，方便根据排名下标取回原始文本。
        self.documents = list(documents)

        # 将每个 Chunk 转换成词项列表。
        #
        # 例如：
        #
        # "Vector Store 保存向量"
        #
        # 可能转换为：
        #
        # ["vector", "store", "保存", "向量"]
        self.tokenized_corpus = [
            tokenize_mixed_text(
                document.page_content
            )
            for document in self.documents
        ]

        # 防止出现完全没有有效词项的 Chunk。
        empty_indices = [
            index
            for index, tokens in enumerate(
                self.tokenized_corpus
            )
            if not tokens
        ]

        if empty_indices:
            raise ValueError(
                "检测到分词结果为空的 Chunk，"
                f"位置：{empty_indices[:10]}"
            )

        # 根据整个语料的词频、文档频率和文档长度
        # 建立 BM25Okapi 索引。
        self.bm25 = BM25Okapi(
            self.tokenized_corpus,
            k1=k1,
            b=b,
        )

    def search(
        self,
        query: str,
        top_k: int,
    ) -> list[tuple[Document, float]]:
        """检索 BM25 分数最高的 Top-K Chunk。

        Returns:
            list[tuple[Document, float]]:

            每个元素包含：
            - Chunk Document；
            - BM25 分数。

            BM25 分数越高通常表示匹配越强。
        """
        cleaned_query = query.strip()

        if not cleaned_query:
            raise ValueError(
                "查询文本不能为空。"
            )

        if top_k <= 0:
            raise ValueError(
                f"top_k 必须大于 0，当前值：{top_k}"
            )

        query_tokens = tokenize_mixed_text(
            cleaned_query
        )

        if not query_tokens:
            return []

        # 为语料库中的每一个 Chunk 计算 BM25 分数。
        scores = self.bm25.get_scores(
            query_tokens
        )

        # scores[index] 对应 self.documents[index]。
        ranked_indices = sorted(
            range(len(self.documents)),
            key=lambda index: float(
                scores[index]
            ),
            reverse=True,
        )

        result_count = min(
            top_k,
            len(ranked_indices),
        )

        results: list[
            tuple[Document, float]
        ] = []

        for index in ranked_indices[
            :result_count
        ]:
            results.append(
                (
                    self.documents[index],
                    float(scores[index]),
                )
            )

        return results