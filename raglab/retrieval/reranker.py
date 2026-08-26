"""基于 BGE Cross-Encoder 的候选文档重排模块。

工作流程：

查询 + 候选 Document
→ 构造 Query-Passage 文本对
→ BGE Reranker 计算相关性分数
→ 按相关性分数降序排列
→ 返回重排后的 Document

Reranker 不负责从完整语料中检索新文档，
它只对第一阶段检索器已经召回的候选文档进行重新排序。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from langchain_core.documents import Document
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


@dataclass(frozen=True)
class RerankedResult:
    """保存单条重排结果。"""

    document: Document
    score: float
    original_rank: int


def resolve_device(
    device: str | None = None,
) -> torch.device:
    """确定模型运行设备。

    Parameters
    ----------
    device:
        可以显式指定：

        cpu
        cuda
        cuda:0

        如果为 None，则自动判断是否存在可用 CUDA。
    """

    if device is not None:
        resolved = torch.device(device)

        if (
            resolved.type == "cuda"
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "配置要求使用 CUDA，"
                "但是当前环境没有可用的 CUDA 设备。"
            )

        return resolved

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


class BGEReranker:
    """使用 BGE Cross-Encoder 对候选文档进行重排。"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        *,
        device: str | None = None,
        batch_size: int = 8,
        max_length: int = 512,
        use_fp16: bool = False,
    ) -> None:
        """初始化 Reranker。

        Parameters
        ----------
        model_name:
            Hugging Face 模型名称或本地模型路径。

        device:
            模型运行设备。为 None 时自动选择 CUDA 或 CPU。

        batch_size:
            每次送入模型的 Query-Passage 对数量。

        max_length:
            Query 与 Passage 拼接后的最大 Token 长度。

        use_fp16:
            是否在 CUDA 上使用半精度推理。
            CPU 环境下该参数会被忽略。
        """

        if batch_size <= 0:
            raise ValueError(
                "batch_size 必须大于 0。"
            )

        if max_length <= 0:
            raise ValueError(
                "max_length 必须大于 0。"
            )

        self.model_name = str(model_name)
        self.device = resolve_device(device)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)

        self.use_fp16 = bool(
            use_fp16
            and self.device.type == "cuda"
        )

        print(
            "正在加载 Reranker Tokenizer："
            f"{self.model_name}"
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                self.model_name
            )
        )

        print(
            "正在加载 Reranker 模型："
            f"{self.model_name}"
        )

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                self.model_name
            )
        )

        self.model.to(self.device)

        if self.use_fp16:
            self.model.half()

        self.model.eval()

        print(
            "Reranker 加载完成："
            f"device={self.device}, "
            f"batch_size={self.batch_size}, "
            f"max_length={self.max_length}, "
            f"fp16={self.use_fp16}"
        )

    def _validate_documents(
        self,
        documents: Sequence[Document],
    ) -> None:
        """检查候选文档是否合法。"""

        for index, document in enumerate(
            documents
        ):
            if not isinstance(
                document,
                Document,
            ):
                raise TypeError(
                    f"候选项[{index}] 不是 Document，"
                    f"实际类型：{type(document)!r}"
                )

            if not document.page_content.strip():
                raise ValueError(
                    f"候选 Document[{index}] "
                    "的 page_content 为空。"
                )

    def score(
        self,
        query: str,
        documents: Sequence[Document],
    ) -> list[float]:
        """计算查询与每个候选文档的相关性分数。

        Parameters
        ----------
        query:
            用户查询。

        documents:
            第一阶段检索得到的候选 Document。

        Returns
        -------
        list[float]
            每个候选文档对应的 Reranker 分数。
            分数越大，表示模型认为越相关。
        """

        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError(
                "query 不能为空。"
            )

        if not documents:
            return []

        self._validate_documents(documents)

        all_scores: list[float] = []

        for batch_start in range(
            0,
            len(documents),
            self.batch_size,
        ):
            batch_documents = documents[
                batch_start:
                batch_start + self.batch_size
            ]

            pairs = [
                [
                    normalized_query,
                    document.page_content,
                ]
                for document in batch_documents
            ]

            encoded = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            encoded = {
                key: value.to(self.device)
                for key, value in encoded.items()
            }

            with torch.inference_mode():
                outputs = self.model(
                    **encoded,
                    return_dict=True,
                )

                logits = outputs.logits

            if logits.ndim == 1:
                batch_scores = logits

            elif logits.ndim == 2:
                if logits.shape[1] == 1:
                    batch_scores = logits[:, 0]
                else:
                    # 兼容输出多个类别分数的模型。
                    # 默认使用最后一个类别作为相关类别。
                    batch_scores = logits[:, -1]

            else:
                raise RuntimeError(
                    "无法识别 Reranker logits 形状："
                    f"{tuple(logits.shape)}"
                )

            batch_scores = (
                batch_scores
                .float()
                .detach()
                .cpu()
                .tolist()
            )

            all_scores.extend(
                float(score)
                for score in batch_scores
            )

        if len(all_scores) != len(documents):
            raise RuntimeError(
                "Reranker 输出分数数量与候选文档数量不一致："
                f"scores={len(all_scores)}, "
                f"documents={len(documents)}"
            )

        return all_scores

    def rerank(
        self,
        query: str,
        documents: Sequence[Document],
        *,
        top_k: int | None = None,
    ) -> list[Document]:
        """对候选文档进行重排。

        Parameters
        ----------
        query:
            用户查询。

        documents:
            第一阶段检索得到的候选文档。

        top_k:
            重排后保留的文档数量。
            为 None 时保留全部候选。

        Returns
        -------
        list[Document]
            按 Reranker 分数降序排列的 Document。
        """

        if not documents:
            return []

        if top_k is None:
            final_top_k = len(documents)
        else:
            final_top_k = int(top_k)

        if final_top_k <= 0:
            raise ValueError(
                "top_k 必须大于 0。"
            )

        scores = self.score(
            query=query,
            documents=documents,
        )

        reranked_results: list[
            RerankedResult
        ] = []

        for original_rank, (
            document,
            score,
        ) in enumerate(
            zip(
                documents,
                scores,
                strict=True,
            ),
            start=1,
        ):
            reranked_results.append(
                RerankedResult(
                    document=document,
                    score=float(score),
                    original_rank=original_rank,
                )
            )

        def sort_key(
            result: RerankedResult,
        ) -> tuple[float, int, str]:
            """生成稳定排序键。"""

            chunk_id = str(
                result.document.metadata.get(
                    "chunk_id",
                    "",
                )
            )

            return (
                -result.score,
                result.original_rank,
                chunk_id,
            )

        reranked_results.sort(
            key=sort_key
        )

        output: list[Document] = []

        for rerank_rank, result in enumerate(
            reranked_results[:final_top_k],
            start=1,
        ):
            metadata = dict(
                result.document.metadata
            )

            metadata.update(
                {
                    "retrieval_method": (
                        "hybrid_rrf_bge_reranker"
                    ),
                    "reranker_model": (
                        self.model_name
                    ),
                    "reranker_score": (
                        float(result.score)
                    ),
                    "pre_rerank_rank": (
                        result.original_rank
                    ),
                    "rerank_rank": (
                        rerank_rank
                    ),
                }
            )

            output.append(
                Document(
                    page_content=(
                        result.document.page_content
                    ),
                    metadata=metadata,
                )
            )

        return output