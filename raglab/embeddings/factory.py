"""Embedding 模型创建模块。

当前基线使用本地 Hugging Face Embedding 模型。

本模块只负责创建 Embedding 对象，不负责：

- 加载 PDF；
- 文本分块；
- 保存向量；
- 执行检索。
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings


def create_huggingface_embeddings(
    model_name: str,
    normalize_embeddings: bool = True,
    device: str = "cpu",
) -> Embeddings:
    """创建 Hugging Face Embedding 模型。

    Args:
        model_name:
            Hugging Face 模型名称，例如：
            BAAI/bge-small-zh-v1.5。

        normalize_embeddings:
            是否将每个向量归一化为单位向量。

        device:
            模型运行设备。
            当前基线使用 cpu，后续可以改为 cuda。

    Returns:
        满足 LangChain Embeddings 接口的模型对象。
    """
    if not model_name.strip():
        raise ValueError("Embedding 模型名称不能为空。")

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={
            "device": device,
        },
        encode_kwargs={
            "normalize_embeddings": normalize_embeddings,
        },
    )