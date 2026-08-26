"""使用 BM25 + DeepSeek V4 执行单步 RAG 问答。

完整流程：

用户问题
→ 加载 PDF
→ 按 BM25 配置切分 Chunk
→ 构建 BM25 内存索引
→ 检索 Top-K
→ 构造受控上下文
→ 构造 RAG Prompt
→ 调用 DeepSeek V4
→ 输出答案、来源、Token 与耗时

当前脚本是单步 RAG：

一次问题
→ 一次检索
→ 一次模型调用
→ 一次回答
→ 程序结束

它暂时不保存多轮历史，也不包含 Agent 工具循环。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from raglab.generation.chat_model_factory import (
    create_chat_model,
)
from raglab.generation.rag_chain import (
    RAGAnswer,
    RAGChain,
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
from raglab.settings import (
    CONFIG_DIR,
    PROJECT_ROOT,
)


DEFAULT_CONFIG_PATH = (
    CONFIG_DIR
    / "generation.yaml"
)


def parse_args() -> argparse.Namespace:
    """读取命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "使用 BM25 和 DeepSeek V4 "
            "执行一次 RAG 问答。"
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="generation.yaml 配置文件路径。",
    )

    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help=(
            "需要回答的问题。"
            "未提供时，会在控制台要求输入。"
        ),
    )

    parser.add_argument(
        "--show-context",
        action="store_true",
        help="强制显示实际发送给模型的上下文。",
    )

    return parser.parse_args()


def load_yaml_config(
    config_path: Path,
) -> dict[str, Any]:
    """读取 YAML 配置文件。"""

    if not config_path.exists():
        raise FileNotFoundError(
            f"配置文件不存在：{config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "配置文件根节点必须是字典："
            f"{config_path}"
        )

    return config


def resolve_project_path(
    path_value: str,
) -> Path:
    """将项目相对路径转换为绝对路径。"""

    path = Path(path_value)

    if path.is_absolute():
        return path

    return (
        PROJECT_ROOT
        / path
    ).resolve()


def require_mapping(
    config: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    """读取必须存在的字典配置节。"""

    value = config.get(key)

    if not isinstance(value, dict):
        raise ValueError(
            f"配置缺少有效字典节点：{key}"
        )

    return value


def require_string(
    config: dict[str, Any],
    key: str,
) -> str:
    """读取必须存在的字符串配置。"""

    value = config.get(key)

    if value is None:
        raise ValueError(
            f"配置缺少字段：{key}"
        )

    normalized = str(value).strip()

    if not normalized:
        raise ValueError(
            f"配置字段不能为空：{key}"
        )

    return normalized


def build_bm25_index(
    bm25_config_path: Path,
) -> tuple[
    BM25SearchIndex,
    dict[str, Any],
]:
    """根据现有 bm25.yaml 构建 BM25 索引。"""

    bm25_config = load_yaml_config(
        bm25_config_path
    )

    splitter_config = require_mapping(
        bm25_config,
        "splitter",
    )

    retrieval_config = require_mapping(
        bm25_config,
        "retrieval",
    )

    chunk_size = int(
        splitter_config["chunk_size"]
    )

    chunk_overlap = int(
        splitter_config["chunk_overlap"]
    )

    k1 = float(
        retrieval_config.get(
            "k1",
            1.5,
        )
    )

    b = float(
        retrieval_config.get(
            "b",
            0.75,
        )
    )

    print()
    print("[1/4] 加载 PDF 语料……")

    page_documents = load_pdf_corpus()

    print(
        f"      页面数量："
        f"{len(page_documents)}"
    )

    print(
        "[2/4] 构建 Chunk："
        f"chunk_size={chunk_size}, "
        f"chunk_overlap={chunk_overlap}"
    )

    splitter = create_recursive_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = split_page_documents(
        documents=page_documents,
        text_splitter=splitter,
    )

    print(
        f"      Chunk 数量：{len(chunks)}"
    )

    print(
        "[3/4] 构建 BM25 索引："
        f"k1={k1}, b={b}"
    )

    bm25_index = BM25SearchIndex(
        documents=chunks,
        k1=k1,
        b=b,
    )

    build_info = {
        "page_count": len(
            page_documents
        ),
        "chunk_count": len(chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "k1": k1,
        "b": b,
    }

    return (
        bm25_index,
        build_info,
    )


def create_deepseek_model(
    model_config: dict[str, Any],
) -> Any:
    """根据 generation.yaml 创建 DeepSeek 模型。"""

    provider = require_string(
        model_config,
        "provider",
    )

    model_name = require_string(
        model_config,
        "model_name",
    )

    api_key_env = str(
        model_config.get(
            "api_key_env",
            "DEEPSEEK_API_KEY",
        )
    )

    base_url = str(
        model_config.get(
            "base_url",
            "https://api.deepseek.com",
        )
    )

    temperature = float(
        model_config.get(
            "temperature",
            0.0,
        )
    )

    max_output_tokens_value = (
        model_config.get(
            "max_output_tokens",
            1024,
        )
    )

    max_output_tokens = (
        None
        if max_output_tokens_value is None
        else int(max_output_tokens_value)
    )

    timeout_value = model_config.get(
        "timeout",
        120,
    )

    timeout = (
        None
        if timeout_value is None
        else float(timeout_value)
    )

    max_retries = int(
        model_config.get(
            "max_retries",
            2,
        )
    )

    thinking_enabled = bool(
        model_config.get(
            "thinking_enabled",
            False,
        )
    )

    reasoning_effort = str(
        model_config.get(
            "reasoning_effort",
            "high",
        )
    )

    print(
        "[4/4] 创建 DeepSeek 模型："
        f"{model_name}"
    )

    print(
        "      思考模式："
        f"{'开启' if thinking_enabled else '关闭'}"
    )

    return create_chat_model(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        max_output_tokens=(
            max_output_tokens
        ),
        timeout=timeout,
        max_retries=max_retries,
        api_key_env=api_key_env,
        base_url=base_url,
        thinking_enabled=(
            thinking_enabled
        ),
        reasoning_effort=(
            reasoning_effort
        ),
    )


def resolve_question(
    command_line_question: str | None,
) -> str:
    """读取并检查用户问题。"""

    if command_line_question is None:
        question = input(
            "\n请输入问题："
        )
    else:
        question = (
            command_line_question
        )

    normalized = question.strip()

    if not normalized:
        raise ValueError(
            "问题不能为空。"
        )

    return normalized


def print_sources(
    result: RAGAnswer,
) -> None:
    """打印实际进入上下文的资料来源。"""

    print()
    print("=" * 80)
    print("资料来源")
    print("=" * 80)

    if not result.references:
        print("没有资料进入最终上下文。")
        return

    for reference in result.references:
        title = (
            reference.title
            or "未命名文档"
        )

        doc_id = (
            reference.doc_id
            or "N/A"
        )

        chunk_id = (
            reference.chunk_id
            or "N/A"
        )

        page_number = (
            reference.page_number
            if reference.page_number
            is not None
            else "N/A"
        )

        print(
            f"{reference.reference_id}："
            f"title={title} | "
            f"doc_id={doc_id} | "
            f"chunk_id={chunk_id} | "
            f"page={page_number}"
        )


def print_usage(
    usage_metadata: dict[str, Any],
) -> None:
    """打印模型 Token 使用信息。"""

    print()
    print("=" * 80)
    print("Token 使用")
    print("=" * 80)

    if not usage_metadata:
        print(
            "模型响应中没有返回可识别的 "
            "Token 使用数据。"
        )
        return

    for key, value in (
        usage_metadata.items()
    ):
        print(
            f"{key}：{value}"
        )


def print_latency(
    result: RAGAnswer,
) -> None:
    """打印各阶段耗时。"""

    print()
    print("=" * 80)
    print("耗时统计")
    print("=" * 80)

    print(
        "检索耗时："
        f"{result.retrieval_latency_ms:.2f} ms"
    )

    print(
        "上下文与 Prompt 构造耗时："
        f"{result.context_latency_ms:.2f} ms"
    )

    print(
        "DeepSeek 生成耗时："
        f"{result.generation_latency_ms:.2f} ms"
    )

    print(
        "总耗时："
        f"{result.total_latency_ms:.2f} ms"
    )


def main() -> None:
    """程序入口。"""

    args = parse_args()

    config_path = Path(
        args.config
    ).resolve()

    config = load_yaml_config(
        config_path
    )

    experiment_name = str(
        config.get(
            "experiment_name",
            "deepseek_rag",
        )
    )

    retrieval_config = require_mapping(
        config,
        "retrieval",
    )

    context_config = require_mapping(
        config,
        "context",
    )

    model_config = require_mapping(
        config,
        "model",
    )

    display_config = require_mapping(
        config,
        "display",
    )

    retrieval_type = require_string(
        retrieval_config,
        "type",
    ).lower()

    if retrieval_type != "bm25":
        raise ValueError(
            "当前 ask_rag.py 只接入 BM25，"
            f"实际配置：{retrieval_type}"
        )

    bm25_config_path = (
        resolve_project_path(
            require_string(
                retrieval_config,
                "config_path",
            )
        )
    )

    retrieval_top_k = int(
        retrieval_config.get(
            "top_k",
            5,
        )
    )

    max_documents = int(
        context_config.get(
            "max_documents",
            retrieval_top_k,
        )
    )

    max_context_characters = int(
        context_config.get(
            "max_characters",
            8000,
        )
    )

    include_metadata = bool(
        context_config.get(
            "include_metadata",
            True,
        )
    )

    deduplicate = bool(
        context_config.get(
            "deduplicate",
            True,
        )
    )

    show_sources = bool(
        display_config.get(
            "show_sources",
            True,
        )
    )

    show_context = bool(
        display_config.get(
            "show_context",
            False,
        )
    ) or bool(args.show_context)

    show_usage = bool(
        display_config.get(
            "show_usage",
            True,
        )
    )

    show_latency = bool(
        display_config.get(
            "show_latency",
            True,
        )
    )

    print("=" * 80)
    print("RAGLab BM25 + DeepSeek V4")
    print("=" * 80)
    print(
        f"实验名称：{experiment_name}"
    )
    print(
        f"配置文件：{config_path}"
    )
    print(
        f"BM25 配置：{bm25_config_path}"
    )
    print(
        f"检索 Top-K：{retrieval_top_k}"
    )
    print(
        f"上下文最大 Chunk 数："
        f"{max_documents}"
    )
    print(
        f"上下文最大字符数："
        f"{max_context_characters}"
    )

    bm25_index, _ = build_bm25_index(
        bm25_config_path
    )

    chat_model = create_deepseek_model(
        model_config
    )

    rag_chain = RAGChain(
        retriever=bm25_index.search,
        chat_model=chat_model,
        retrieval_top_k=(
            retrieval_top_k
        ),
        max_documents=max_documents,
        max_context_characters=(
            max_context_characters
        ),
        include_metadata=(
            include_metadata
        ),
        deduplicate=deduplicate,
    )

    question = resolve_question(
        args.question
    )

    print()
    print("=" * 80)
    print("用户问题")
    print("=" * 80)
    print(question)

    result = rag_chain.answer(
        question
    )

    print()
    print("=" * 80)
    print("DeepSeek 回答")
    print("=" * 80)
    print(result.answer)

    if show_sources:
        print_sources(result)

    if show_context:
        print()
        print("=" * 80)
        print("实际发送给模型的检索上下文")
        print("=" * 80)

        if result.context:
            print(result.context)
        else:
            print("没有可用上下文。")

    if show_usage:
        print_usage(
            result.usage_metadata
        )

    if show_latency:
        print_latency(result)

    print()
    print("=" * 80)
    print("本次单步 RAG 执行完成")
    print("=" * 80)


if __name__ == "__main__":
    main()