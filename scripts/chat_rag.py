"""BM25 + DeepSeek 多轮 RAG 交互入口。

运行流程：

第一轮：
用户问题
→ BM25 检索
→ 构造上下文
→ DeepSeek 回答
→ 保存历史

第二轮及以后：
历史 + 当前问题
→ DeepSeek 改写为独立检索问题
→ BM25 检索
→ 构造上下文
→ DeepSeek 回答
→ 保存历史

当前依然是固定流程的 Conversational RAG，
不是能够自主选择工具和循环执行的 Agent。

支持命令：

/help
/history
/clear
/exit
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from raglab.generation.conversational_rag_chain import (
    ConversationalRAGAnswer,
    ConversationalRAGChain,
)
from raglab.settings import (
    CONFIG_DIR,
)
from scripts.ask_rag import (
    build_bm25_index,
    create_deepseek_model,
    load_yaml_config,
    require_mapping,
    require_string,
    resolve_project_path,
)


DEFAULT_CONFIG_PATH = (
    CONFIG_DIR
    / "generation.yaml"
)


def parse_args() -> argparse.Namespace:
    """读取命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "使用 BM25 与 DeepSeek 执行"
            "多轮 RAG 对话。"
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="generation.yaml 配置文件路径。",
    )

    return parser.parse_args()


def print_help() -> None:
    """打印可用命令。"""

    print()
    print("=" * 80)
    print("可用命令")
    print("=" * 80)
    print("/help     查看命令帮助")
    print("/history  查看当前会话历史")
    print("/clear    清空当前会话历史")
    print("/exit     退出程序")
    print()
    print(
        "除以上命令外，输入任意文本即可提问。"
    )


def print_history(
    conversation_chain: (
        ConversationalRAGChain
    ),
) -> None:
    """打印当前会话历史。"""

    history = (
        conversation_chain.get_history()
    )

    print()
    print("=" * 80)
    print("当前会话历史")
    print("=" * 80)

    if not history:
        print("当前没有历史对话。")
        return

    for turn in history:
        print()
        print(f"第 {turn.turn_index} 轮")

        print(
            f"用户原始问题："
            f"{turn.question}"
        )

        print(
            f"实际检索问题："
            f"{turn.retrieval_question}"
        )

        print(
            f"助手回答："
            f"{turn.answer}"
        )

        print("-" * 80)


def print_sources(
    result: ConversationalRAGAnswer,
) -> None:
    """打印进入本轮上下文的资料来源。"""

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


def print_usage_section(
    title: str,
    usage: dict[str, Any],
) -> None:
    """打印一组 Token 使用信息。"""

    print()
    print(title)
    print("-" * 80)

    if not usage:
        print("没有返回可识别的 Token 数据。")
        return

    for key, value in usage.items():
        print(f"{key}：{value}")


def print_usage(
    result: ConversationalRAGAnswer,
) -> None:
    """打印问题改写和最终回答的 Token 使用。"""

    print()
    print("=" * 80)
    print("Token 使用")
    print("=" * 80)

    if result.rewrite_response is None:
        print("本轮没有调用问题改写模型。")
    else:
        print_usage_section(
            "问题改写调用",
            result.rewrite_usage_metadata,
        )

    print_usage_section(
        "最终回答调用",
        result.generation_usage_metadata,
    )


def print_latency(
    result: ConversationalRAGAnswer,
) -> None:
    """打印本轮各阶段耗时。"""

    print()
    print("=" * 80)
    print("耗时统计")
    print("=" * 80)

    print(
        "问题改写耗时："
        f"{result.rewrite_latency_ms:.2f} ms"
    )

    print(
        "检索耗时："
        f"{result.retrieval_latency_ms:.2f} ms"
    )

    print(
        "上下文构造耗时："
        f"{result.context_latency_ms:.2f} ms"
    )

    print(
        "DeepSeek 回答耗时："
        f"{result.generation_latency_ms:.2f} ms"
    )

    print(
        "本轮总耗时："
        f"{result.total_latency_ms:.2f} ms"
    )


def print_result(
    result: ConversationalRAGAnswer,
    *,
    show_sources: bool,
    show_context: bool,
    show_usage: bool,
    show_latency: bool,
    show_retrieval_question: bool,
) -> None:
    """打印一次完整问答结果。"""

    print()
    print("=" * 80)
    print(
        f"DeepSeek 回答｜第 "
        f"{result.turn_index} 轮"
    )
    print("=" * 80)

    if show_retrieval_question:
        print(
            "用户原始问题："
            f"{result.question}"
        )

        print(
            "实际检索问题："
            f"{result.retrieval_question}"
        )

        print(
            "是否发生改写："
            f"{'是' if result.query_rewritten else '否'}"
        )

        print("-" * 80)

    print(result.answer)

    if show_sources:
        print_sources(result)

    if show_context:
        print()
        print("=" * 80)
        print("本轮实际发送给模型的检索上下文")
        print("=" * 80)

        if result.context:
            print(result.context)
        else:
            print("本轮没有可用上下文。")

    if show_usage:
        print_usage(result)

    if show_latency:
        print_latency(result)


def handle_command(
    command: str,
    conversation_chain: (
        ConversationalRAGChain
    ),
) -> bool:
    """处理控制台命令。

    Returns
    -------
    bool
        True 表示命令已处理并继续运行。
        False 表示需要退出程序。
    """

    normalized = (
        command.strip().lower()
    )

    if normalized == "/help":
        print_help()
        return True

    if normalized == "/history":
        print_history(
            conversation_chain
        )
        return True

    if normalized == "/clear":
        conversation_chain.clear_history()

        print()
        print("当前会话历史已清空。")
        return True

    if normalized in {
        "/exit",
        "/quit",
    }:
        print()
        print("多轮 RAG 会话已结束。")
        return False

    print()
    print(
        f"无法识别命令：{command}"
    )
    print(
        "输入 /help 查看可用命令。"
    )

    return True


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
            "conversational_rag",
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

    conversation_config = (
        require_mapping(
            config,
            "conversation",
        )
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
            "当前 chat_rag.py 只接入 BM25，"
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

    max_history_turns = int(
        conversation_config.get(
            "max_history_turns",
            6,
        )
    )

    rewrite_mode = str(
        conversation_config.get(
            "rewrite_mode",
            "always",
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
    )

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

    show_retrieval_question = bool(
        display_config.get(
            "show_retrieval_question",
            True,
        )
    )

    print("=" * 80)
    print("RAGLab BM25 + DeepSeek 多轮 RAG")
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
        f"最大历史轮数：{max_history_turns}"
    )

    print(
        f"问题改写模式：{rewrite_mode}"
    )

    bm25_index, _ = build_bm25_index(
        bm25_config_path
    )

    chat_model = create_deepseek_model(
        model_config
    )

    conversation_chain = (
        ConversationalRAGChain(
            retriever=bm25_index.search,
            chat_model=chat_model,
            retrieval_top_k=(
                retrieval_top_k
            ),
            max_documents=max_documents,
            max_context_characters=(
                max_context_characters
            ),
            max_history_turns=(
                max_history_turns
            ),
            rewrite_mode=rewrite_mode,
            include_metadata=(
                include_metadata
            ),
            deduplicate=deduplicate,
        )
    )

    print()
    print("多轮 RAG 已启动。")
    print(
        "输入 /help 查看命令，"
        "输入 /exit 退出。"
    )

    while True:
        try:
            user_input = input(
                "\n你："
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError,
        ):
            print()
            print("多轮 RAG 会话已结束。")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            should_continue = handle_command(
                user_input,
                conversation_chain,
            )

            if not should_continue:
                break

            continue

        try:
            result = (
                conversation_chain.answer(
                    user_input
                )
            )

        except Exception as error:
            print()
            print("=" * 80)
            print("本轮执行失败")
            print("=" * 80)
            print(
                f"{type(error).__name__}："
                f"{error}"
            )
            continue

        print_result(
            result,
            show_sources=show_sources,
            show_context=show_context,
            show_usage=show_usage,
            show_latency=show_latency,
            show_retrieval_question=(
                show_retrieval_question
            ),
        )


if __name__ == "__main__":
    main()