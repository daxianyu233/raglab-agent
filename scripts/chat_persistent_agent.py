"""带 Checkpointer 和 thread_id 的 LangGraph Agent 交互入口。

支持：

1. 同一 thread_id 保持连续对话；
2. 不同 thread_id 隔离会话；
3. 查看当前线程保存的完整消息；
4. 创建和切换线程；
5. 清除指定线程的检查点；
6. 查看每轮模型调用、工具调用和节点轨迹。

当前使用 InMemorySaver：

    只在当前 Python 进程中保存状态；
    程序关闭后，所有线程状态都会消失。
"""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from raglab.agent.persistent_langgraph_agent import (
    PersistentLangGraphRetrievalAgent,
    PersistentLangGraphResult,
)
from raglab.agent.tools import (
    create_bm25_search_tool,
)
from raglab.settings import CONFIG_DIR
from scripts.ask_rag import (
    build_bm25_index,
    create_deepseek_model,
    load_yaml_config,
    require_mapping,
    require_string,
    resolve_project_path,
)
from scripts.chat_retrieval_agent import (
    CONVERSATIONAL_AGENT_SYSTEM_PROMPT,
)


DEFAULT_CONFIG_PATH = (
    CONFIG_DIR
    / "agent.yaml"
)


def parse_args() -> argparse.Namespace:
    """读取命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "运行带 Checkpointer 的 "
            "LangGraph Retrieval Agent。"
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Agent YAML 配置文件路径。",
    )

    parser.add_argument(
        "--thread-id",
        type=str,
        default=None,
        help=(
            "初始会话 thread_id。"
            "未提供时自动生成。"
        ),
    )

    parser.add_argument(
        "--history-preview",
        type=int,
        default=800,
        help=(
            "查看历史时，每条消息最多显示"
            "多少个字符。默认 800。"
        ),
    )

    return parser.parse_args()


def create_thread_id() -> str:
    """生成新的会话线程 ID。"""

    short_uuid = (
        uuid.uuid4()
        .hex[:8]
    )

    return f"session-{short_uuid}"


def normalize_thread_id(
    value: str | None,
) -> str:
    """检查初始 thread_id。"""

    if value is None:
        return create_thread_id()

    normalized = str(value).strip()

    if not normalized:
        return create_thread_id()

    return normalized


def content_to_text(
    content: Any,
) -> str:
    """将 LangChain 消息内容转换成文本。"""

    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: list[str] = []

        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
                continue

            if isinstance(item, dict):
                text_value = item.get(
                    "text"
                )

                if isinstance(
                    text_value,
                    str,
                ):
                    text_parts.append(
                        text_value
                    )
                else:
                    text_parts.append(
                        str(item)
                    )

                continue

            text_parts.append(
                str(item)
            )

        return "\n".join(text_parts)

    return str(content)


def truncate_text(
    text: str,
    maximum_characters: int,
) -> str:
    """截断过长的消息。"""

    normalized = str(text).strip()

    if maximum_characters <= 0:
        return normalized

    if (
        len(normalized)
        <= maximum_characters
    ):
        return normalized

    return (
        normalized[
            :maximum_characters
        ].rstrip()
        + "\n……消息内容已截断……"
    )


def message_role(
    message: BaseMessage,
) -> str:
    """返回消息角色名称。"""

    if isinstance(
        message,
        HumanMessage,
    ):
        return "用户"

    if isinstance(
        message,
        AIMessage,
    ):
        if message.tool_calls:
            return "助手（工具决策）"

        return "助手"

    if isinstance(
        message,
        ToolMessage,
    ):
        return "工具"

    if isinstance(
        message,
        SystemMessage,
    ):
        return "系统"

    return str(
        getattr(
            message,
            "type",
            type(message).__name__,
        )
    )


def print_help() -> None:
    """显示交互命令。"""

    print()
    print("=" * 80)
    print("可用命令")
    print("=" * 80)

    print(
        "/help                "
        "查看帮助"
    )

    print(
        "/thread              "
        "查看当前 thread_id"
    )

    print(
        "/history             "
        "查看最近保留的详细消息"
    )

    print(
        "/summary             "
        "查看滚动摘要和记忆状态"
    )

    print(
        "/memory              "
        "与 /summary 相同"
    )

    print(
        "/new                 "
        "创建并切换到新线程"
    )

    print(
        "/use <thread_id>     "
        "切换到指定线程"
    )

    print(
        "/clear               "
        "清除当前线程状态"
    )

    print(
        "/exit                "
        "退出程序"
    )

def print_thread_history(
    agent: PersistentLangGraphRetrievalAgent,
    thread_id: str,
    *,
    maximum_characters: int,
) -> None:
    """打印指定线程保存的完整消息状态。"""

    messages = (
        agent.get_thread_messages(
            thread_id
        )
    )

    print()
    print("=" * 80)
    print("当前线程消息历史")
    print("=" * 80)

    print(
        f"thread_id：{thread_id}"
    )

    print(
        f"消息数量：{len(messages)}"
    )

    if not messages:
        print()
        print(
            "当前线程没有历史消息。"
        )
        return

    for index, message in enumerate(
        messages,
        start=1,
    ):
        role = message_role(
            message
        )

        content = content_to_text(
            message.content
        )

        content = truncate_text(
            content,
            maximum_characters,
        )

        print()
        print("-" * 80)

        print(
            f"消息 {index}｜{role}"
        )

        if isinstance(
            message,
            AIMessage,
        ):
            tool_calls = (
                message.tool_calls
            )

            if tool_calls:
                print(
                    f"工具调用数量："
                    f"{len(tool_calls)}"
                )

                for tool_index, tool_call in enumerate(
                    tool_calls,
                    start=1,
                ):
                    print(
                        f"  {tool_index}. "
                        f"{tool_call.get('name')}"
                        f"("
                        f"{tool_call.get('args')}"
                        f")"
                    )

        if isinstance(
            message,
            ToolMessage,
        ):
            print(
                f"工具名称："
                f"{message.name}"
            )

            print(
                f"Tool Call ID："
                f"{message.tool_call_id}"
            )

        if content:
            print()
            print(content)
        else:
            print()
            print(
                "该消息正文为空。"
            )

def print_memory_status(
    agent: PersistentLangGraphRetrievalAgent,
    thread_id: str,
    *,
    maximum_characters: int,
) -> None:
    """显示当前线程的摘要和详细记忆状态。"""

    state = agent.get_thread_state(
        thread_id
    )

    messages = state.get(
        "messages",
        [],
    )

    if not isinstance(
        messages,
        list,
    ):
        messages = []

    summary = str(
        state.get(
            "summary",
            "",
        )
    ).strip()

    recent_turn_count = sum(
        isinstance(
            message,
            HumanMessage,
        )
        for message in messages
    )

    total_summarized_turns = int(
        state.get(
            "total_summarized_turns",
            0,
        )
        or 0
    )

    print()
    print("=" * 80)
    print("当前线程记忆状态")
    print("=" * 80)

    print(
        f"thread_id：{thread_id}"
    )

    print(
        "累计已压缩轮数："
        f"{total_summarized_turns}"
    )

    print(
        "当前保留详细轮数："
        f"{recent_turn_count}"
    )

    print(
        "当前详细消息数量："
        f"{len(messages)}"
    )

    print(
        "是否已经生成摘要："
        f"{bool(summary)}"
    )

    print()
    print("-" * 80)
    print("滚动摘要")
    print("-" * 80)

    if not summary:
        print(
            "当前还没有生成滚动摘要。"
        )
        return

    print(
        truncate_text(
            summary,
            maximum_characters,
        )
    )

def build_execution_path(
    result: PersistentLangGraphResult,
) -> list[str]:
    """根据当前轮轨迹还原节点执行路径。"""

    path = ["START"]

    tool_trace_index = 0

    for model_trace in (
        result.model_trace
    ):
        node_name = str(
            model_trace.get(
                "node",
                "agent",
            )
        )

        path.append(node_name)

        tool_call_count = int(
            model_trace.get(
                "tool_call_count",
                0,
            )
            or 0
        )

        if tool_call_count > 0:
            path.append("tools")

            tool_trace_index += (
                tool_call_count
            )

    path.append("END")

    return path


def print_result(
    result: PersistentLangGraphResult,
) -> None:
    """打印当前轮回答、执行统计和摘要状态。"""

    print()
    print("=" * 80)
    print("Agent 回答")
    print("=" * 80)

    print(result.answer)

    print()
    print("-" * 80)

    path = build_execution_path(
        result
    )

    print(
        "节点路径："
        + " → ".join(path)
    )

    print(
        f"当前 thread_id："
        f"{result.thread_id}"
    )

    print(
        f"本轮模型调用次数："
        f"{result.turn_llm_call_count}"
    )

    print(
        f"本轮工具调用次数："
        f"{result.turn_tool_call_count}"
    )

    print(
        f"本轮摘要模型调用次数："
        f"{result.turn_summary_call_count}"
    )

    print(
        f"本轮是否更新摘要："
        f"{result.summary_updated}"
    )

    print(
        "本轮压缩历史轮数："
        f"{result.summarized_turns_this_run}"
    )

    print(
        "累计已压缩轮数："
        f"{result.total_summarized_turns}"
    )

    print(
        "当前保留详细轮数："
        f"{result.recent_turn_count}"
    )

    print(
        f"当前详细消息数量："
        f"{result.total_message_count}"
    )

    print(
        f"当前是否存在摘要："
        f"{bool(result.summary)}"
    )

    print(
        f"本轮总耗时："
        f"{result.total_latency_ms:.2f} ms"
    )

    if result.tool_trace:
        print()
        print("本轮工具调用：")

        for index, trace in enumerate(
            result.tool_trace,
            start=1,
        ):
            tool_name = trace.get(
                "name",
                trace.get(
                    "tool_name",
                    "N/A",
                ),
            )

            arguments = trace.get(
                "args",
                trace.get(
                    "arguments",
                    {},
                ),
            )

            status = trace.get(
                "status",
                "unknown",
            )

            print(
                f"  {index}. "
                f"{tool_name}"
                f"({arguments})"
                f" | status={status}"
            )
    else:
        print(
            "本轮未调用工具。"
        )
        
    if result.summary_updated:
        print()
        print(
            "本轮已执行滚动摘要："
            f"压缩了 "
            f"{result.summarized_turns_this_run} "
            "轮较早对话，"
            f"保留最近 "
            f"{result.recent_turn_count} 轮。"
        )

def build_agent(
    config_path: Path,
) -> PersistentLangGraphRetrievalAgent:
    """根据项目配置创建持久化 Agent。"""

    config = load_yaml_config(
        config_path
    )

    retrieval_config = require_mapping(
        config,
        "retrieval",
    )

    model_config = require_mapping(
        config,
        "model",
    )

    tool_config = require_mapping(
        config,
        "tool",
    )

    agent_config = require_mapping(
        config,
        "agent",
    )

    retrieval_type = require_string(
        retrieval_config,
        "type",
    ).lower()

    if retrieval_type != "bm25":
        raise ValueError(
            "当前脚本只支持 BM25，"
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

    default_top_k = int(
        tool_config.get(
            "default_top_k",
            5,
        )
    )

    maximum_top_k = int(
        tool_config.get(
            "maximum_top_k",
            10,
        )
    )

    max_characters_per_document = int(
        tool_config.get(
            "max_characters_per_document",
            1500,
        )
    )

    max_steps = int(
        agent_config.get(
            "max_steps",
            4,
        )
    )

    print("=" * 80)
    print("初始化 LangGraph 会话 Agent")
    print("=" * 80)

    print(
        f"Agent 配置：{config_path}"
    )

    print(
        f"BM25 配置："
        f"{bm25_config_path}"
    )

    print(
        f"最大 Agent 步数："
        f"{max_steps}"
    )

    bm25_index, build_info = (
        build_bm25_index(
            bm25_config_path
        )
    )

    print()
    print(
        "BM25 索引构建完成："
        f"{build_info['chunk_count']} "
        "个 Chunk"
    )

    chat_model = create_deepseek_model(
        model_config
    )

    search_tool = (
        create_bm25_search_tool(
            bm25_index=bm25_index,
            default_top_k=(
                default_top_k
            ),
            maximum_top_k=(
                maximum_top_k
            ),
            max_characters_per_document=(
                max_characters_per_document
            ),
        )
    )

    return (
        PersistentLangGraphRetrievalAgent(
            chat_model=chat_model,
            tools=[search_tool],
            max_steps=max_steps,

            # 连续对话使用软约束提示：
            #
            # 文本转换不建议调用；
            # 新事实问题建议调用；
            # 历史充分时允许不调用。
            system_prompt=(
                CONVERSATIONAL_AGENT_SYSTEM_PROMPT
            ),
        )
    )


def main() -> None:
    """交互式程序入口。"""

    args = parse_args()

    if args.history_preview <= 0:
        raise ValueError(
            "--history-preview 必须大于 0。"
        )

    config_path = Path(
        args.config
    ).resolve()

    agent = build_agent(
        config_path
    )

    current_thread_id = (
        normalize_thread_id(
            args.thread_id
        )
    )

    # 记录当前进程中使用过的线程，
    # 方便查看和切换。
    known_thread_ids: list[str] = [
        current_thread_id
    ]

    print()
    print("=" * 80)
    print("LangGraph 持久化会话 Agent")
    print("=" * 80)

    print(
        f"当前 thread_id："
        f"{current_thread_id}"
    )

    print(
        "输入 /help 查看命令。"
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
            print("程序已结束。")
            break

        if not user_input:
            continue

        normalized_command = (
            user_input.lower()
        )

        if normalized_command in {
            "/exit",
            "/quit",
        }:
            print()
            print("程序已结束。")
            break

        if normalized_command == "/help":
            print_help()
            continue

        if normalized_command == "/thread":
            print()
            print(
                f"当前 thread_id："
                f"{current_thread_id}"
            )

            print(
                "当前进程已使用的线程："
            )

            for thread_id in (
                known_thread_ids
            ):
                marker = (
                    "（当前）"
                    if (
                        thread_id
                        == current_thread_id
                    )
                    else ""
                )

                print(
                    f"  - {thread_id}"
                    f"{marker}"
                )

            continue

        if normalized_command == "/history":
            print_thread_history(
                agent,
                current_thread_id,
                maximum_characters=(
                    args.history_preview
                ),
            )

            continue

        if normalized_command in {
            "/summary",
            "/memory",
        }:
            print_memory_status(
                agent,
                current_thread_id,
                maximum_characters=(
                    args.history_preview
                ),
            )

            continue

        if normalized_command == "/new":
            current_thread_id = (
                create_thread_id()
            )

            if (
                current_thread_id
                not in known_thread_ids
            ):
                known_thread_ids.append(
                    current_thread_id
                )

            print()
            print(
                "已创建并切换到新线程："
                f"{current_thread_id}"
            )

            continue

        if normalized_command.startswith(
            "/use "
        ):
            target_thread_id = (
                user_input[5:].strip()
            )

            if not target_thread_id:
                print()
                print(
                    "请提供需要切换的 "
                    "thread_id。"
                )
                continue

            current_thread_id = (
                target_thread_id
            )

            if (
                current_thread_id
                not in known_thread_ids
            ):
                known_thread_ids.append(
                    current_thread_id
                )

            print()
            print(
                "已切换到线程："
                f"{current_thread_id}"
            )

            continue

        if normalized_command == "/clear":
            try:
                agent.clear_thread(
                    current_thread_id
                )

            except Exception as error:
                print()
                print(
                    "清除线程失败："
                    f"{type(error).__name__}："
                    f"{error}"
                )

                continue

            print()
            print(
                "当前线程状态已清除："
                f"{current_thread_id}"
            )

            continue

        if user_input.startswith("/"):
            print()
            print(
                f"无法识别命令："
                f"{user_input}"
            )

            print(
                "输入 /help 查看命令。"
            )

            continue

        try:
            result = agent.run(
                user_input,
                thread_id=(
                    current_thread_id
                ),
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

        print_result(result)


if __name__ == "__main__":
    main()