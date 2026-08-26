"""带长期记忆 Store 的 LangGraph Agent 控制台。"""

from __future__ import annotations

import argparse
from pathlib import Path

from raglab.agent.long_term_memory_agent import (
    LongTermMemoryRetrievalAgent,
    normalize_user_id,
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
from scripts.chat_persistent_agent import (
    create_thread_id,
    normalize_thread_id,
    print_memory_status,
    print_result,
    print_thread_history,
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
            "运行带跨会话长期记忆的 "
            "LangGraph Agent。"
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(
            DEFAULT_CONFIG_PATH
        ),
    )

    parser.add_argument(
        "--user-id",
        type=str,
        default="local-user",
    )

    parser.add_argument(
        "--thread-id",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--history-preview",
        type=int,
        default=800,
    )

    return parser.parse_args()


def print_help() -> None:
    """打印命令帮助。"""

    print()
    print("=" * 80)
    print("可用命令")
    print("=" * 80)

    print("/help                 查看帮助")
    print("/user                 查看当前 user_id")
    print("/thread               查看当前 thread_id")
    print("/history              查看当前线程详细消息")
    print("/summary              查看当前线程滚动摘要")
    print("/new                  创建并切换新线程")
    print("/use <thread_id>      切换指定线程")
    print("/clear                清除当前线程短期状态")

    print(
        "/remember key=value  "
        "新增或更新长期记忆"
    )

    print(
        "/memories             "
        "查看当前用户长期记忆"
    )

    print(
        "/forget <key>         "
        "删除指定长期记忆"
    )

    print("/exit                 退出")


def build_agent(
    config_path: Path,
) -> LongTermMemoryRetrievalAgent:
    """根据配置构建 Agent。"""

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
            "当前脚本只支持 BM25。"
        )

    bm25_config_path = (
        resolve_project_path(
            require_string(
                retrieval_config,
                "config_path",
            )
        )
    )

    bm25_index, build_info = (
        build_bm25_index(
            bm25_config_path
        )
    )

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

            default_top_k=int(
                tool_config.get(
                    "default_top_k",
                    5,
                )
            ),

            maximum_top_k=int(
                tool_config.get(
                    "maximum_top_k",
                    10,
                )
            ),

            max_characters_per_document=int(
                tool_config.get(
                    "max_characters_per_document",
                    1500,
                )
            ),
        )
    )

    return LongTermMemoryRetrievalAgent(
        chat_model=chat_model,
        tools=[search_tool],

        max_steps=int(
            agent_config.get(
                "max_steps",
                4,
            )
        ),

        system_prompt=(
            CONVERSATIONAL_AGENT_SYSTEM_PROMPT
        ),

        keep_recent_turns=4,
        summarize_trigger_turns=7,
    )


def print_long_term_memories(
    agent: LongTermMemoryRetrievalAgent,
    user_id: str,
) -> None:
    """打印指定用户长期记忆。"""

    memories = agent.list_memories(
        user_id=user_id
    )

    print()
    print("=" * 80)
    print("长期记忆")
    print("=" * 80)

    print(f"user_id：{user_id}")
    print(f"记忆数量：{len(memories)}")

    if not memories:
        print()
        print("当前没有长期记忆。")
        return

    for index, memory in enumerate(
        memories,
        start=1,
    ):
        value = memory["value"]

        print()
        print("-" * 80)
        print(
            f"{index}. key="
            f"{memory['key']}"
        )

        print(
            "category="
            f"{value.get('category')}"
        )

        print(
            "source="
            f"{value.get('source')}"
        )

        print(
            "content="
            f"{value.get('content')}"
        )


def main() -> None:
    """程序入口。"""

    args = parse_args()

    config_path = Path(
        args.config
    ).resolve()

    user_id = normalize_user_id(
        args.user_id
    )

    thread_id = normalize_thread_id(
        args.thread_id
        if args.thread_id
        else create_thread_id()
    )

    agent = build_agent(
        config_path
    )

    known_thread_ids = [
        thread_id
    ]

    print()
    print("=" * 80)
    print("长期记忆 LangGraph Agent")
    print("=" * 80)

    print(f"user_id：{user_id}")
    print(f"thread_id：{thread_id}")
    print("输入 /help 查看命令。")

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

        command = user_input.lower()

        if command in {
            "/exit",
            "/quit",
        }:
            print()
            print("程序已结束。")
            break

        if command == "/help":
            print_help()
            continue

        if command == "/user":
            print()
            print(
                f"当前 user_id："
                f"{user_id}"
            )
            continue

        if command == "/thread":
            print()
            print(
                f"当前 thread_id："
                f"{thread_id}"
            )

            print(
                "当前进程使用过的线程："
            )

            for current in (
                known_thread_ids
            ):
                marker = (
                    "（当前）"
                    if current == thread_id
                    else ""
                )

                print(
                    f"  - {current}{marker}"
                )

            continue

        if command == "/history":
            print_thread_history(
                agent,
                thread_id,
                maximum_characters=(
                    args.history_preview
                ),
            )
            continue

        if command in {
            "/summary",
            "/memory",
        }:
            print_memory_status(
                agent,
                thread_id,
                maximum_characters=(
                    args.history_preview
                ),
            )
            continue

        if command == "/new":
            thread_id = create_thread_id()

            known_thread_ids.append(
                thread_id
            )

            print()
            print(
                "已创建并切换到新线程："
                f"{thread_id}"
            )

            continue

        if command.startswith(
            "/use "
        ):
            target = user_input[5:].strip()

            if not target:
                print(
                    "请提供 thread_id。"
                )
                continue

            thread_id = normalize_thread_id(
                target
            )

            if (
                thread_id
                not in known_thread_ids
            ):
                known_thread_ids.append(
                    thread_id
                )

            print()
            print(
                "已切换到线程："
                f"{thread_id}"
            )

            continue

        if command == "/clear":
            agent.clear_thread(
                thread_id
            )

            print()
            print(
                "当前线程短期状态已清除。"
            )

            continue

        if command.startswith(
            "/remember "
        ):
            expression = (
                user_input[
                    len("/remember "):
                ].strip()
            )

            if "=" not in expression:
                print()
                print(
                    "格式错误，应为："
                    "/remember key=value"
                )
                continue

            key, content = (
                expression.split(
                    "=",
                    maxsplit=1,
                )
            )

            try:
                result = agent.remember(
                    user_id=user_id,
                    key=key,
                    content=content,
                )

            except Exception as error:
                print()
                print(
                    "保存失败："
                    f"{type(error).__name__}："
                    f"{error}"
                )
                continue

            print()
            print(
                "长期记忆已保存："
            )

            print(
                f"key={result['key']}"
            )

            print(
                "content="
                f"{result['value']['content']}"
            )

            continue

        if command == "/memories":
            print_long_term_memories(
                agent,
                user_id,
            )
            continue

        if command.startswith(
            "/forget "
        ):
            key = user_input[
                len("/forget "):
            ].strip()

            try:
                deleted = agent.forget(
                    user_id=user_id,
                    key=key,
                )

            except Exception as error:
                print()
                print(
                    "删除失败："
                    f"{type(error).__name__}："
                    f"{error}"
                )
                continue

            print()

            if deleted:
                print(
                    f"长期记忆已删除：{key}"
                )
            else:
                print(
                    f"没有找到长期记忆：{key}"
                )

            continue

        if user_input.startswith("/"):
            print()
            print(
                f"未知命令：{user_input}"
            )
            continue

        try:
            result = agent.run(
                user_input,
                user_id=user_id,
                thread_id=thread_id,
            )

        except Exception as error:
            print()
            print(
                "执行失败："
                f"{type(error).__name__}："
                f"{error}"
            )
            continue

        print_result(result)


if __name__ == "__main__":
    main()