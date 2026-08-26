"""
自动长期记忆 LangGraph 多工具 Agent 控制台。

该文件现在只负责 CLI 交互：

1. 读取命令行参数；
2. 创建 user_id / thread_id；
3. 调用统一 Agent Factory；
4. 接收控制台用户输入；
5. 处理 CLI 专属命令；
6. 调用 Agent；
7. 打印执行结果。

Agent 的模型、Retriever、Tool、SkillRuntime 等组装逻辑，
已经迁移到：

    raglab.application.agent_factory
"""

from __future__ import annotations

import argparse

from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from raglab.agent.automatic_long_term_memory_agent import (
    AutomaticLongTermMemoryAgent,
)
from raglab.agent.long_term_memory_agent import (
    normalize_user_id,
)

# ------------------------------------------------------------
# 统一 Agent Factory
# ------------------------------------------------------------

from raglab.application.agent_factory import (
    build_agent,
)

from raglab.settings import (
    CONFIG_DIR,
)

# ------------------------------------------------------------
# CLI 专属显示 / 会话辅助函数
# ------------------------------------------------------------

from scripts.chat_long_term_memory_agent import (
    print_long_term_memories,
)

from scripts.chat_persistent_agent import (
    create_thread_id,
    normalize_thread_id,
    print_memory_status,
    print_result,
    print_thread_history,
)


# ============================================================
# 默认配置
# ============================================================


DEFAULT_CONFIG_PATH = (
    CONFIG_DIR
    / "agent.yaml"
)


# ============================================================
# CLI 参数
# ============================================================


def parse_args() -> argparse.Namespace:
    """
    读取命令行参数。

    这里属于 CLI 层。

    例如：

        python -m scripts.chat_automatic_memory_agent \
            --user-id huangwu
    """

    parser = argparse.ArgumentParser(
        description=(
            "运行带 GitHub Skill 的"
            "自动长期记忆 Agent。"
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


# ============================================================
# CLI 输出函数
# ============================================================


def print_flush_report(
    report: dict,
) -> None:
    """
    打印长期记忆整理报告。

    这是控制台显示逻辑，
    因此仍然属于 CLI。
    """

    print()
    print("=" * 80)
    print("长期记忆整理结果")
    print("=" * 80)

    for key, value in (
        report.items()
    ):
        print(
            f"{key}：{value}"
        )


def print_skill_status(
    agent: AutomaticLongTermMemoryAgent,
) -> None:
    """
    打印当前 Skill Runtime 状态。

    这里只负责把 Agent Runtime 状态
    以控制台文本形式展示出来。
    """

    runtime = getattr(
        agent,
        "skill_runtime",
        None,
    )

    print()
    print("=" * 80)
    print("Skill Runtime 状态")
    print("=" * 80)

    if runtime is None:
        print(
            "当前 Agent 没有配置 SkillRuntime。"
        )
        return

    status = runtime.status()

    available_skills = status.get(
        "available_skills",
        [],
    )

    if available_skills:
        print(
            "可用 Skills："
        )

        for skill in available_skills:
            loaded_text = (
                "已加载"
                if skill.get(
                    "loaded",
                    False,
                )
                else "未加载"
            )

            print(
                "  - "
                f"{skill.get('id', 'unknown')} "
                f"[{loaded_text}]"
            )

            print(
                "    "
                f"{skill.get('description', '')}"
            )

    else:
        print(
            "可用 Skills：无"
        )

    loaded_skill_ids = status.get(
        "loaded_skill_ids",
        [],
    )

    print(
        "已加载 Skills："
        + (
            ", ".join(
                loaded_skill_ids
            )
            if loaded_skill_ids
            else "无"
        )
    )

    active_tool_names = (
        agent.get_active_tool_names()
    )

    print(
        "当前 Active Tools："
        + (
            ", ".join(
                active_tool_names
            )
            if active_tool_names
            else "无"
        )
    )

def format_checkpoint_time(
    value: Any,
) -> str:
    """把 LangGraph UTC Checkpoint 时间转为北京时间。"""

    text = str(
        value
        if value is not None
        else ""
    ).strip()

    if not text:
        return ""

    try:

        normalized = (
            text.replace(
                "Z",
                "+00:00",
            )
        )

        parsed = (
            datetime.fromisoformat(
                normalized
            )
        )

        if parsed.tzinfo is None:

            parsed = parsed.replace(
                tzinfo=ZoneInfo(
                    "UTC"
                )
            )

        local_time = (
            parsed.astimezone(
                ZoneInfo(
                    "Asia/Shanghai"
                )
            )
        )

        return local_time.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    except (
        TypeError,
        ValueError,
    ):

        return text

def _preview_checkpoint_text(
    value: Any,
    maximum_characters: int = 500,
) -> str:
    """把 Checkpoint 中较长内容压缩为单行预览。"""

    text = str(
        value
        if value is not None
        else ""
    )

    text = " ".join(
        text.split()
    )

    if (
        len(text)
        <= maximum_characters
    ):
        return text

    return (
        text[
            :maximum_characters
        ]
        + "..."
    )


def print_checkpoint_history(
    agent: AutomaticLongTermMemoryAgent,
    thread_id: str,
    *,
    limit: int = 20,
) -> None:
    """打印当前 thread 的 Checkpoint 历史。"""

    checkpoints = (
        agent.list_thread_checkpoints(
            thread_id,
            limit=limit,
        )
    )

    print()
    print("=" * 80)
    print(
        "Checkpoint History"
    )
    print("=" * 80)

    print(
        f"thread_id：{thread_id}"
    )

    if not checkpoints:
        print(
            "当前 thread 尚无 Checkpoint。"
        )
        return

    print(
        "说明：最新 Checkpoint 排在最前。"
    )

    print()

    for index, checkpoint in enumerate(
        checkpoints,
        start=1,
    ):

        next_nodes = (
            checkpoint.get(
                "next_nodes",
                [],
            )
            or []
        )

        write_nodes = (
            checkpoint.get(
                "write_nodes",
                [],
            )
            or []
        )

        checkpoint_id = str(
            checkpoint.get(
                "checkpoint_id",
                "",
            )
        )

        print(
            f"[{index}] "
            f"step={checkpoint.get('step')} "
            f"source={checkpoint.get('source')}"
        )

        print(
            "    checkpoint_id："
            f"{checkpoint_id}"
        )

        print(
            "    时间："
            + format_checkpoint_time(
                checkpoint.get(
                    "created_at",
                    "",
                )
            )
        )

        print(
            "    下一步："
            + (
                ", ".join(
                    str(
                        current
                    )
                    for current in next_nodes
                )
                if next_nodes
                else "END / 无待执行节点"
            )
        )

        completed_nodes = list(
            checkpoint.get(
                "completed_nodes",
                [],
            )
            or []
        )

        print(
            "    事件："
            f"{checkpoint.get('event', '')}"
        )

        print(
            "    刚完成节点："
            + (
                ", ".join(
                    completed_nodes
                )
                if completed_nodes
                else "无 / 无法判断"
            )
        )

        completed_nodes_source = str(
            checkpoint.get(
                "completed_nodes_source",
                "unknown",
            )
        )

        print(
            "    节点来源："
            + completed_nodes_source
        )

        print(
            "    messages："
            f"{checkpoint.get('message_count', 0)}"
        )

        print(
            "    本轮 LLM / Tool / Summary："
            f"{checkpoint.get('turn_llm_calls', 0)} / "
            f"{checkpoint.get('turn_tool_calls', 0)} / "
            f"{checkpoint.get('turn_summary_calls', 0)}"
        )

        print(
            "    Trace："
            f"model={checkpoint.get('model_trace_count', 0)}, "
            f"tool={checkpoint.get('tool_trace_count', 0)}"
        )

        if checkpoint.get(
            "interrupt_count",
            0,
        ):

            print(
                "    interrupts："
                f"{checkpoint['interrupt_count']}"
            )

        task_errors = (
            checkpoint.get(
                "task_errors",
                [],
            )
            or []
        )

        if task_errors:

            print(
                "    errors："
                + " | ".join(
                    str(
                        current
                    )
                    for current in task_errors
                )
            )

        print()


def print_checkpoint_detail(
    agent: AutomaticLongTermMemoryAgent,
    thread_id: str,
    checkpoint_id: str,
    *,
    maximum_characters: int = 800,
) -> None:
    """打印指定 Checkpoint 的详细状态。"""

    checkpoint = (
        agent.get_thread_checkpoint(
            thread_id,
            checkpoint_id,
        )
    )

    print()
    print("=" * 80)
    print(
        "Checkpoint Detail"
    )
    print("=" * 80)

    if not checkpoint:

        print(
            "未找到该 Checkpoint："
            f"{checkpoint_id}"
        )

        return

    print(
        f"thread_id：{thread_id}"
    )

    print(
        "checkpoint_id："
        f"{checkpoint.get('checkpoint_id', '')}"
    )

    print(
        "parent_checkpoint_id："
        + (
            checkpoint.get(
                "parent_checkpoint_id",
                "",
            )
            or "无"
        )
    )

    print(
        "step："
        f"{checkpoint.get('step')}"
    )

    print(
        "source："
        f"{checkpoint.get('source', '')}"
    )

    print(
        "时间："
        + format_checkpoint_time(
            checkpoint.get(
                "created_at",
                "",
            )
        )
    )

    print(
        "事件："
        f"{checkpoint.get('event', '')}"
    )

    print(
        "next："
        + (
            ", ".join(
                checkpoint.get(
                    "next_nodes",
                    [],
                )
            )
            if checkpoint.get(
                "next_nodes"
            )
            else "END / 无待执行节点"
        )
    )

    print(
        "writes："
        + (
            ", ".join(
                checkpoint.get(
                    "write_nodes",
                    [],
                )
            )
            if checkpoint.get(
                "write_nodes"
            )
            else "无"
        )
    )

    print(
        "messages："
        f"{checkpoint.get('message_count', 0)}"
    )

    print(
        "state keys："
        ", ".join(
            checkpoint.get(
                "state_keys",
                [],
            )
        )
    )

    values = (
        checkpoint.get(
            "values",
            {},
        )
        or {}
    )

    summary = str(
        values.get(
            "summary",
            "",
        )
        or ""
    ).strip()

    print()
    print(
        "滚动摘要："
    )

    print(
        _preview_checkpoint_text(
            summary,
            maximum_characters,
        )
        if summary
        else "无"
    )

    # --------------------------------------------------------
    # 最近消息
    # --------------------------------------------------------

    messages = list(
        values.get(
            "messages",
            [],
        )
        or []
    )

    print()
    print(
        "最近消息："
    )

    if not messages:

        print(
            "无"
        )

    else:

        for message in messages[-5:]:

            role = (
                type(
                    message
                ).__name__
            )

            content = getattr(
                message,
                "content",
                "",
            )

            print(
                f"- {role}: "
                + _preview_checkpoint_text(
                    content,
                    maximum_characters=300,
                )
            )

    # --------------------------------------------------------
    # Trace
    # --------------------------------------------------------

    model_trace = list(
        values.get(
            "model_trace",
            [],
        )
        or []
    )

    tool_trace = list(
        values.get(
            "tool_trace",
            [],
        )
        or []
    )

    print()
    print(
        "Model Trace："
    )

    if model_trace:
        for item in model_trace:

            print(
                "- node="
                f"{item.get('node', '')}, "
                "llm_call="
                f"{item.get('llm_call_index', '')}, "
                "tool_calls="
                f"{item.get('tool_call_count', 0)}"
            )

    else:
        print(
            "无"
        )

    print()
    print(
        "Tool Trace："
    )

    if tool_trace:

        for item in tool_trace:

            print(
                "- "
                f"{item.get('name', item.get('tool_name', 'tool'))} "
                f"status={item.get('status', '')}"
            )

    else:

        print(
            "无"
        )

    task_errors = list(
        checkpoint.get(
            "task_errors",
            [],
        )
        or []
    )

    if task_errors:

        print()
        print(
            "Task Errors："
        )

        for error in task_errors:
            print(
                f"- {error}"
            )

    if checkpoint.get(
        "interrupt_count",
        0,
    ):

        print()
        print(
            "Interrupts："
            f"{checkpoint['interrupt_count']}"
        )


def print_checkpoint_replay_result(
    result: Any,
) -> None:
    """打印 Checkpoint Replay 结果。"""

    print()
    print("=" * 80)
    print(
        "Checkpoint Replay Result"
    )
    print("=" * 80)

    print(
        "thread_id："
        f"{result.thread_id}"
    )

    print(
        "Replay 起点 Step："
        f"{result.replayed_from_step}"
    )

    print(
        "Replay 起点 Checkpoint："
        f"{result.replayed_from_checkpoint_id}"
    )

    print(
        "重新执行起始节点："
        + (
            ", ".join(
                result.replay_start_next_nodes
            )
            if result.replay_start_next_nodes
            else "无"
        )
    )

    print(
        "Replay 后最终 Step："
        f"{result.final_step}"
    )

    print(
        "Replay 后最终 Checkpoint："
        f"{result.final_checkpoint_id}"
    )

    print(
        "最终 next："
        + (
            ", ".join(
                result.final_next_nodes
            )
            if result.final_next_nodes
            else "END"
        )
    )

    print(
        "最终 messages："
        f"{result.message_count}"
    )

    print(
        "Replay 耗时："
        f"{result.latency_ms:.2f} ms"
    )

    print()

    print(
        "Replay 最终回答："
    )

    print(
        result.answer
        if result.answer
        else "无新的 AI 回答。"
    )

def print_help() -> None:
    """打印命令。"""

    print()
    print("/skills               查看 Skill Catalog 和加载状态")
    print("/tools                查看当前 Active Tools")
    print("/history              查看短期详细历史")
    print("/summary              查看滚动摘要")
    print("/checkpoints           查看当前 thread 的 Checkpoint 历史")
    print("/checkpoint <id>      查看指定 Checkpoint 的详细状态")
    print("/memories             查看长期记忆")
    print("/flush-memory         手动执行保底整理")
    print("/memory-report        查看最近整理报告")
    print("/remember key=value  显式写入长期记忆")
    print("/forget key          删除长期记忆")
    print("/new                 整理后创建新会话")
    print("/exit                整理后退出")
    print("/replay <id>         从指定 Checkpoint 重新执行后续节点"
)

# ============================================================
# CLI Main
# ============================================================


def main() -> None:
    """
    CLI 程序入口。

    现在 main() 不再负责：

        BM25 怎么创建；
        DeepSeek 怎么创建；
        Tool 怎么注册；
        SkillRuntime 怎么创建；
        Agent 怎么组装。

    这些工作全部交给：

        build_agent()

    CLI 只负责人与 Agent 的命令行交互。
    """

    # --------------------------------------------------------
    # 1. 解析命令行参数
    # --------------------------------------------------------

    args = parse_args()

    # --------------------------------------------------------
    # 2. user_id
    # --------------------------------------------------------

    user_id = normalize_user_id(
        args.user_id
    )

    # --------------------------------------------------------
    # 3. thread_id
    # --------------------------------------------------------

    thread_id = normalize_thread_id(
        args.thread_id
        if args.thread_id
        else create_thread_id()
    )

    # --------------------------------------------------------
    # 4. 调用统一 Agent Factory
    # --------------------------------------------------------
    #
    # CLI 已经不再知道 Agent 内部
    # 到底如何进行组装。
    #
    # 它只知道：
    #
    #     给 build_agent 一个配置文件
    #
    #     得到一个可以运行的 Agent。
    #
    # --------------------------------------------------------

    agent = build_agent(
        Path(
            args.config
        ).resolve()
    )

    # --------------------------------------------------------
    # 5. CLI 启动信息
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "自动长期记忆多工具 Agent"
    )
    print("=" * 80)

    print(
        f"user_id：{user_id}"
    )

    print(
        f"thread_id：{thread_id}"
    )

    print(
        "输入 /help 查看命令。"
    )

    print_skill_status(
        agent
    )

    # --------------------------------------------------------
    # 6. CLI 主循环
    # --------------------------------------------------------

    while True:

        # ----------------------------------------------------
        # 读取用户输入
        # ----------------------------------------------------

        try:
            user_input = input(
                "\n你："
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError,
        ):
            report = (
                agent.flush_long_term_memory(
                    thread_id=thread_id,
                    user_id=user_id,
                    trigger=(
                        "console_interrupted"
                    ),
                )
            )

            print_flush_report(
                report
            )

            print(
                "程序已结束。"
            )

            break

        # 空输入直接忽略。
        if not user_input:
            continue

        command = (
            user_input.lower()
        )

        # ----------------------------------------------------
        # /help
        # ----------------------------------------------------

        if command == "/help":
            print_help()
            continue

        # ----------------------------------------------------
        # /skills
        # ----------------------------------------------------

        if command == "/skills":
            print_skill_status(
                agent
            )
            continue

        # ----------------------------------------------------
        # /tools
        # ----------------------------------------------------

        if command == "/tools":
            print()

            print(
                "当前 Active Tools："
                + ", ".join(
                    agent.get_active_tool_names()
                )
            )

            continue

        # ----------------------------------------------------
        # /exit
        # ----------------------------------------------------

        if command in {
            "/exit",
            "/quit",
        }:
            report = (
                agent.flush_long_term_memory(
                    thread_id=thread_id,
                    user_id=user_id,
                    trigger=(
                        "session_exit"
                    ),
                )
            )

            print_flush_report(
                report
            )

            print(
                "程序已结束。"
            )

            break
        if command == "/checkpoints":

            print_checkpoint_history(
                agent,
                thread_id,
                limit=20,
            )

            continue


        if command == "/checkpoint":

            print(
                "格式："
                "/checkpoint <checkpoint_id>"
            )

            continue


        if command.startswith(
            "/checkpoint "
        ):

            checkpoint_id = (
                user_input[
                    len(
                        "/checkpoint "
                    ):
                ]
                .strip()
            )

            if not checkpoint_id:

                print(
                    "格式："
                    "/checkpoint <checkpoint_id>"
                )

                continue

            print_checkpoint_detail(
                agent,
                thread_id,
                checkpoint_id,
                maximum_characters=(
                    args.history_preview
                ),
            )

            
            continue



        # ----------------------------------------------------
        # /history
        # ----------------------------------------------------

        if command == "/history":
            print_thread_history(
                agent,
                thread_id,
                maximum_characters=(
                    args.history_preview
                ),
            )

            continue

        # ----------------------------------------------------
        # /replay
        # ----------------------------------------------------

        if command == "/replay":

            print(
                "格式："
                "/replay <checkpoint_id>"
            )

            continue


        if command.startswith(
            "/replay "
        ):

            checkpoint_id = (
                user_input[
                    len("/replay "):
                ]
                .strip()
            )

            if not checkpoint_id:

                print(
                    "格式："
                    "/replay <checkpoint_id>"
                )

                continue

            # ------------------------------------------------
            # 1. 读取目标 Checkpoint
            # ------------------------------------------------

            try:

                checkpoint = (
                    agent.get_thread_checkpoint(
                        thread_id,
                        checkpoint_id,
                    )
                )

            except Exception as error:

                print(
                    "读取 Checkpoint 失败："
                    f"{type(error).__name__}："
                    f"{error}"
                )

                continue

            if not checkpoint:

                print(
                    "未找到 Checkpoint："
                    f"{checkpoint_id}"
                )

                continue

            next_nodes = list(
                checkpoint.get(
                    "next_nodes",
                    [],
                )
                or []
            )

            # ------------------------------------------------
            # 2. Replay 预览
            # ------------------------------------------------

            print()
            print("=" * 80)
            print(
                "Checkpoint Replay Preview"
            )
            print("=" * 80)

            print(
                "thread_id："
                f"{thread_id}"
            )

            print(
                "Step："
                f"{checkpoint.get('step')}"
            )

            print(
                "Checkpoint："
                f"{checkpoint_id}"
            )

            print(
                "时间："
                + format_checkpoint_time(
                    checkpoint.get(
                        "created_at",
                        "",
                    )
                )
            )

            print(
                "下一步："
                + (
                    ", ".join(
                        str(node)
                        for node in next_nodes
                    )
                    if next_nodes
                    else "END"
                )
            )

            # ------------------------------------------------
            # 3. END Checkpoint 无法继续执行
            # ------------------------------------------------

            if not next_nodes:

                print()
                print(
                    "该 Checkpoint 已经位于 END，"
                    "没有后续节点可以 Replay。"
                )

                print(
                    "请选择 next=agent、tools、"
                    "finalize 或 memory_manager "
                    "的更早 Checkpoint。"
                )

                continue

            # ------------------------------------------------
            # 4. 第一版暂不从 __start__ 回放
            # ------------------------------------------------

            if "__start__" in next_nodes:

                print()
                print(
                    "当前 Replay 暂不支持"
                    "从 __start__ 输入边界开始。"
                )

                print(
                    "请选择 next=agent、tools、"
                    "finalize 或 memory_manager "
                    "的 Checkpoint。"
                )

                continue

            # ------------------------------------------------
            # 5. 风险提示
            # ------------------------------------------------

            print()
            print(
                "警告：Replay 会真正重新执行"
                "该 Checkpoint 之后的节点。"
            )

            print(
                "LLM、Tool、API 调用和 Interrupt "
                "都可能再次触发。"
            )

            print(
                "原来的 Checkpoint 历史不会被删除，"
                "新的执行会形成新的历史分支。"
            )

            print()

            # ------------------------------------------------
            # 6. 二次确认
            # ------------------------------------------------

            confirmation = input(
                "确认执行请输入 REPLAY："
            ).strip()

            if confirmation != "REPLAY":

                print(
                    "已取消 Replay。"
                )

                continue

            # ------------------------------------------------
            # 7. 真正执行 Replay
            # ------------------------------------------------

            try:

                replay_result = (
                    agent.replay_checkpoint(
                        thread_id=thread_id,
                        user_id=user_id,
                        checkpoint_id=(
                            checkpoint_id
                        ),
                    )
                )

            except Exception as error:

                print(
                    "Replay 失败："
                    f"{type(error).__name__}："
                    f"{error}"
                )

                continue

            # ------------------------------------------------
            # 8. 输出结果
            # ------------------------------------------------

            print_checkpoint_replay_result(
                replay_result
            )

            continue

        # ----------------------------------------------------
        # /summary
        # ----------------------------------------------------

        if command == "/summary":
            print_memory_status(
                agent,
                thread_id,
                maximum_characters=(
                    args.history_preview
                ),
            )

            continue

        # ----------------------------------------------------
        # /memories
        # ----------------------------------------------------

        if command == "/memories":
            print_long_term_memories(
                agent,
                user_id,
            )

            continue

        # ----------------------------------------------------
        # /flush-memory
        # ----------------------------------------------------

        if command == "/flush-memory":
            report = (
                agent.flush_long_term_memory(
                    thread_id=thread_id,
                    user_id=user_id,
                    trigger=(
                        "manual_flush"
                    ),
                )
            )

            print_flush_report(
                report
            )

            continue

        # ----------------------------------------------------
        # /memory-report
        # ----------------------------------------------------

        if command == "/memory-report":
            report = (
                agent.get_last_auto_memory_report(
                    thread_id=thread_id
                )
            )

            print_flush_report(
                report
                if report
                else {
                    "status": (
                        "尚未执行自动整理"
                    )
                }
            )

            continue

        # ----------------------------------------------------
        # /new
        # ----------------------------------------------------

        if command == "/new":
            report = (
                agent.flush_long_term_memory(
                    thread_id=thread_id,
                    user_id=user_id,
                    trigger=(
                        "before_new_thread"
                    ),
                )
            )

            print_flush_report(
                report
            )

            thread_id = (
                create_thread_id()
            )

            print()

            print(
                "已切换到新 thread："
                f"{thread_id}"
            )

            continue

        # ----------------------------------------------------
        # /remember
        # ----------------------------------------------------

        if command.startswith(
            "/remember "
        ):
            expression = user_input[
                len("/remember "):
            ].strip()

            if "=" not in expression:
                print(
                    "格式："
                    "/remember key=value"
                )

                continue

            key, content = (
                expression.split(
                    "=",
                    maxsplit=1,
                )
            )

            result = agent.remember(
                user_id=user_id,
                key=key,
                content=content,
            )

            print(
                "长期记忆已保存："
                f"{result['key']}"
            )

            continue

        # ----------------------------------------------------
        # /forget
        # ----------------------------------------------------

        if command.startswith(
            "/forget "
        ):
            key = user_input[
                len("/forget "):
            ].strip()

            deleted = agent.forget(
                user_id=user_id,
                key=key,
            )

            print(
                "删除成功。"
                if deleted
                else "未找到该记忆。"
            )

            continue

        # ----------------------------------------------------
        # 普通自然语言输入
        # ----------------------------------------------------
        #
        # CLI 最终真正需要做的事情只有：
        #
        #     用户输入
        #         ↓
        #     agent.run()
        #         ↓
        #     print_result()
        #
        # ----------------------------------------------------

        try:
            result = agent.run(
                user_input,
                user_id=user_id,
                thread_id=thread_id,
            )

        except Exception as error:
            print(
                "执行失败："
                f"{type(error).__name__}："
                f"{error}"
            )

            continue

        print_result(
            result
        )


# ============================================================
# Python CLI Entry Point
# ============================================================


if __name__ == "__main__":
    main()