"""自动长期记忆 LangGraph 多工具 Agent 控制台。

当前 Agent 支持：

1. 普通 PDF 知识库查询；
2. 已有 GitHub 技术情报查询；
3. Skill Catalog 查看；
4. Skill 按需加载；
5. 已加载 Skill 的动态 Tool 调用；
6. 短期会话持久状态；
7. 滚动摘要；
8. 自动长期记忆提取。

GitHub 技术情报更新不再作为启动时静态 Tool 注册。

当用户明确要求更新 GitHub 技术情报时：

    Agent
      -> load_skill("github-intelligence-update")
      -> Agent 重新决策
      -> update_github_intelligence
      -> Agent 生成最终回答
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from raglab.agent.automatic_long_term_memory_agent import (
    AutomaticLongTermMemoryAgent,
)
from raglab.agent.long_term_memory_agent import (
    normalize_user_id,
)
from raglab.agent.skill_runtime import (
    SkillRuntime,
)
from raglab.agent.tools import (
    create_agent_tools,
    get_tool_names,
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


def read_optional_mapping(
    config: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    """读取可选的字典配置节点。"""

    value = config.get(
        key,
        {},
    )

    if value is None:
        return {}

    if not isinstance(
        value,
        dict,
    ):
        raise ValueError(
            f"配置节点 {key} 必须是字典。"
        )

    return dict(
        value
    )


def build_agent(
    config_path: Path,
) -> AutomaticLongTermMemoryAgent:
    """创建多工具自动长期记忆 Agent。"""

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

    github_config = (
        read_optional_mapping(
            config,
            "github_intelligence",
        )
    )

    retrieval_type = require_string(
        retrieval_config,
        "type",
    ).lower()

    if retrieval_type != "bm25":
        raise ValueError(
            "当前普通 PDF 知识库只支持 BM25，"
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

    include_github_search = bool(
        github_config.get(
            "enable_search_tool",
            True,
        )
    )

    github_default_top_k = int(
        github_config.get(
            "default_top_k",
            default_top_k,
        )
    )

    github_maximum_top_k = int(
        github_config.get(
            "maximum_top_k",
            maximum_top_k,
        )
    )

    github_max_characters = int(
        github_config.get(
            "max_characters_per_document",
            1800,
        )
    )

    bm25_index, build_info = (
        build_bm25_index(
            bm25_config_path
        )
    )

    print(
        "PDF BM25 索引构建完成："
        f"{build_info['chunk_count']} "
        "个 Chunk"
    )

    chat_model = create_deepseek_model(
        model_config
    )

    # --------------------------------------------------------
    # Skill Runtime
    # --------------------------------------------------------
    #
    # 启动时只 Discover Skill。
    #
    # 此时 github-intelligence-update 只是
    # available skill，并没有被加载，
    # update_github_intelligence 也还不是 Active Tool。
    skill_runtime = SkillRuntime()

    skill_catalog_prompt = (
        skill_runtime.render_catalog_prompt()
    )

    dynamic_system_prompt = (
        f"{CONVERSATIONAL_AGENT_SYSTEM_PROMPT}\n\n"
        "# Skill 按需加载规则\n\n"
        "系统支持 Skill Runtime。"
        "Skill 出现在 Catalog 中只表示它可以被发现，"
        "不表示它已经加载。\n\n"
        "当用户请求明确匹配某个尚未加载的 Skill 时，"
        "先调用 load_skill，并使用 Catalog 中的完整 skill id。"
        "load_skill 成功后不要立即假设任务已经完成，"
        "而应在下一次 Agent 决策中调用该 Skill 新开放的业务 Tool。\n\n"
        "只有用户询问有哪些 Skill、当前 Skill 状态等问题时，"
        "才需要调用 list_skills；"
        "执行普通业务任务时不要求先机械调用 list_skills。\n\n"
        f"{skill_catalog_prompt}"
    )

    agent_tools = create_agent_tools(
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

        include_github_search=(
            include_github_search
        ),

        skill_runtime=(
            skill_runtime
        ),

        github_default_top_k=(
            github_default_top_k
        ),

        github_maximum_top_k=(
            github_maximum_top_k
        ),

        github_max_characters_per_document=(
            github_max_characters
        ),
    )

    print(
        "启动时基础工具："
        + ", ".join(
            get_tool_names(
                agent_tools
            )
        )
    )

    available_skill_ids = (
        skill_runtime.available_skill_ids()
    )

    print(
        "发现的 Skills："
        + (
            ", ".join(
                available_skill_ids
            )
            if available_skill_ids
            else "无"
        )
    )

    print(
        "启动时已加载 Skills：无"
    )

    return AutomaticLongTermMemoryAgent(
        chat_model=chat_model,
        tools=agent_tools,

        max_steps=int(
            agent_config.get(
                "max_steps",
                4,
            )
        ),

        system_prompt=(
            dynamic_system_prompt
        ),

        skill_runtime=(
            skill_runtime
        ),

        keep_recent_turns=int(
            agent_config.get(
                "keep_recent_turns",
                4,
            )
        ),

        summarize_trigger_turns=int(
            agent_config.get(
                "summarize_trigger_turns",
                7,
            )
        ),

        minimum_memory_confidence=float(
            agent_config.get(
                "minimum_memory_confidence",
                0.80,
            )
        ),
    )


def print_flush_report(
    report: dict,
) -> None:
    """打印长期记忆整理报告。"""

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
    """打印当前 Skill Runtime 状态。"""

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
        print("当前 Agent 没有配置 SkillRuntime。")
        return

    status = runtime.status()

    available_skills = status.get(
        "available_skills",
        [],
    )

    if available_skills:
        print("可用 Skills：")

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
        print("可用 Skills：无")

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


def print_help() -> None:
    """打印命令。"""

    print()
    print("/skills               查看 Skill Catalog 和加载状态")
    print("/tools                查看当前 Active Tools")
    print("/history              查看短期详细历史")
    print("/summary              查看滚动摘要")
    print("/memories             查看长期记忆")
    print("/flush-memory         手动执行保底整理")
    print("/memory-report        查看最近整理报告")
    print("/remember key=value  显式写入长期记忆")
    print("/forget key          删除长期记忆")
    print("/new                 整理后创建新会话")
    print("/exit                整理后退出")


def main() -> None:
    """程序入口。"""

    args = parse_args()

    user_id = normalize_user_id(
        args.user_id
    )

    thread_id = normalize_thread_id(
        args.thread_id
        if args.thread_id
        else create_thread_id()
    )

    agent = build_agent(
        Path(
            args.config
        ).resolve()
    )

    print()
    print("=" * 80)
    print("自动长期记忆多工具 Agent")
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

    while True:
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

        if not user_input:
            continue

        command = (
            user_input.lower()
        )

        if command == "/help":
            print_help()
            continue

        if command == "/skills":
            print_skill_status(
                agent
            )
            continue

        if command == "/tools":
            print()
            print(
                "当前 Active Tools："
                + ", ".join(
                    agent.get_active_tool_names()
                )
            )
            continue

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

        if command == "/history":
            print_thread_history(
                agent,
                thread_id,
                maximum_characters=(
                    args.history_preview
                ),
            )
            continue

        if command == "/summary":
            print_memory_status(
                agent,
                thread_id,
                maximum_characters=(
                    args.history_preview
                ),
            )
            continue

        if command == "/memories":
            print_long_term_memories(
                agent,
                user_id,
            )
            continue

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


if __name__ == "__main__":
    main()