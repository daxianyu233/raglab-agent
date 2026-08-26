"""带会话历史的 DeepSeek 多工具 Agent。

当前 Agent 支持三个工具：

1. search_knowledge_base
   查询普通 PDF 学习知识库。

2. search_github_intelligence
   查询已经采集并建立索引的 GitHub 技术情报。

3. update_github_intelligence
   执行 GitHub 技术情报增量更新流水线。

工具路由原则：

- 查询 PDF 中的技术概念、实现方法和实验结论：
  使用 search_knowledge_base。

- 查询已经收录的 GitHub 项目、热点、日报和趋势：
  使用 search_github_intelligence。

- 用户明确要求更新、刷新、同步或重新采集 GitHub 情报：
  使用 update_github_intelligence。

- 用户只要求改写、翻译或整理已有回答：
  不调用工具。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)

from raglab.agent.retrieval_agent import (
    RetrievalAgent,
    RetrievalAgentResult,
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


DEFAULT_CONFIG_PATH = (
    CONFIG_DIR
    / "agent.yaml"
)


CONVERSATIONAL_AGENT_SYSTEM_PROMPT = """你是一个带有会话历史和多工具能力的本地技术助手。

你可以使用以下三个工具：

1. search_knowledge_base
   搜索本地 PDF 学习知识库。

2. search_github_intelligence
   搜索已经采集并建立索引的 GitHub 技术情报。

3. update_github_intelligence
   执行 GitHub 技术情报更新流水线，包括采集、差异检测、项目分析、日报生成和索引重建。

你必须根据用户当前请求和会话历史，选择正确的工具。

一、普通 PDF 知识库查询

下列情况应使用 search_knowledge_base：

1. 用户询问 PDF 学习语料中的技术概念、实现方法、实验结果、配置、步骤或结论；
2. 用户的问题与 GitHub 每日技术情报无关，但需要本地 PDF 资料支持；
3. 用户明确要求查询普通知识库、PDF 资料或学习资料。

search_knowledge_base 内部已经固定使用 BM25。
你不能选择或切换为 Dense、Hybrid 或其他检索算法。

二、已有 GitHub 技术情报查询

下列情况应使用 search_github_intelligence：

1. 用户查询已经收录的 GitHub 项目；
2. 用户询问某一天的 GitHub 技术热点；
3. 用户要求查看已有 GitHub 日报、项目摘要或热点分析；
4. 用户要求比较已收录的多个 GitHub 项目；
5. 用户询问 Agent、RAG、Skill、MCP、开源模型等 GitHub 技术趋势；
6. 用户说“查看今天的 GitHub 情报”“今天有哪些值得关注的项目”等查询性表达，但没有明确要求重新采集；
7. 用户询问刚刚更新完成的 GitHub 情报。

search_github_intelligence 只读取已有持久化索引，不会访问 GitHub 网站，也不会执行更新。

三、GitHub 技术情报更新

只有在用户明确表达以下意图时，才调用 update_github_intelligence：

1. 更新 GitHub 技术情报；
2. 刷新 GitHub 数据；
3. 同步最新 GitHub 项目；
4. 重新采集今天的 GitHub 信息；
5. 运行 GitHub 每日情报流水线；
6. 重建或更新 GitHub 情报索引；
7. 检查 GitHub 是否有新的项目变化，并要求执行更新。

例如：

- “更新今天的 GitHub 技术情报”
- “重新抓取一次 GitHub 项目”
- “刷新 GitHub 日报”
- “同步最新的 GitHub Agent 项目”
- “运行今天的信息更新 Skill”

这些请求应调用 update_github_intelligence，而不是 search_knowledge_base 或 search_github_intelligence。

四、更新工具的安全规则

1. update_github_intelligence 可能执行较长时间，并可能调用外部 API；
2. 同一轮用户请求中最多主动调用一次 update_github_intelligence；
3. 不要因为用户只是查询已有资料就擅自执行更新；
4. 不要为了展示 Agent 能力而执行更新；
5. 更新工具返回 failed、busy 或 timeout 时，不得声称更新成功；
6. 更新失败后，不要在同一轮中机械重复调用更新工具；
7. 用户没有明确要求时，不得更新 GitHub 情报；
8. 用户要求更新 ArXiv、Hugging Face、新闻网站或其他非 GitHub 来源时，不得调用该工具；
9. update_github_intelligence 只适用于本项目已经定义好的 GitHub 技术情报流水线。

五、同时更新并查询

当用户明确要求：

- 先更新 GitHub 情报，再回答最新项目；
- 更新后列出今天的热点；
- 刷新数据并总结结果；

可以按以下顺序执行：

1. 调用 update_github_intelligence；
2. 检查工具返回状态；
3. 只有更新成功时，再调用 search_github_intelligence；
4. 根据新索引中的资料生成最终回答。

如果更新失败、超时或正在被其他进程执行，应直接说明状态。
不要假装已经查询到最新数据。

六、明确不应调用工具的情况

当用户只是要求处理当前会话中已有内容，并且不需要新事实时，不要调用工具，包括：

1. 缩短、总结或概括上一轮回答；
2. 改写、润色或重新组织上一轮回答；
3. 翻译上一轮回答；
4. 将上一轮回答转换成表格、列表或其他格式；
5. 调整已有回答的语气、长度或结构；
6. 从已有回答中提取要点；
7. 对已有回答进行不增加新事实的解释或重述；
8. 普通打招呼、致谢或结束对话。

这些情况下只能使用会话历史中已有内容。

七、检索查询参数

调用 search_knowledge_base 或 search_github_intelligence 时：

1. query 必须是完整、独立、适合检索的问题；
2. 不要直接使用“它”“这个”“刚才那个”等模糊表达；
3. 应根据会话历史补全项目名、主题、日期和上下文；
4. 第一次检索结果不足时，可以修改 query 后再次检索；
5. 不要机械重复完全相同的查询；
6. 不要进行没有明确目的的多次检索；
7. top_k 通常使用 3 到 5；
8. 只有需要比较多个项目或多个来源时才提高 top_k。

八、回答依据和引用

1. search_knowledge_base 返回的资料使用 [资料1]、[资料2] 等编号；
2. search_github_intelligence 返回的资料使用 [GitHub资料1]、[GitHub资料2] 等编号；
3. 只有本轮实际调用对应检索工具后，才能使用相应资料编号；
4. 引用编号只能指向本轮工具返回的资料；
5. 不得把上一轮的临时资料编号当作本轮证据；
6. 工具资料不足时，应明确说明资料不足；
7. 资料之间存在冲突时，应指出冲突；
8. 不得使用模型自身知识补充本地资料中没有的事实；
9. update_github_intelligence 的执行结果属于运行状态，不需要使用资料编号；
10. 最终回答应直接、清晰，不要完整复述工具返回的全部内容。

九、不调用工具时的限制

如果本轮没有调用任何工具：

1. 只能使用当前会话历史中已经明确出现的信息；
2. 不得增加历史中没有出现的新事实、数字、定义、例子或结论；
3. 不得声称“根据知识库”或“根据 GitHub 情报库”；
4. 不得输出临时资料编号；
5. 历史信息不足时，应调用正确的工具，而不是猜测。

十、总体路由原则

1. PDF 学习资料问题：
   search_knowledge_base。

2. 已有 GitHub 情报查询：
   search_github_intelligence。

3. 明确要求刷新 GitHub 数据：
   update_github_intelligence。

4. 明确要求更新后再查询：
   update_github_intelligence 成功后，再使用 search_github_intelligence。

5. 非 GitHub 更新任务：
   不调用 update_github_intelligence。

6. 对已有回答做文本处理：
   不调用工具。

7. 工具失败：
   如实说明，不虚构成功结果。

8. 不要为了展示工具能力而进行无意义调用。"""


@dataclass(frozen=True)
class ConversationTurn:
    """记录一轮连续对话。"""

    turn_index: int
    question: str
    answer: str
    model_call_count: int
    tool_call_count: int
    total_latency_ms: float
    tool_queries: list[str]


class ConversationMemory:
    """保存精简后的短期会话历史。"""

    def __init__(
        self,
        *,
        max_history_turns: int = 6,
    ) -> None:
        if max_history_turns <= 0:
            raise ValueError(
                "max_history_turns 必须大于 0。"
            )

        self.max_history_turns = int(
            max_history_turns
        )

        self.messages: list[
            BaseMessage
        ] = []

        self.turns: list[
            ConversationTurn
        ] = []

    def clear(self) -> None:
        """清空全部会话历史。"""

        self.messages.clear()
        self.turns.clear()

    def get_messages(
        self,
    ) -> list[BaseMessage]:
        """返回提供给 Agent 的历史消息副本。"""

        return list(
            self.messages
        )

    def get_turns(
        self,
    ) -> list[ConversationTurn]:
        """返回会话轮次副本。"""

        return list(
            self.turns
        )

    def append(
        self,
        *,
        question: str,
        result: RetrievalAgentResult,
    ) -> ConversationTurn:
        """保存当前一轮的精简历史。

        工具的完整输出不会进入会话历史。

        PDF 的 [资料n] 和 GitHub 的 [GitHub资料n]
        都只在当前工具调用轮次有效，因此会在写入历史前删除。
        """

        import re

        tool_queries: list[str] = []

        for tool_call in (
            result.tool_calls
        ):
            query = (
                tool_call.arguments.get(
                    "query"
                )
            )

            if isinstance(
                query,
                str,
            ):
                normalized_query = (
                    query.strip()
                )

                if normalized_query:
                    tool_queries.append(
                        normalized_query
                    )

        turn = ConversationTurn(
            turn_index=(
                len(self.turns) + 1
            ),
            question=question,
            answer=result.answer,
            model_call_count=(
                result.model_call_count
            ),
            tool_call_count=(
                result.tool_call_count
            ),
            total_latency_ms=(
                result.total_latency_ms
            ),
            tool_queries=tool_queries,
        )

        self.turns.append(
            turn
        )

        history_answer = re.sub(
            r"\[资料\d+\]",
            "",
            result.answer,
        )

        history_answer = re.sub(
            r"\[GitHub资料\d+\]",
            "",
            history_answer,
        )

        history_answer = re.sub(
            r"[ \t]+\n",
            "\n",
            history_answer,
        )

        history_answer = re.sub(
            r"\n{3,}",
            "\n\n",
            history_answer,
        ).strip()

        self.messages.extend(
            [
                HumanMessage(
                    content=question
                ),
                AIMessage(
                    content=history_answer
                ),
            ]
        )

        self._trim()

        return turn

    def _trim(self) -> None:
        """只保留最近若干轮历史。"""

        maximum_messages = (
            self.max_history_turns
            * 2
        )

        if (
            len(self.messages)
            > maximum_messages
        ):
            self.messages = (
                self.messages[
                    -maximum_messages:
                ]
            )

        if (
            len(self.turns)
            > self.max_history_turns
        ):
            self.turns = (
                self.turns[
                    -self.max_history_turns:
                ]
            )


def parse_args() -> argparse.Namespace:
    """读取命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "运行带会话状态和 GitHub Skill 的 "
            "DeepSeek 多工具 Agent。"
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(
            DEFAULT_CONFIG_PATH
        ),
        help="Agent 配置文件路径。",
    )

    parser.add_argument(
        "--max-history-turns",
        type=int,
        default=6,
        help=(
            "最多保留多少轮精简会话历史，"
            "默认值为 6。"
        ),
    )

    return parser.parse_args()


def read_optional_mapping(
    config: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    """读取可选字典配置节。"""

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


def format_arguments(
    value: dict[str, Any],
) -> str:
    """格式化工具参数。"""

    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        default=str,
    )


def print_help() -> None:
    """显示交互命令。"""

    print()
    print("=" * 80)
    print("可用命令")
    print("=" * 80)
    print("/help      查看帮助")
    print("/history   查看精简会话历史")
    print("/trace     查看上一轮工具调用过程")
    print("/clear     清空会话历史")
    print("/exit      退出程序")


def print_history(
    memory: ConversationMemory,
) -> None:
    """打印当前会话历史。"""

    turns = memory.get_turns()

    print()
    print("=" * 80)
    print("当前会话历史")
    print("=" * 80)

    if not turns:
        print(
            "当前没有历史对话。"
        )
        return

    for turn in turns:
        print()
        print(
            f"第 {turn.turn_index} 轮"
        )

        print(
            f"用户：{turn.question}"
        )

        print(
            f"助手：{turn.answer}"
        )

        print(
            "工具调用次数："
            f"{turn.tool_call_count}"
        )

        if turn.tool_queries:
            print(
                "工具查询："
            )

            for query in (
                turn.tool_queries
            ):
                print(
                    f"  - {query}"
                )

        print(
            "-" * 80
        )


def print_trace(
    result: (
        RetrievalAgentResult
        | None
    ),
) -> None:
    """打印上一轮 Agent 运行轨迹。"""

    print()
    print("=" * 80)
    print("上一轮 Agent 轨迹")
    print("=" * 80)

    if result is None:
        print(
            "当前还没有执行任何问答。"
        )
        return

    print(
        "模型调用次数："
        f"{result.model_call_count}"
    )

    print(
        "工具调用次数："
        f"{result.tool_call_count}"
    )

    print(
        "总耗时："
        f"{result.total_latency_ms:.2f} ms"
    )

    print(
        "是否正常结束："
        f"{result.completed_normally}"
    )

    print(
        "是否达到最大步数："
        f"{result.stopped_by_max_steps}"
    )

    print()
    print(
        "模型步骤："
    )

    for model_call in (
        result.model_calls
    ):
        print(
            f"  步骤 {model_call.step_index}："
            f"tool_calls="
            f"{model_call.tool_call_count}，"
            f"耗时="
            f"{model_call.latency_ms:.2f} ms"
        )

    print()
    print(
        "工具调用："
    )

    if not result.tool_calls:
        print(
            "  本轮没有调用工具。"
        )
        return

    for index, tool_call in enumerate(
        result.tool_calls,
        start=1,
    ):
        print()
        print(
            f"  第 {index} 次工具调用"
        )

        print(
            "  工具名称："
            f"{tool_call.tool_name}"
        )

        print(
            "  工具参数："
        )

        formatted = format_arguments(
            tool_call.arguments
        )

        for line in (
            formatted.splitlines()
        ):
            print(
                f"    {line}"
            )

        print(
            "  工具耗时："
            f"{tool_call.latency_ms:.2f} ms"
        )

        output_preview = str(
            tool_call.output
        ).strip()

        if len(output_preview) > 600:
            output_preview = (
                output_preview[:600]
                + "……"
            )

        print(
            "  工具结果预览："
        )

        if output_preview:
            for line in (
                output_preview.splitlines()
            ):
                print(
                    f"    {line}"
                )
        else:
            print(
                "    无输出"
            )


def describe_tool_call(
    tool_name: str,
    arguments: dict[str, Any],
) -> str:
    """生成简洁的工具调用说明。"""

    if tool_name in {
        "search_knowledge_base",
        "search_github_intelligence",
    }:
        query = arguments.get(
            "query",
            "N/A",
        )

        top_k = arguments.get(
            "top_k",
            "默认",
        )

        return (
            f"{tool_name}"
            f"(query={query!r}, "
            f"top_k={top_k!r})"
        )

    if (
        tool_name
        == "update_github_intelligence"
    ):
        return (
            "update_github_intelligence"
            "(运行 GitHub 技术情报更新流水线)"
        )

    return (
        f"{tool_name}"
        f"(arguments={arguments!r})"
    )


def print_result(
    result: RetrievalAgentResult,
    *,
    turn_index: int,
) -> None:
    """打印一轮 Agent 回答。"""

    print()
    print("=" * 80)
    print(
        f"Agent 回答｜第 {turn_index} 轮"
    )
    print("=" * 80)

    print(
        result.answer
    )

    print()
    print(
        "-" * 80
    )

    if (
        result.tool_call_count
        == 0
    ):
        print(
            "本轮未调用工具。"
        )

    else:
        print(
            "本轮工具调用次数："
            f"{result.tool_call_count}"
        )

        for index, tool_call in enumerate(
            result.tool_calls,
            start=1,
        ):
            description = (
                describe_tool_call(
                    tool_name=(
                        tool_call.tool_name
                    ),
                    arguments=(
                        tool_call.arguments
                    ),
                )
            )

            print(
                f"  {index}. {description}"
            )

    print(
        "本轮模型调用次数："
        f"{result.model_call_count}"
    )

    print(
        "本轮总耗时："
        f"{result.total_latency_ms:.2f} ms"
    )


def handle_command(
    command: str,
    *,
    memory: ConversationMemory,
    last_result: (
        RetrievalAgentResult
        | None
    ),
) -> bool:
    """处理控制台命令。

    返回 True 表示继续运行；
    返回 False 表示退出程序。
    """

    normalized = (
        command.strip().lower()
    )

    if normalized == "/help":
        print_help()
        return True

    if normalized == "/history":
        print_history(
            memory
        )
        return True

    if normalized == "/trace":
        print_trace(
            last_result
        )
        return True

    if normalized == "/clear":
        memory.clear()

        print()
        print(
            "当前会话历史已清空。"
        )
        return True

    if normalized in {
        "/exit",
        "/quit",
    }:
        print()
        print(
            "连续对话 Agent 已结束。"
        )
        return False

    print()
    print(
        f"无法识别命令：{command}"
    )

    print(
        "输入 /help 查看命令。"
    )

    return True


def main() -> None:
    """程序入口。"""

    args = parse_args()

    if (
        args.max_history_turns
        <= 0
    ):
        raise ValueError(
            "--max-history-turns 必须大于 0。"
        )

    config_path = Path(
        args.config
    ).resolve()

    config = load_yaml_config(
        config_path
    )

    experiment_name = str(
        config.get(
            "experiment_name",
            "conversational_multi_tool_agent",
        )
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
            "当前脚本的普通 PDF 知识库只支持 BM25，"
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

    include_github_update = bool(
        github_config.get(
            "enable_update_tool",
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

    max_steps = int(
        agent_config.get(
            "max_steps",
            4,
        )
    )

    print(
        "=" * 80
    )

    print(
        "RAGLab 连续对话多工具 Agent"
    )

    print(
        "=" * 80
    )

    print(
        f"实验名称：{experiment_name}"
    )

    print(
        f"Agent 配置：{config_path}"
    )

    print(
        f"PDF BM25 配置：{bm25_config_path}"
    )

    print(
        "最大历史轮数："
        f"{args.max_history_turns}"
    )

    print(
        "最大 Agent 步数："
        f"{max_steps}"
    )

    bm25_index, build_info = (
        build_bm25_index(
            bm25_config_path
        )
    )

    print()
    print(
        "PDF BM25 索引构建完成："
        f"{build_info['chunk_count']} 个 Chunk"
    )

    chat_model = create_deepseek_model(
        model_config
    )

    agent_tools = create_agent_tools(
        bm25_index=bm25_index,
        default_top_k=default_top_k,
        maximum_top_k=maximum_top_k,
        max_characters_per_document=(
            max_characters_per_document
        ),
        include_github_search=(
            include_github_search
        ),
        include_github_update=(
            include_github_update
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
        "已注册工具："
        + ", ".join(
            get_tool_names(
                agent_tools
            )
        )
    )

    retrieval_agent = RetrievalAgent(
        chat_model=chat_model,
        tools=agent_tools,
        max_steps=max_steps,
        system_prompt=(
            CONVERSATIONAL_AGENT_SYSTEM_PROMPT
        ),
    )

    memory = ConversationMemory(
        max_history_turns=(
            args.max_history_turns
        )
    )

    last_result: (
        RetrievalAgentResult
        | None
    ) = None

    print()
    print(
        "连续对话多工具 Agent 已启动。"
    )

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
            print(
                "连续对话 Agent 已结束。"
            )
            break

        if not user_input:
            continue

        if user_input.startswith(
            "/"
        ):
            should_continue = (
                handle_command(
                    user_input,
                    memory=memory,
                    last_result=(
                        last_result
                    ),
                )
            )

            if not should_continue:
                break

            continue

        try:
            result = retrieval_agent.run(
                user_input,
                history_messages=(
                    memory.get_messages()
                ),
            )

        except Exception as error:
            print()
            print(
                "=" * 80
            )

            print(
                "本轮执行失败"
            )

            print(
                "=" * 80
            )

            print(
                f"{type(error).__name__}："
                f"{error}"
            )

            continue

        turn = memory.append(
            question=user_input,
            result=result,
        )

        last_result = result

        print_result(
            result,
            turn_index=(
                turn.turn_index
            ),
        )


if __name__ == "__main__":
    main()