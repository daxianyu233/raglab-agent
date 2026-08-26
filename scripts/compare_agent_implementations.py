"""比较手写循环 Agent 和 LangGraph Agent。

本脚本使用：

1. 同一个 DeepSeek 模型；
2. 同一个 BM25 索引；
3. 同一个 search_knowledge_base 工具；
4. 同一个系统提示；
5. 同一个用户问题。

分别运行：

A. 手写 for 循环 RetrievalAgent；
B. LangGraph StateGraph RetrievalAgent。

比较重点不是答案文字必须完全相同，而是：

1. 是否都产生工具调用；
2. 是否都执行同一个知识库工具；
3. 工具查询参数是否合理；
4. 是否都在读取工具结果后生成最终回答；
5. 模型调用次数是否相近；
6. 工具调用次数是否相近；
7. LangGraph 实际经过了哪些节点；
8. 两种实现的耗时差异。

注意：

即使 temperature=0，两次独立 API 请求生成的答案
也不一定逐字完全一致。因此不要用字符串完全相等
作为两种 Agent 逻辑是否一致的判断标准。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

from raglab.agent.langgraph_retrieval_agent import (
    LangGraphRetrievalAgent,
    LangGraphRetrievalResult,
)
from raglab.agent.retrieval_agent import (
    RETRIEVAL_AGENT_SYSTEM_PROMPT,
    RetrievalAgent,
    RetrievalAgentResult,
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


DEFAULT_CONFIG_PATH = (
    CONFIG_DIR
    / "agent.yaml"
)

ImplementationName = Literal[
    "manual",
    "langgraph",
    "both",
]


def parse_args() -> argparse.Namespace:
    """读取命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "比较手写循环 Retrieval Agent "
            "和 LangGraph Retrieval Agent。"
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Agent YAML 配置文件路径。",
    )

    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help=(
            "需要提交给两个 Agent 的问题。"
            "未提供时在控制台输入。"
        ),
    )

    parser.add_argument(
        "--implementation",
        type=str,
        choices=[
            "manual",
            "langgraph",
            "both",
        ],
        default="both",
        help=(
            "选择运行手写 Agent、LangGraph Agent，"
            "或者同时运行两者。默认 both。"
        ),
    )

    parser.add_argument(
        "--answer-preview",
        type=int,
        default=2000,
        help=(
            "每个最终答案最多显示多少字符。"
            "默认 2000。"
        ),
    )

    parser.add_argument(
        "--show-tool-output",
        action="store_true",
        help="显示完整的工具返回内容。",
    )

    parser.add_argument(
        "--tool-output-preview",
        type=int,
        default=500,
        help=(
            "未使用 --show-tool-output 时，"
            "工具结果最多预览多少字符。"
        ),
    )

    return parser.parse_args()


def resolve_question(
    command_line_question: str | None,
) -> str:
    """读取并检查问题。"""

    if command_line_question is None:
        question = input(
            "\n请输入测试问题："
        )
    else:
        question = command_line_question

    normalized = question.strip()

    if not normalized:
        raise ValueError(
            "测试问题不能为空。"
        )

    return normalized


def format_json(
    value: Any,
) -> str:
    """将对象转换成便于阅读的 JSON。"""

    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        default=str,
    )


def truncate_text(
    text: str,
    maximum_characters: int,
) -> str:
    """按字符数截断文本。"""

    normalized = str(text).strip()

    if maximum_characters <= 0:
        return ""

    if len(normalized) <= maximum_characters:
        return normalized

    return (
        normalized[:maximum_characters]
        .rstrip()
        + "\n……内容已截断……"
    )


def sum_usage(
    usage_items: list[
        dict[str, Any]
    ],
) -> dict[str, int]:
    """汇总多次模型调用的 Token 使用量。"""

    input_tokens = 0
    output_tokens = 0
    total_tokens = 0

    for usage in usage_items:
        if not isinstance(usage, dict):
            continue

        try:
            input_tokens += int(
                usage.get(
                    "input_tokens",
                    0,
                )
                or 0
            )

        except (
            TypeError,
            ValueError,
        ):
            pass

        try:
            output_tokens += int(
                usage.get(
                    "output_tokens",
                    0,
                )
                or 0
            )

        except (
            TypeError,
            ValueError,
        ):
            pass

        try:
            total_tokens += int(
                usage.get(
                    "total_tokens",
                    0,
                )
                or 0
            )

        except (
            TypeError,
            ValueError,
        ):
            pass

    if total_tokens == 0:
        total_tokens = (
            input_tokens
            + output_tokens
        )

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def manual_execution_path(
    result: RetrievalAgentResult,
) -> list[str]:
    """根据手写 Agent 记录还原执行路径。"""

    path = ["START"]

    for model_call in result.model_calls:
        path.append("agent")

        if model_call.has_tool_calls:
            path.append("tools")

    if result.stopped_by_max_steps:
        path.append("finalize")

    path.append("END")

    return path


def langgraph_execution_path(
    result: LangGraphRetrievalResult,
) -> list[str]:
    """根据 LangGraph 节点轨迹还原执行路径。"""

    path = ["START"]

    for trace in result.model_trace:
        node_name = str(
            trace.get(
                "node",
                "unknown",
            )
        )

        path.append(node_name)

        if (
            node_name == "agent"
            and bool(
                trace.get(
                    "has_tool_calls",
                    False,
                )
            )
        ):
            path.append("tools")

    path.append("END")

    return path


def print_path(
    title: str,
    path: list[str],
) -> None:
    """打印执行路径。"""

    print()
    print(title)
    print("-" * 80)
    print(" → ".join(path))


def print_manual_result(
    result: RetrievalAgentResult,
    *,
    answer_preview: int,
    show_tool_output: bool,
    tool_output_preview: int,
) -> None:
    """打印手写循环 Agent 结果。"""

    print()
    print("=" * 80)
    print("手写循环 Agent")
    print("=" * 80)

    print()
    print("最终答案：")
    print(
        truncate_text(
            result.answer,
            answer_preview,
        )
    )

    print_path(
        "执行顺序",
        manual_execution_path(
            result
        ),
    )

    print()
    print("模型调用轨迹")
    print("-" * 80)

    for model_call in result.model_calls:
        print(
            f"步骤 {model_call.step_index}："
            f"tool_calls="
            f"{model_call.tool_call_count}，"
            f"耗时="
            f"{model_call.latency_ms:.2f} ms"
        )

        if model_call.usage_metadata:
            print(
                "Token："
                f"{format_json(model_call.usage_metadata)}"
            )

    print()
    print("工具调用轨迹")
    print("-" * 80)

    if not result.tool_calls:
        print("没有调用工具。")

    for index, tool_call in enumerate(
        result.tool_calls,
        start=1,
    ):
        print()
        print(
            f"第 {index} 次工具调用"
        )

        print(
            f"工具名称："
            f"{tool_call.tool_name}"
        )

        print("工具参数：")
        print(
            format_json(
                tool_call.arguments
            )
        )

        print(
            f"工具耗时："
            f"{tool_call.latency_ms:.2f} ms"
        )

        print("工具输出：")

        if show_tool_output:
            print(tool_call.output)
        else:
            print(
                truncate_text(
                    tool_call.output,
                    tool_output_preview,
                )
            )

    usage_summary = sum_usage(
        [
            item.usage_metadata
            for item in result.model_calls
        ]
    )

    print()
    print("整体统计")
    print("-" * 80)

    print(
        f"模型调用次数："
        f"{result.model_call_count}"
    )

    print(
        f"工具调用次数："
        f"{result.tool_call_count}"
    )

    print(
        "是否正常结束："
        f"{result.completed_normally}"
    )

    print(
        "是否达到最大步骤："
        f"{result.stopped_by_max_steps}"
    )

    print(
        f"输入 Token："
        f"{usage_summary['input_tokens']}"
    )

    print(
        f"输出 Token："
        f"{usage_summary['output_tokens']}"
    )

    print(
        f"总 Token："
        f"{usage_summary['total_tokens']}"
    )

    print(
        f"总耗时："
        f"{result.total_latency_ms:.2f} ms"
    )


def print_langgraph_result(
    result: LangGraphRetrievalResult,
    *,
    answer_preview: int,
    show_tool_output: bool,
    tool_output_preview: int,
) -> None:
    """打印 LangGraph Agent 结果。"""

    print()
    print("=" * 80)
    print("LangGraph Agent")
    print("=" * 80)

    print()
    print("最终答案：")
    print(
        truncate_text(
            result.answer,
            answer_preview,
        )
    )

    print_path(
        "图节点执行路径",
        langgraph_execution_path(
            result
        ),
    )

    print()
    print("模型节点轨迹")
    print("-" * 80)

    for trace in result.model_trace:
        node_name = trace.get(
            "node",
            "unknown",
        )

        llm_call_index = trace.get(
            "llm_call_index",
            "N/A",
        )

        tool_call_count = trace.get(
            "tool_call_count",
            0,
        )

        latency_ms = float(
            trace.get(
                "latency_ms",
                0.0,
            )
            or 0.0
        )

        print(
            f"节点={node_name}，"
            f"模型调用={llm_call_index}，"
            f"tool_calls={tool_call_count}，"
            f"耗时={latency_ms:.2f} ms"
        )

        tool_calls = trace.get(
            "tool_calls",
            [],
        )

        if tool_calls:
            print("模型生成的工具调用：")
            print(
                format_json(
                    tool_calls
                )
            )

        usage = trace.get(
            "usage_metadata",
            {},
        )

        if usage:
            print("Token：")
            print(
                format_json(
                    usage
                )
            )

        print("-" * 80)

    print()
    print("工具节点轨迹")
    print("-" * 80)

    if not result.tool_trace:
        print("没有经过 tools 节点。")

    for index, trace in enumerate(
        result.tool_trace,
        start=1,
    ):
        print()
        print(
            f"第 {index} 条工具记录"
        )

        print(
            f"工具名称："
            f"{trace.get('tool_name', 'N/A')}"
        )

        print(
            f"Tool Call ID："
            f"{trace.get('tool_call_id', 'N/A')}"
        )

        print("工具参数：")
        print(
            format_json(
                trace.get(
                    "arguments",
                    {},
                )
            )
        )

        print(
            "ToolNode 整体耗时："
            f"{float(trace.get('node_latency_ms', 0.0)):.2f} ms"
        )

        output = str(
            trace.get(
                "output",
                "",
            )
        )

        print("工具输出：")

        if show_tool_output:
            print(output)
        else:
            print(
                truncate_text(
                    output,
                    tool_output_preview,
                )
            )

    usage_summary = sum_usage(
        [
            trace.get(
                "usage_metadata",
                {},
            )
            for trace in (
                result.model_trace
            )
        ]
    )

    print()
    print("整体统计")
    print("-" * 80)

    print(
        f"模型调用次数："
        f"{result.llm_call_count}"
    )

    print(
        f"工具调用次数："
        f"{result.tool_call_count}"
    )

    print(
        "是否正常结束："
        f"{result.completed_normally}"
    )

    print(
        "是否达到最大步骤："
        f"{result.stopped_by_max_steps}"
    )

    print(
        f"输入 Token："
        f"{usage_summary['input_tokens']}"
    )

    print(
        f"输出 Token："
        f"{usage_summary['output_tokens']}"
    )

    print(
        f"总 Token："
        f"{usage_summary['total_tokens']}"
    )

    print(
        f"总耗时："
        f"{result.total_latency_ms:.2f} ms"
    )


def print_comparison(
    manual_result: RetrievalAgentResult,
    graph_result: LangGraphRetrievalResult,
) -> None:
    """对比两个实现的核心运行指标。"""

    manual_usage = sum_usage(
        [
            item.usage_metadata
            for item in (
                manual_result.model_calls
            )
        ]
    )

    graph_usage = sum_usage(
        [
            trace.get(
                "usage_metadata",
                {},
            )
            for trace in (
                graph_result.model_trace
            )
        ]
    )

    print()
    print("=" * 80)
    print("实现方式对比")
    print("=" * 80)

    header = (
        f"{'指标':<24}"
        f"{'手写循环':>18}"
        f"{'LangGraph':>18}"
    )

    print(header)
    print("-" * 60)

    rows = [
        (
            "模型调用次数",
            manual_result.model_call_count,
            graph_result.llm_call_count,
        ),
        (
            "工具调用次数",
            manual_result.tool_call_count,
            graph_result.tool_call_count,
        ),
        (
            "输入 Token",
            manual_usage["input_tokens"],
            graph_usage["input_tokens"],
        ),
        (
            "输出 Token",
            manual_usage["output_tokens"],
            graph_usage["output_tokens"],
        ),
        (
            "总 Token",
            manual_usage["total_tokens"],
            graph_usage["total_tokens"],
        ),
        (
            "是否正常结束",
            manual_result.completed_normally,
            graph_result.completed_normally,
        ),
        (
            "是否达到步骤上限",
            manual_result.stopped_by_max_steps,
            graph_result.stopped_by_max_steps,
        ),
        (
            "总耗时/ms",
            f"{manual_result.total_latency_ms:.2f}",
            f"{graph_result.total_latency_ms:.2f}",
        ),
    ]

    for name, manual_value, graph_value in rows:
        print(
            f"{name:<24}"
            f"{str(manual_value):>18}"
            f"{str(graph_value):>18}"
        )

    manual_tool_names = [
        call.tool_name
        for call in manual_result.tool_calls
    ]

    graph_tool_names = [
        str(
            trace.get(
                "tool_name",
                "",
            )
        )
        for trace in graph_result.tool_trace
    ]

    print()
    print(
        "工具名称是否一致："
        f"{manual_tool_names == graph_tool_names}"
    )

    print(
        "手写循环工具："
        f"{manual_tool_names}"
    )

    print(
        "LangGraph 工具："
        f"{graph_tool_names}"
    )

    print()
    print(
        "说明：两个 Agent 会分别调用一次 DeepSeek，"
        "因此答案文字、Token 和耗时不要求完全一致。"
    )

    print(
        "这里主要确认两者是否遵循相同的"
        "“模型 → 工具 → 模型”业务流程。"
    )


def main() -> None:
    """程序入口。"""

    args = parse_args()

    if args.answer_preview <= 0:
        raise ValueError(
            "--answer-preview 必须大于 0。"
        )

    if args.tool_output_preview <= 0:
        raise ValueError(
            "--tool-output-preview 必须大于 0。"
        )

    implementation: ImplementationName = (
        args.implementation
    )

    config_path = Path(
        args.config
    ).resolve()

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
            "当前对比脚本只支持 BM25，"
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

    question = resolve_question(
        args.question
    )

    print("=" * 80)
    print("手写 Agent 与 LangGraph Agent 对比")
    print("=" * 80)

    print(
        f"运行模式：{implementation}"
    )

    print(
        f"Agent 配置：{config_path}"
    )

    print(
        f"BM25 配置：{bm25_config_path}"
    )

    print(
        f"最大 Agent 步数：{max_steps}"
    )

    print()
    print("测试问题：")
    print(question)

    bm25_index, build_info = (
        build_bm25_index(
            bm25_config_path
        )
    )

    print()
    print(
        "BM25 索引构建完成："
        f"{build_info['chunk_count']} 个 Chunk"
    )

    chat_model = create_deepseek_model(
        model_config
    )

    search_tool = create_bm25_search_tool(
        bm25_index=bm25_index,
        default_top_k=default_top_k,
        maximum_top_k=maximum_top_k,
        max_characters_per_document=(
            max_characters_per_document
        ),
    )

    manual_result: (
        RetrievalAgentResult
        | None
    ) = None

    graph_result: (
        LangGraphRetrievalResult
        | None
    ) = None

    if implementation in {
        "manual",
        "both",
    }:
        manual_agent = RetrievalAgent(
            chat_model=chat_model,
            tools=[search_tool],
            max_steps=max_steps,
            system_prompt=(
                RETRIEVAL_AGENT_SYSTEM_PROMPT
            ),
        )

        print()
        print("正在运行手写循环 Agent……")

        manual_result = manual_agent.run(
            question
        )

        print_manual_result(
            manual_result,
            answer_preview=(
                args.answer_preview
            ),
            show_tool_output=(
                args.show_tool_output
            ),
            tool_output_preview=(
                args.tool_output_preview
            ),
        )

    if implementation in {
        "langgraph",
        "both",
    }:
        graph_agent = (
            LangGraphRetrievalAgent(
                chat_model=chat_model,
                tools=[search_tool],
                max_steps=max_steps,
                system_prompt=(
                    RETRIEVAL_AGENT_SYSTEM_PROMPT
                ),
            )
        )

        print()
        print("正在运行 LangGraph Agent……")

        graph_result = graph_agent.run(
            question
        )

        print_langgraph_result(
            graph_result,
            answer_preview=(
                args.answer_preview
            ),
            show_tool_output=(
                args.show_tool_output
            ),
            tool_output_preview=(
                args.tool_output_preview
            ),
        )

    if (
        manual_result is not None
        and graph_result is not None
    ):
        print_comparison(
            manual_result,
            graph_result,
        )


if __name__ == "__main__":
    main()