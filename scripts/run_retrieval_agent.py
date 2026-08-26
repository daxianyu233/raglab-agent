"""运行并验证 BM25 + DeepSeek Tool-Calling Agent。

本脚本验证的重点不是：

    模型是否需要自行决定使用 BM25、Dense 或 Hybrid。

当前检索算法已经由系统固定为 BM25。

本脚本验证的是：

1. DeepSeek 是否生成 tool_call；
2. tool_call 是否包含正确的工具名和参数；
3. Python 是否成功执行 BM25 工具；
4. 工具结果是否通过 ToolMessage 返回模型；
5. DeepSeek 是否基于工具结果生成最终回答；
6. Agent 循环是否能够正常终止。

验证规则：

    没有调用工具
    → 验证失败

    调用了错误工具
    → 验证失败

    调用了 search_knowledge_base
    → 工具调用机制验证通过
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from raglab.agent.retrieval_agent import (
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


def parse_args() -> argparse.Namespace:
    """读取命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "运行并验证 BM25 + DeepSeek "
            "Tool-Calling Retrieval Agent。"
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
            "需要提交给 Agent 的问题。"
            "未提供时在控制台中输入。"
        ),
    )

    return parser.parse_args()


def resolve_question(
    command_line_question: str | None,
) -> str:
    """读取并检查用户问题。"""

    if command_line_question is None:
        question = input(
            "\n请输入验证问题："
        )
    else:
        question = command_line_question

    normalized = question.strip()

    if not normalized:
        raise ValueError(
            "验证问题不能为空。"
        )

    return normalized


def format_json(
    value: Any,
) -> str:
    """把字典格式化为可读 JSON。"""

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
    """按字符数截断显示文本。"""

    normalized = str(text).strip()

    if len(normalized) <= maximum_characters:
        return normalized

    return (
        normalized[:maximum_characters]
        .rstrip()
        + "\n……工具输出已截断……"
    )


def print_model_calls(
    result: RetrievalAgentResult,
    *,
    show_usage: bool,
) -> None:
    """打印每一次模型决策。"""

    print()
    print("=" * 80)
    print("模型调用记录")
    print("=" * 80)

    if not result.model_calls:
        print("没有记录到模型调用。")
        return

    for model_call in result.model_calls:
        print()
        print(
            f"模型步骤：{model_call.step_index}"
        )

        print(
            "是否产生工具调用："
            f"{'是' if model_call.has_tool_calls else '否'}"
        )

        print(
            "本步骤工具调用数量："
            f"{model_call.tool_call_count}"
        )

        print(
            "模型调用耗时："
            f"{model_call.latency_ms:.2f} ms"
        )

        if show_usage:
            print("Token 使用：")

            if model_call.usage_metadata:
                print(
                    format_json(
                        model_call.usage_metadata
                    )
                )
            else:
                print(
                    "模型没有返回可识别的 "
                    "Token 使用数据。"
                )

        print("-" * 80)


def print_tool_calls(
    result: RetrievalAgentResult,
    *,
    show_arguments: bool,
    show_output: bool,
    output_preview_characters: int,
) -> None:
    """打印工具调用记录。"""

    print()
    print("=" * 80)
    print("工具调用记录")
    print("=" * 80)

    if not result.tool_calls:
        print("本次没有发生任何工具调用。")
        return

    for index, tool_call in enumerate(
        result.tool_calls,
        start=1,
    ):
        print()
        print(f"工具调用序号：{index}")
        print(
            f"Agent 步骤："
            f"{tool_call.step_index}"
        )
        print(
            f"Tool Call ID："
            f"{tool_call.tool_call_id}"
        )
        print(
            f"工具名称："
            f"{tool_call.tool_name}"
        )

        if show_arguments:
            print("工具参数：")
            print(
                format_json(
                    tool_call.arguments
                )
            )

        print(
            "工具执行耗时："
            f"{tool_call.latency_ms:.2f} ms"
        )

        print("工具输出：")

        if show_output:
            print(tool_call.output)
        else:
            print(
                truncate_text(
                    tool_call.output,
                    output_preview_characters,
                )
            )

        print("-" * 80)


def validate_result(
    result: RetrievalAgentResult,
    *,
    require_tool_call: bool,
    expected_tool_name: str,
) -> list[str]:
    """验证 Agent 的工具调用结果。

    Returns
    -------
    list[str]
        验证失败原因。

        空列表表示验证通过。
    """

    errors: list[str] = []

    if (
        require_tool_call
        and result.tool_call_count == 0
    ):
        errors.append(
            "该验证任务要求调用知识库工具，"
            "但模型没有产生任何工具调用。"
        )

    if result.tool_call_count > 0:
        actual_tool_names = {
            tool_call.tool_name
            for tool_call in result.tool_calls
        }

        unexpected_names = (
            actual_tool_names
            - {expected_tool_name}
        )

        if unexpected_names:
            errors.append(
                "检测到非预期工具："
                + ", ".join(
                    sorted(unexpected_names)
                )
            )

        if expected_tool_name not in (
            actual_tool_names
        ):
            errors.append(
                "模型没有调用预期工具："
                f"{expected_tool_name}"
            )

    for index, tool_call in enumerate(
        result.tool_calls,
        start=1,
    ):
        query = tool_call.arguments.get(
            "query"
        )

        if not isinstance(query, str):
            errors.append(
                f"第 {index} 次工具调用缺少"
                "有效字符串 query 参数。"
            )

        elif not query.strip():
            errors.append(
                f"第 {index} 次工具调用的 "
                "query 参数为空。"
            )

        if not tool_call.output.strip():
            errors.append(
                f"第 {index} 次工具调用"
                "没有返回任何内容。"
            )

        if tool_call.output.startswith(
            "工具执行失败"
        ):
            errors.append(
                f"第 {index} 次工具调用执行失败："
                f"{tool_call.output}"
            )

    if not result.answer.strip():
        errors.append(
            "Agent 没有生成最终答案。"
        )

    return errors


def print_validation_result(
    errors: list[str],
) -> None:
    """打印最终验证结果。"""

    print()
    print("=" * 80)
    print("验证结果")
    print("=" * 80)

    if not errors:
        print("验证通过。")
        print()
        print(
            "DeepSeek 已生成工具调用，"
            "Python 已执行 BM25 检索，"
            "工具结果也已返回模型。"
        )
        return

    print("验证失败。")

    for index, error in enumerate(
        errors,
        start=1,
    ):
        print(
            f"{index}. {error}"
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
            "retrieval_agent_validation",
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

    validation_config = require_mapping(
        config,
        "validation",
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
            "当前验证脚本只支持 BM25，"
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

    require_tool_call = bool(
        validation_config.get(
            "require_tool_call",
            True,
        )
    )

    expected_tool_name = str(
        validation_config.get(
            "expected_tool_name",
            "search_knowledge_base",
        )
    ).strip()

    show_tool_arguments = bool(
        display_config.get(
            "show_tool_arguments",
            True,
        )
    )

    show_tool_output = bool(
        display_config.get(
            "show_tool_output",
            False,
        )
    )

    output_preview_characters = int(
        display_config.get(
            "tool_output_preview_characters",
            600,
        )
    )

    show_model_calls = bool(
        display_config.get(
            "show_model_calls",
            True,
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

    question = resolve_question(
        args.question
    )

    print("=" * 80)
    print("RAGLab Retrieval Agent 验证")
    print("=" * 80)
    print(
        f"实验名称：{experiment_name}"
    )
    print(
        f"Agent 配置：{config_path}"
    )
    print(
        f"BM25 配置：{bm25_config_path}"
    )
    print(
        f"预期工具：{expected_tool_name}"
    )
    print(
        f"最大 Agent 步数：{max_steps}"
    )

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

    retrieval_agent = RetrievalAgent(
        chat_model=chat_model,
        tools=[search_tool],
        max_steps=max_steps,
    )

    print()
    print("=" * 80)
    print("验证问题")
    print("=" * 80)
    print(question)

    result = retrieval_agent.run(
        question
    )

    print()
    print("=" * 80)
    print("Agent 最终回答")
    print("=" * 80)
    print(result.answer)

    if show_model_calls:
        print_model_calls(
            result,
            show_usage=show_usage,
        )

    print_tool_calls(
        result,
        show_arguments=show_tool_arguments,
        show_output=show_tool_output,
        output_preview_characters=(
            output_preview_characters
        ),
    )

    if show_latency:
        print()
        print("=" * 80)
        print("整体统计")
        print("=" * 80)
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
            "是否因最大步数停止："
            f"{result.stopped_by_max_steps}"
        )
        print(
            "总耗时："
            f"{result.total_latency_ms:.2f} ms"
        )

    validation_errors = (
        validate_result(
            result,
            require_tool_call=(
                require_tool_call
            ),
            expected_tool_name=(
                expected_tool_name
            ),
        )
    )

    print_validation_result(
        validation_errors
    )

    if validation_errors:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()