"""LangGraph 工具失败、重试、参数纠错与降级演示。

本脚本包含四个相互独立的案例：

1. 暂时性错误：
   工具前两次失败，第三次成功，
   由 RetryPolicy 自动重试。

2. 固定错误：
   工具抛出 ValueError，
   retry_on 判断为不可重试，
   异常直接返回程序入口。

3. 参数错误：
   模拟 LLM 为工具生成错误参数，
   ToolNode 将参数校验错误包装为 ToolMessage。

4. 服务降级：
   第一次调用主服务失败，
   第二次节点尝试时切换到备用服务。

运行方式：

    python -m scripts.test_tool_failure_modes
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import (
    AIMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.graph import (
    END,
    START,
    MessagesState,
    StateGraph,
)
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy
from typing_extensions import TypedDict


class TemporaryServiceError(Exception):
    """模拟网络超时、限流或服务暂时不可用。"""


class FallbackState(TypedDict):
    """服务降级案例的图状态。"""

    result: str


COUNTERS: dict[str, int] = {
    "transient": 0,
    "permanent": 0,
    "fallback_primary": 0,
    "fallback_secondary": 0,
}


def print_title(
    title: str,
) -> None:
    """打印案例标题。"""

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def should_retry(
    error: Exception,
) -> bool:
    """判断一个异常是否允许重试。

    当前只允许 TemporaryServiceError
    触发自动重试。

    ValueError、TypeError 等固定错误
    不会重试。
    """

    retryable = isinstance(
        error,
        TemporaryServiceError,
    )

    decision = (
        "重试"
        if retryable
        else "不重试"
    )

    print(
        "RetryPolicy 判断："
        f"{type(error).__name__}"
        f" -> {decision}"
    )

    return retryable


def build_tool_call_message(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    call_id: str,
) -> AIMessage:
    """构造一条模拟 LLM 工具调用的消息。

    在真实 Agent 中，这条 AIMessage
    由绑定了工具的 LLM 自动生成。

    本测试不调用 DeepSeek，
    因此直接手动构造。
    """

    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": tool_name,
                "args": arguments,
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


def get_last_tool_message(
    result: dict[str, Any],
) -> ToolMessage:
    """读取图执行结果中的最后一条 ToolMessage。"""

    messages = list(
        result.get(
            "messages",
            [],
        )
    )

    if not messages:
        raise RuntimeError(
            "图执行完成后没有返回消息。"
        )

    last_message = messages[-1]

    if not isinstance(
        last_message,
        ToolMessage,
    ):
        raise TypeError(
            "最后一条消息不是 ToolMessage，"
            f"实际类型："
            f"{type(last_message)!r}"
        )

    return last_message


# =============================================================================
# 案例一：暂时性错误，重试后成功
# =============================================================================


@tool
def unstable_search(
    query: str,
) -> str:
    """模拟前两次失败、第三次成功的检索服务。"""

    COUNTERS["transient"] += 1

    attempt = COUNTERS[
        "transient"
    ]

    print(
        "unstable_search 实际执行："
        f"第 {attempt} 次"
    )

    if attempt < 3:
        raise TemporaryServiceError(
            "模拟暂时性服务故障，"
            f"第 {attempt} 次调用失败。"
        )

    return (
        "检索成功："
        f"{query}"
    )


def run_transient_error_demo() -> None:
    """运行暂时性错误案例。"""

    print_title(
        "案例 1：暂时性错误，自动重试后成功"
    )

    COUNTERS["transient"] = 0

    builder = StateGraph(
        MessagesState
    )

    tool_node = ToolNode(
        [
            unstable_search,
        ]
    )

    builder.add_node(
        "tools",
        tool_node,
        retry_policy=RetryPolicy(
            # 包括第一次正常执行在内，
            # 最多尝试三次。
            max_attempts=3,

            # 第一次重试前等待时间。
            initial_interval=0.2,

            # 后续等待时间按两倍增长。
            backoff_factor=2.0,

            # 最大等待时间。
            max_interval=1.0,

            # 测试时关闭随机抖动，
            # 便于观察固定结果。
            jitter=False,

            # 决定哪些异常可以重试。
            retry_on=should_retry,
        ),
    )

    builder.add_edge(
        START,
        "tools",
    )

    builder.add_edge(
        "tools",
        END,
    )

    graph = builder.compile()

    result = graph.invoke(
        {
            "messages": [
                build_tool_call_message(
                    tool_name=(
                        "unstable_search"
                    ),
                    arguments={
                        "query": (
                            "LangGraph "
                            "RetryPolicy"
                        )
                    },
                    call_id=(
                        "call-transient-001"
                    ),
                )
            ]
        }
    )

    tool_message = (
        get_last_tool_message(
            result
        )
    )

    print()
    print(
        "ToolMessage.status："
        f"{getattr(tool_message, 'status', 'unknown')}"
    )

    print(
        "ToolMessage.content："
        f"{tool_message.content}"
    )

    print(
        "最终实际执行次数："
        f"{COUNTERS['transient']}"
    )


# =============================================================================
# 案例二：固定错误，不重试
# =============================================================================


@tool
def broken_search(
    query: str,
) -> str:
    """模拟每次都会发生固定错误的工具。"""

    COUNTERS["permanent"] += 1

    attempt = COUNTERS[
        "permanent"
    ]

    print(
        "broken_search 实际执行："
        f"第 {attempt} 次"
    )

    raise ValueError(
        "模拟固定错误："
        f"无法处理查询 {query!r}。"
    )


def run_permanent_error_demo() -> None:
    """运行固定错误案例。"""

    print_title(
        "案例 2：固定错误，不重试并向外报错"
    )

    COUNTERS["permanent"] = 0

    builder = StateGraph(
        MessagesState
    )

    tool_node = ToolNode(
        [
            broken_search,
        ]
    )

    builder.add_node(
        "tools",
        tool_node,
        retry_policy=RetryPolicy(
            max_attempts=3,
            initial_interval=0.2,
            backoff_factor=2.0,
            max_interval=1.0,
            jitter=False,
            retry_on=should_retry,
        ),
    )

    builder.add_edge(
        START,
        "tools",
    )

    builder.add_edge(
        "tools",
        END,
    )

    graph = builder.compile()

    try:
        graph.invoke(
            {
                "messages": [
                    build_tool_call_message(
                        tool_name=(
                            "broken_search"
                        ),
                        arguments={
                            "query": (
                                "固定错误测试"
                            )
                        },
                        call_id=(
                            "call-permanent-001"
                        ),
                    )
                ]
            }
        )

    except Exception as error:
        print()
        print(
            "程序入口捕获异常："
            f"{type(error).__name__}: "
            f"{error}"
        )

    print(
        "最终实际执行次数："
        f"{COUNTERS['permanent']}"
    )


# =============================================================================
# 案例三：工具参数错误
# =============================================================================


@tool
def typed_search(
    query: str,
    top_k: int,
) -> str:
    """接收字符串查询和整数 top_k 的检索工具。"""

    return (
        f"query={query}, "
        f"top_k={top_k}"
    )


def run_argument_error_demo() -> None:
    """运行工具参数错误案例。"""

    print_title(
        "案例 3：工具参数错误，交给模型修正"
    )

    builder = StateGraph(
        MessagesState
    )

    tool_node = ToolNode(
        [
            typed_search,
        ]
    )

    builder.add_node(
        "tools",
        tool_node,
        retry_policy=RetryPolicy(
            max_attempts=3,
            initial_interval=0.2,
            backoff_factor=2.0,
            max_interval=1.0,
            jitter=False,
            retry_on=should_retry,
        ),
    )

    builder.add_edge(
        START,
        "tools",
    )

    builder.add_edge(
        "tools",
        END,
    )

    graph = builder.compile()

    # typed_search 要求 top_k 是 int，
    # 这里故意模拟 LLM 传入非法字符串。
    result = graph.invoke(
        {
            "messages": [
                build_tool_call_message(
                    tool_name=(
                        "typed_search"
                    ),
                    arguments={
                        "query": (
                            "参数校验测试"
                        ),
                        "top_k": "很多",
                    },
                    call_id=(
                        "call-argument-001"
                    ),
                )
            ]
        }
    )

    tool_message = (
        get_last_tool_message(
            result
        )
    )

    print()
    print(
        "ToolMessage.status："
        f"{getattr(tool_message, 'status', 'unknown')}"
    )

    print(
        "ToolMessage.content："
        f"{tool_message.content}"
    )

    print()
    print(
        "这里没有机械重试相同参数。"
    )

    print(
        "在完整 Agent 中，"
        "这条错误 ToolMessage 会返回给 LLM，"
        "让模型重新生成正确的 top_k。"
    )


# =============================================================================
# 案例四：重试时切换备用服务
# =============================================================================


def primary_service() -> str:
    """模拟始终不可用的主服务。"""

    COUNTERS[
        "fallback_primary"
    ] += 1

    attempt = COUNTERS[
        "fallback_primary"
    ]

    print(
        "主服务实际执行："
        f"第 {attempt} 次"
    )

    raise TemporaryServiceError(
        "主服务暂时不可用。"
    )


def secondary_service() -> str:
    """模拟正常工作的备用服务。"""

    COUNTERS[
        "fallback_secondary"
    ] += 1

    attempt = COUNTERS[
        "fallback_secondary"
    ]

    print(
        "备用服务实际执行："
        f"第 {attempt} 次"
    )

    return "备用服务返回结果"


def primary_then_fallback_node(
    state: FallbackState,
    runtime: Runtime,
) -> dict[str, str]:
    """第一次调用主服务，重试时调用备用服务。

    runtime.execution_info.node_attempt：

    1：
        当前节点第一次执行。

    2：
        当前节点第一次重试。

    3：
        当前节点第二次重试。
    """

    node_attempt = (
        runtime
        .execution_info
        .node_attempt
    )

    print(
        "当前 LangGraph 节点尝试次数："
        f"{node_attempt}"
    )

    if node_attempt == 1:
        result = primary_service()

    else:
        result = secondary_service()

    return {
        "result": result
    }


def run_fallback_demo() -> None:
    """运行主服务失败后降级案例。"""

    print_title(
        "案例 4：主服务失败后降级到备用服务"
    )

    COUNTERS[
        "fallback_primary"
    ] = 0

    COUNTERS[
        "fallback_secondary"
    ] = 0

    builder = StateGraph(
        FallbackState
    )

    builder.add_node(
        "call_service",
        primary_then_fallback_node,
        retry_policy=RetryPolicy(
            max_attempts=3,
            initial_interval=0.2,
            backoff_factor=2.0,
            max_interval=1.0,
            jitter=False,
            retry_on=should_retry,
        ),
    )

    builder.add_edge(
        START,
        "call_service",
    )

    builder.add_edge(
        "call_service",
        END,
    )

    graph = builder.compile()

    result = graph.invoke(
        {
            "result": ""
        }
    )

    print()
    print(
        "最终结果："
        f"{result['result']}"
    )

    print(
        "主服务执行次数："
        f"{COUNTERS['fallback_primary']}"
    )

    print(
        "备用服务执行次数："
        f"{COUNTERS['fallback_secondary']}"
    )


# =============================================================================
# 程序入口
# =============================================================================


def main() -> None:
    """依次执行四个错误处理案例。"""

    run_transient_error_demo()

    run_permanent_error_demo()

    run_argument_error_demo()

    run_fallback_demo()

    print()
    print("=" * 80)
    print("全部案例执行完成")
    print("=" * 80)


if __name__ == "__main__":
    main()