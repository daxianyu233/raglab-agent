"""验证 LangGraph 节点 RetryPolicy。

前两次调用故意抛出暂时性异常，第三次成功。
"""

from __future__ import annotations

from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy


class RetryDemoState(TypedDict):
    """测试图状态。"""

    result: str


class TemporaryServiceError(Exception):
    """模拟可能通过重试恢复的外部服务错误。"""


ATTEMPT_COUNTER = {"count": 0}


def call_unstable_service(state: RetryDemoState) -> dict[str, str]:
    """前两次失败，第三次成功。"""

    del state

    ATTEMPT_COUNTER["count"] += 1
    current_attempt = ATTEMPT_COUNTER["count"]

    print(f"节点执行次数：{current_attempt}")

    if current_attempt < 3:
        raise TemporaryServiceError("模拟临时服务不可用")

    return {"result": "第三次调用成功"}


def should_retry(error: Exception) -> bool:
    """只重试模拟的暂时性服务错误。"""

    return isinstance(error, TemporaryServiceError)


def main() -> None:
    """构建并运行测试图。"""

    builder = StateGraph(RetryDemoState)

    builder.add_node(
        "unstable_service",
        call_unstable_service,
        retry_policy=RetryPolicy(
            max_attempts=3,
            initial_interval=0.2,
            backoff_factor=2.0,
            max_interval=1.0,
            jitter=False,
            retry_on=should_retry,
        ),
    )

    builder.add_edge(START, "unstable_service")
    builder.add_edge("unstable_service", END)

    graph = builder.compile()
    result = graph.invoke({"result": ""})

    print()
    print(f"最终结果：{result['result']}")


if __name__ == "__main__":
    main()
