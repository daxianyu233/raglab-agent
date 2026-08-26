"""LangGraph interrupt 交互式中断与恢复演示。

本脚本包含三个交互案例：

1. 缺少必要信息：
   图暂停并要求用户输入订单号。

2. 高风险操作审批通过：
   图暂停，用户输入 approve 后执行操作。

3. 高风险操作审批拒绝：
   图暂停，用户输入 reject 后取消操作。

同时通过 graph.get_state(config) 查看：

- 当前图状态；
- 下一步准备执行的节点；
- 尚未解决的 interrupt；
- Checkpointer 中保存的状态值。

运行：

    python -m scripts.test_langgraph_interrupt
"""

from __future__ import annotations

from pprint import pformat
from typing import Any, Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict


# =============================================================================
# 通用显示与输入函数
# =============================================================================


def print_title(
    title: str,
) -> None:
    """打印案例标题。"""

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_interrupt_result(
    result: dict[str, Any],
) -> None:
    """打印 graph.invoke() 返回的中断信息。"""

    interrupts = result.get(
        "__interrupt__",
        [],
    )

    if not interrupts:
        print()
        print("本次图执行没有产生中断。")
        return

    print()
    print(
        "图已经暂停，"
        f"待处理的中断数量：{len(interrupts)}"
    )

    for index, current_interrupt in enumerate(
        interrupts,
        start=1,
    ):
        interrupt_id = getattr(
            current_interrupt,
            "id",
            "unknown",
        )

        interrupt_value = getattr(
            current_interrupt,
            "value",
            current_interrupt,
        )

        print()
        print("-" * 80)
        print(f"中断 {index}")
        print(f"interrupt_id：{interrupt_id}")
        print("interrupt_value：")

        print(
            pformat(
                interrupt_value,
                width=80,
                sort_dicts=False,
            )
        )


def print_checkpoint_snapshot(
    graph: Any,
    config: dict[str, Any],
    *,
    title: str,
) -> None:
    """从 Checkpointer 读取并打印当前线程状态。

    graph.get_state(config) 返回 StateSnapshot。

    重点字段：

    values:
        当前保存的图状态。

    next:
        下一步准备执行或恢复的节点。

    interrupts:
        当前尚未解决的中断。

    created_at:
        该检查点的创建时间。

    config:
        当前检查点对应的配置信息，
        其中包含 thread_id 和 checkpoint_id。
    """

    snapshot = graph.get_state(
        config
    )

    print()
    print("-" * 80)
    print(title)
    print("-" * 80)

    print(
        "检查点创建时间："
        f"{snapshot.created_at}"
    )

    print(
        "下一步节点 next："
        f"{snapshot.next}"
    )

    print(
        "未解决中断数量："
        f"{len(snapshot.interrupts)}"
    )

    if snapshot.interrupts:
        for index, current_interrupt in enumerate(
            snapshot.interrupts,
            start=1,
        ):
            print(
                f"  中断 {index}："
                f"id={current_interrupt.id}"
            )

            print(
                "  value="
                f"{pformat(current_interrupt.value, sort_dicts=False)}"
            )

    print()
    print("当前状态 values：")

    print(
        pformat(
            snapshot.values,
            width=100,
            sort_dicts=False,
        )
    )

    print()
    print("检查点 config：")

    print(
        pformat(
            snapshot.config,
            width=100,
            sort_dicts=False,
        )
    )

    print()
    print("待执行任务 tasks：")

    print(
        pformat(
            snapshot.tasks,
            width=100,
            sort_dicts=False,
        )
    )


def read_non_empty_input(
    prompt: str,
) -> str:
    """持续读取，直到用户输入非空字符串。"""

    while True:
        value = input(
            prompt
        ).strip()

        if value:
            return value

        print(
            "输入不能为空，请重新输入。"
        )


def read_approval_input(
    prompt: str,
) -> bool:
    """读取 approve 或 reject。

    返回：

    True:
        用户批准。

    False:
        用户拒绝。
    """

    approve_values = {
        "approve",
        "approved",
        "yes",
        "y",
        "true",
        "1",
        "批准",
        "同意",
        "是",
    }

    reject_values = {
        "reject",
        "rejected",
        "no",
        "n",
        "false",
        "0",
        "拒绝",
        "不同意",
        "否",
    }

    while True:
        raw_value = input(
            prompt
        ).strip().lower()

        if raw_value in approve_values:
            return True

        if raw_value in reject_values:
            return False

        print(
            "输入无效，请输入 approve 或 reject。"
        )


# =============================================================================
# 案例一：缺少必要信息
# =============================================================================


class OrderQueryState(
    TypedDict
):
    """订单查询图状态。"""

    request: str
    order_id: str
    result: str


def request_order_id_node(
    state: OrderQueryState,
) -> dict[str, str]:
    """缺少订单号时暂停并等待用户输入。

    第一次执行：

        order_id 为空
        → 调用 interrupt()
        → 图暂停

    恢复执行：

        节点从开头重新执行
        → 再次到达同一个 interrupt()
        → interrupt() 返回 Command(resume=...) 中的值
        → 节点继续向下执行
    """

    print()
    print(
        "[request_order_id_node] "
        "节点从开头开始执行"
    )

    order_id = str(
        state.get(
            "order_id",
            "",
        )
    ).strip()

    if not order_id:
        print(
            "[request_order_id_node] "
            "状态中没有订单号"
        )

        print(
            "[request_order_id_node] "
            "即将调用 interrupt()"
        )

        # ---------------------------------------------------------------------
        # 这里是真正的中断代码。
        #
        # 第一次运行到这里：
        #     图暂停，并把这个字典返回给外部程序。
        #
        # 使用 Command(resume=订单号) 恢复时：
        #     interrupt() 返回订单号字符串。
        # ---------------------------------------------------------------------
        order_id = interrupt(
            {
                "type": (
                    "missing_information"
                ),
                "field": "order_id",
                "question": (
                    "请提供需要查询的订单号。"
                ),
                "current_request": (
                    state["request"]
                ),
            }
        )

    normalized_order_id = str(
        order_id
    ).strip()

    if not normalized_order_id:
        raise ValueError(
            "订单号不能为空。"
        )

    print(
        "[request_order_id_node] "
        "已经获得订单号："
        f"{normalized_order_id}"
    )

    return {
        "order_id": normalized_order_id
    }


def query_order_node(
    state: OrderQueryState,
) -> dict[str, str]:
    """模拟查询订单。"""

    order_id = state[
        "order_id"
    ]

    print()
    print(
        "[query_order_node] "
        f"正在查询订单：{order_id}"
    )

    return {
        "result": (
            f"订单 {order_id} "
            "当前状态为：运输中。"
        )
    }


def build_order_query_graph() -> Any:
    """构建订单查询图。"""

    builder = StateGraph(
        OrderQueryState
    )

    builder.add_node(
        "request_order_id",
        request_order_id_node,
    )

    builder.add_node(
        "query_order",
        query_order_node,
    )

    builder.add_edge(
        START,
        "request_order_id",
    )

    builder.add_edge(
        "request_order_id",
        "query_order",
    )

    builder.add_edge(
        "query_order",
        END,
    )

    # -------------------------------------------------------------------------
    # 这里创建状态存储器。
    #
    # InMemorySaver 会保存：
    # - 当前状态 values；
    # - 当前执行位置；
    # - interrupt 信息；
    # - thread_id 对应的检查点。
    #
    # 当前只是内存版，程序退出后消失。
    # -------------------------------------------------------------------------
    checkpointer = InMemorySaver()

    # -------------------------------------------------------------------------
    # 这里把 Checkpointer 接入 LangGraph。
    #
    # 之后不需要手动调用 checkpointer.put()。
    # graph.invoke() 执行过程中，LangGraph 会自动写入检查点。
    # -------------------------------------------------------------------------
    graph = builder.compile(
        checkpointer=checkpointer
    )

    return graph


def run_missing_information_demo() -> None:
    """交互式演示缺少订单号。"""

    print_title(
        "案例 1：缺少订单号，暂停等待用户输入"
    )

    graph = build_order_query_graph()

    # -------------------------------------------------------------------------
    # thread_id 是读取和恢复检查点的关键。
    #
    # 首次运行和恢复运行必须使用同一个 thread_id。
    # -------------------------------------------------------------------------
    config = {
        "configurable": {
            "thread_id": (
                "interactive-order-query"
            )
        }
    }

    print()
    print("第一次执行图。")

    first_result = graph.invoke(
        {
            "request": (
                "请帮我查询订单状态"
            ),
            "order_id": "",
            "result": "",
        },
        config=config,
    )

    print_interrupt_result(
        first_result
    )

    # 中断后主动读取 Checkpointer，
    # 查看图保存了什么。
    print_checkpoint_snapshot(
        graph,
        config,
        title=(
            "案例1：中断后的检查点状态"
        ),
    )

    print()
    print(
        "现在轮到你输入订单号。"
    )

    print(
        "建议输入："
        "ORDER-2026-001"
    )

    order_id = read_non_empty_input(
        "\n请输入订单号："
    )

    print()
    print(
        "准备使用下面的恢复命令："
    )

    print(
        "Command("
        f"resume={order_id!r}"
        ")"
    )

    # -------------------------------------------------------------------------
    # 这里是真正的恢复代码。
    #
    # Command(resume=order_id)
    # 会把 order_id 作为 interrupt() 的返回值。
    #
    # config 必须还是之前的同一个 thread_id。
    # -------------------------------------------------------------------------
    resumed_result = graph.invoke(
        Command(
            resume=order_id
        ),
        config=config,
    )

    print()
    print(
        "最终订单号："
        f"{resumed_result['order_id']}"
    )

    print(
        "最终查询结果："
        f"{resumed_result['result']}"
    )

    print_checkpoint_snapshot(
        graph,
        config,
        title=(
            "案例1：恢复并执行完成后的检查点状态"
        ),
    )


# =============================================================================
# 案例二和三：高风险操作审批
# =============================================================================


class ApprovalState(
    TypedDict
):
    """高风险操作审批状态。"""

    action: str
    approved: bool | None
    result: str


EXECUTION_COUNTER = {
    "count": 0
}


def approval_node(
    state: ApprovalState,
) -> Command[
    Literal[
        "execute_action",
        "cancel_action",
    ]
]:
    """暂停图并等待用户审批。"""

    print()
    print(
        "[approval_node] "
        "节点从开头开始执行"
    )

    print(
        "[approval_node] "
        "即将调用 interrupt() 等待审批"
    )

    # -------------------------------------------------------------------------
    # 第一次运行：
    #     图暂停，返回审批信息。
    #
    # 恢复运行：
    #     Command(resume=True/False)
    #     会成为 decision 的值。
    # -------------------------------------------------------------------------
    decision = interrupt(
        {
            "type": "approval",
            "question": (
                "是否批准执行该操作？"
            ),
            "action": state[
                "action"
            ],
            "allowed_decisions": [
                "approve",
                "reject",
            ],
        }
    )

    approved = bool(
        decision
    )

    if approved:
        print(
            "[approval_node] "
            "用户批准了操作"
        )

        return Command(
            update={
                "approved": True
            },
            goto="execute_action",
        )

    print(
        "[approval_node] "
        "用户拒绝了操作"
    )

    return Command(
        update={
            "approved": False
        },
        goto="cancel_action",
    )


def execute_action_node(
    state: ApprovalState,
) -> dict[str, str]:
    """模拟真正有副作用的操作。"""

    EXECUTION_COUNTER[
        "count"
    ] += 1

    print()
    print(
        "[execute_action_node] "
        "正在执行高风险操作"
    )

    print(
        "[execute_action_node] "
        "当前实际执行次数："
        f"{EXECUTION_COUNTER['count']}"
    )

    return {
        "result": (
            "操作已经执行："
            f"{state['action']}"
        )
    }


def cancel_action_node(
    state: ApprovalState,
) -> dict[str, str]:
    """取消高风险操作。"""

    print()
    print(
        "[cancel_action_node] "
        "操作已取消，没有执行副作用"
    )

    return {
        "result": (
            "操作已被用户拒绝，"
            "没有实际执行："
            f"{state['action']}"
        )
    }


def build_approval_graph() -> Any:
    """构建高风险操作审批图。"""

    builder = StateGraph(
        ApprovalState
    )

    builder.add_node(
        "approval",
        approval_node,
    )

    builder.add_node(
        "execute_action",
        execute_action_node,
    )

    builder.add_node(
        "cancel_action",
        cancel_action_node,
    )

    builder.add_edge(
        START,
        "approval",
    )

    builder.add_edge(
        "execute_action",
        END,
    )

    builder.add_edge(
        "cancel_action",
        END,
    )

    # 状态保存器。
    checkpointer = InMemorySaver()

    # 将状态保存器接入图。
    graph = builder.compile(
        checkpointer=checkpointer
    )

    return graph


def run_approval_demo(
    *,
    demo_name: str,
    thread_id: str,
    action: str,
    suggested_input: str,
) -> None:
    """交互式运行一次审批流程。"""

    print_title(
        demo_name
    )

    EXECUTION_COUNTER[
        "count"
    ] = 0

    graph = build_approval_graph()

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    first_result = graph.invoke(
        {
            "action": action,
            "approved": None,
            "result": "",
        },
        config=config,
    )

    print_interrupt_result(
        first_result
    )

    print_checkpoint_snapshot(
        graph,
        config,
        title=(
            f"{demo_name}："
            "中断后的检查点状态"
        ),
    )

    print()
    print(
        "现在轮到你进行审批。"
    )

    print(
        f"建议输入：{suggested_input}"
    )

    approved = read_approval_input(
        "\n请输入 approve 或 reject："
    )

    print()
    print(
        "准备恢复图："
        f"Command(resume={approved})"
    )

    resumed_result = graph.invoke(
        Command(
            resume=approved
        ),
        config=config,
    )

    print()
    print(
        "approved："
        f"{resumed_result['approved']}"
    )

    print(
        "result："
        f"{resumed_result['result']}"
    )

    print(
        "高风险操作实际执行次数："
        f"{EXECUTION_COUNTER['count']}"
    )

    print_checkpoint_snapshot(
        graph,
        config,
        title=(
            f"{demo_name}："
            "恢复后的检查点状态"
        ),
    )


# =============================================================================
# 程序入口
# =============================================================================


def main() -> None:
    """依次运行三个交互案例。"""

    print()
    print("=" * 80)
    print("LangGraph interrupt 交互式演示")
    print("=" * 80)

    print()
    print("本程序需要你依次输入：")
    print("1. ORDER-2026-001")
    print("2. approve")
    print("3. reject")

    run_missing_information_demo()

    run_approval_demo(
        demo_name=(
            "案例 2：高风险操作审批通过"
        ),
        thread_id=(
            "interactive-approval-accepted"
        ),
        action=(
            "删除测试数据库中的"
            "30天前日志"
        ),
        suggested_input="approve",
    )

    run_approval_demo(
        demo_name=(
            "案例 3：高风险操作审批拒绝"
        ),
        thread_id=(
            "interactive-approval-rejected"
        ),
        action=(
            "删除生产数据库中的"
            "全部用户记录"
        ),
        suggested_input="reject",
    )

    print()
    print("=" * 80)
    print("全部交互式 interrupt 案例执行完成")
    print("=" * 80)


if __name__ == "__main__":
    main()