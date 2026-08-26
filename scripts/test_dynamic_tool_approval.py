"""LangGraph 工具风险分级与动态审批演示。

本案例模拟以下过程：

1. LLM 生成工具调用；
2. policy_gate 节点检查工具风险；
3. 只读工具直接交给 ToolNode 执行；
4. 高风险工具通过 interrupt 等待审批；
5. 批准后执行，拒绝后返回 ToolMessage，但不执行工具。

为了保证结果稳定，本脚本不调用真实 LLM，
而是手动构造 AIMessage.tool_calls。

运行：

    python -m scripts.test_dynamic_tool_approval
"""

from __future__ import annotations

from pprint import pformat
from typing import Any, Literal

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import (
    END,
    START,
    MessagesState,
    StateGraph,
)
from langgraph.prebuilt import ToolNode
from langgraph.types import (
    Command,
    interrupt,
)


# =============================================================================
# 工具执行计数器
# =============================================================================


EXECUTION_COUNTERS: dict[str, int] = {
    "read_customer_profile": 0,
    "delete_old_records": 0,
}


# =============================================================================
# 工具定义
# =============================================================================


@tool
def read_customer_profile(
    customer_id: str,
) -> str:
    """读取客户资料，不修改任何外部数据。"""

    EXECUTION_COUNTERS[
        "read_customer_profile"
    ] += 1

    execution_count = EXECUTION_COUNTERS[
        "read_customer_profile"
    ]

    print()
    print(
        "[read_customer_profile] "
        "只读工具开始执行"
    )

    print(
        "[read_customer_profile] "
        f"实际执行次数：{execution_count}"
    )

    return (
        f"客户 {customer_id} 的资料："
        "会员等级为普通会员，"
        "账户状态正常。"
    )


@tool
def delete_old_records(
    table_name: str,
    days: int,
) -> str:
    """删除指定数据表中的过期记录。

    本案例只进行模拟，不会连接真实数据库。
    """

    EXECUTION_COUNTERS[
        "delete_old_records"
    ] += 1

    execution_count = EXECUTION_COUNTERS[
        "delete_old_records"
    ]

    print()
    print(
        "[delete_old_records] "
        "高风险工具开始执行"
    )

    print(
        "[delete_old_records] "
        f"实际执行次数：{execution_count}"
    )

    return (
        "模拟删除完成："
        f"已删除 {table_name} 表中"
        f"超过 {days} 天的记录。"
    )


TOOLS = [
    read_customer_profile,
    delete_old_records,
]


# =============================================================================
# 工具风险策略
# =============================================================================


TOOL_RISK_POLICIES: dict[
    str,
    dict[str, Any],
] = {
    "read_customer_profile": {
        "risk_level": "safe",
        "requires_approval": False,
        "description": (
            "只读取客户信息，"
            "不会修改外部数据。"
        ),
    },
    "delete_old_records": {
        "risk_level": "high",
        "requires_approval": True,
        "description": (
            "会删除数据库记录，"
            "属于有副作用且可能不可逆的操作。"
        ),
    },
}


# =============================================================================
# 通用辅助函数
# =============================================================================


def print_title(
    title: str,
) -> None:
    """打印案例标题。"""

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def build_tool_call_message(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    call_id: str,
) -> AIMessage:
    """构造模拟 LLM 生成的工具调用。

    真实 Agent 中，这条 AIMessage 通常由：

        model.bind_tools(tools).invoke(...)

    自动生成。
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


def get_latest_tool_call(
    messages: list[BaseMessage],
) -> dict[str, Any]:
    """读取最后一条 AIMessage 中的工具调用。"""

    if not messages:
        raise ValueError(
            "当前 State 中没有消息。"
        )

    latest_message = messages[-1]

    if not isinstance(
        latest_message,
        AIMessage,
    ):
        raise TypeError(
            "最后一条消息必须是 AIMessage，"
            f"实际类型：{type(latest_message)!r}"
        )

    tool_calls = list(
        latest_message.tool_calls
    )

    if not tool_calls:
        raise ValueError(
            "最后一条 AIMessage 中没有工具调用。"
        )

    if len(tool_calls) != 1:
        raise ValueError(
            "当前演示每次只允许一个工具调用，"
            f"实际数量：{len(tool_calls)}"
        )

    return tool_calls[0]


def read_approval_input() -> bool:
    """读取用户的 approve 或 reject。"""

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
            "\n请输入 approve 或 reject："
        ).strip().lower()

        if raw_value in approve_values:
            return True

        if raw_value in reject_values:
            return False

        print(
            "输入无效，请输入 approve 或 reject。"
        )


def print_interrupts(
    result: dict[str, Any],
) -> None:
    """打印 graph.invoke() 返回的中断。"""

    interrupts = result.get(
        "__interrupt__",
        [],
    )

    if not interrupts:
        print()
        print("本次执行没有产生中断。")
        return

    print()
    print(
        "图已暂停，"
        f"中断数量：{len(interrupts)}"
    )

    for index, current_interrupt in enumerate(
        interrupts,
        start=1,
    ):
        print()
        print(f"中断 {index}")
        print(
            "interrupt_id："
            f"{current_interrupt.id}"
        )

        print("interrupt_value：")

        print(
            pformat(
                current_interrupt.value,
                width=100,
                sort_dicts=False,
            )
        )


def print_final_messages(
    result: dict[str, Any],
) -> None:
    """打印最终消息列表。"""

    messages = list(
        result.get(
            "messages",
            [],
        )
    )

    print()
    print("-" * 80)
    print("最终消息列表")
    print("-" * 80)

    for index, message in enumerate(
        messages,
        start=1,
    ):
        print()
        print(
            f"消息 {index}："
            f"{type(message).__name__}"
        )

        if isinstance(
            message,
            AIMessage,
        ):
            print(
                "tool_calls："
                f"{pformat(message.tool_calls, sort_dicts=False)}"
            )

            if message.content:
                print(
                    "content："
                    f"{message.content}"
                )

        elif isinstance(
            message,
            ToolMessage,
        ):
            print(
                "tool_call_id："
                f"{message.tool_call_id}"
            )

            print(
                "name："
                f"{message.name}"
            )

            print(
                "status："
                f"{message.status}"
            )

            print(
                "content："
                f"{message.content}"
            )

        else:
            print(
                "content："
                f"{getattr(message, 'content', '')}"
            )


def print_checkpoint(
    graph: Any,
    config: dict[str, Any],
    *,
    title: str,
) -> None:
    """打印当前线程最新检查点。"""

    snapshot = graph.get_state(
        config
    )

    print()
    print("-" * 80)
    print(title)
    print("-" * 80)

    print(
        "next："
        f"{snapshot.next}"
    )

    print(
        "interrupts："
        f"{len(snapshot.interrupts)}"
    )

    print(
        "messages 数量："
        f"{len(snapshot.values.get('messages', []))}"
    )

    print(
        "checkpoint config："
    )

    print(
        pformat(
            snapshot.config,
            width=100,
            sort_dicts=False,
        )
    )


# =============================================================================
# 风险策略节点
# =============================================================================


def policy_gate_node(
    state: MessagesState,
) -> Command[
    Literal[
        "execute_tool",
        "finish",
    ]
]:
    """根据工具风险决定直接执行还是暂停审批。

    关键职责：

    1. 读取 LLM 请求调用的工具；
    2. 查询系统定义的工具风险策略；
    3. 低风险工具直接放行；
    4. 高风险工具调用 interrupt；
    5. 拒绝时创建错误 ToolMessage，
       让 Agent 知道该工具没有执行。
    """

    print()
    print(
        "[policy_gate] "
        "开始检查工具风险"
    )

    messages = list(
        state.get(
            "messages",
            [],
        )
    )

    tool_call = get_latest_tool_call(
        messages
    )

    tool_name = str(
        tool_call["name"]
    )

    tool_arguments = dict(
        tool_call.get(
            "args",
            {},
        )
    )

    tool_call_id = str(
        tool_call["id"]
    )

    print(
        "[policy_gate] "
        f"模型请求调用工具：{tool_name}"
    )

    print(
        "[policy_gate] "
        f"工具参数：{tool_arguments}"
    )

    policy = TOOL_RISK_POLICIES.get(
        tool_name
    )

    # 未登记的工具采用“默认拒绝自动放行”策略。
    if policy is None:
        policy = {
            "risk_level": "unknown",
            "requires_approval": True,
            "description": (
                "该工具尚未登记风险策略，"
                "默认要求人工审批。"
            ),
        }

    risk_level = str(
        policy["risk_level"]
    )

    requires_approval = bool(
        policy["requires_approval"]
    )

    description = str(
        policy["description"]
    )

    print(
        "[policy_gate] "
        f"风险等级：{risk_level}"
    )

    print(
        "[policy_gate] "
        f"是否需要审批：{requires_approval}"
    )

    if not requires_approval:
        print(
            "[policy_gate] "
            "低风险工具，直接放行"
        )

        return Command(
            goto="execute_tool"
        )

    print(
        "[policy_gate] "
        "高风险工具，即将暂停等待审批"
    )

    # -------------------------------------------------------------------------
    # 第一次运行到这里：
    #
    #     图暂停并保存 State。
    #
    # 恢复时：
    #
    #     Command(resume=True)
    #         decision 得到 True。
    #
    #     Command(resume=False)
    #         decision 得到 False。
    # -------------------------------------------------------------------------
    decision = interrupt(
        {
            "type": "tool_approval",
            "tool_name": tool_name,
            "tool_arguments": tool_arguments,
            "risk_level": risk_level,
            "risk_description": description,
            "question": (
                "是否批准执行该工具调用？"
            ),
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
            "[policy_gate] "
            "用户已经批准工具调用"
        )

        return Command(
            goto="execute_tool"
        )

    print(
        "[policy_gate] "
        "用户拒绝了工具调用"
    )

    rejected_message = ToolMessage(
        content=(
            "工具调用被用户拒绝，"
            "没有执行任何外部操作。"
        ),
        tool_call_id=tool_call_id,
        name=tool_name,
        status="error",
    )

    return Command(
        update={
            "messages": [
                rejected_message
            ]
        },
        goto="finish",
    )


# =============================================================================
# 完成节点
# =============================================================================


def finish_node(
    state: MessagesState,
) -> dict[str, Any]:
    """拒绝工具调用后结束流程。"""

    print()
    print(
        "[finish] "
        "流程结束，工具未执行"
    )

    return {}


# =============================================================================
# 构建图
# =============================================================================


def build_graph() -> Any:
    """构建风险感知工具执行图。"""

    builder = StateGraph(
        MessagesState
    )

    builder.add_node(
        "policy_gate",
        policy_gate_node,
    )

    # ToolNode 根据 AIMessage.tool_calls
    # 找到并执行对应 Python 工具。
    builder.add_node(
        "execute_tool",
        ToolNode(
            TOOLS
        ),
    )

    builder.add_node(
        "finish",
        finish_node,
    )

    builder.add_edge(
        START,
        "policy_gate",
    )

    builder.add_edge(
        "execute_tool",
        END,
    )

    builder.add_edge(
        "finish",
        END,
    )

    checkpointer = InMemorySaver()

    return builder.compile(
        checkpointer=checkpointer
    )


# =============================================================================
# 运行单个案例
# =============================================================================


def run_case(
    *,
    title: str,
    thread_id: str,
    tool_message: AIMessage,
) -> None:
    """执行一次工具调用，并处理可能出现的审批中断。"""

    print_title(
        title
    )

    graph = build_graph()

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = graph.invoke(
        {
            "messages": [
                tool_message
            ]
        },
        config=config,
    )

    interrupts = result.get(
        "__interrupt__",
        [],
    )

    if interrupts:
        print_interrupts(
            result
        )

        print_checkpoint(
            graph,
            config,
            title="中断后的检查点",
        )

        print()
        print(
            "现在需要你决定是否批准。"
        )

        approved = read_approval_input()

        print()
        print(
            "准备恢复："
            f"Command(resume={approved})"
        )

        result = graph.invoke(
            Command(
                resume=approved
            ),
            config=config,
        )

    print_final_messages(
        result
    )

    print_checkpoint(
        graph,
        config,
        title="流程结束后的检查点",
    )


# =============================================================================
# 三个演示案例
# =============================================================================


def run_safe_tool_demo() -> None:
    """低风险只读工具，无需人工审批。"""

    EXECUTION_COUNTERS[
        "read_customer_profile"
    ] = 0

    run_case(
        title=(
            "案例 1：只读工具直接执行"
        ),
        thread_id=(
            "risk-demo-safe-read"
        ),
        tool_message=(
            build_tool_call_message(
                tool_name=(
                    "read_customer_profile"
                ),
                arguments={
                    "customer_id": (
                        "CUSTOMER-001"
                    )
                },
                call_id=(
                    "call-safe-001"
                ),
            )
        ),
    )

    print()
    print(
        "只读工具最终执行次数："
        f"{EXECUTION_COUNTERS['read_customer_profile']}"
    )


def run_approved_tool_demo() -> None:
    """高风险工具，用户批准后执行。"""

    EXECUTION_COUNTERS[
        "delete_old_records"
    ] = 0

    print()
    print(
        "本案例建议输入：approve"
    )

    run_case(
        title=(
            "案例 2：高风险工具审批通过"
        ),
        thread_id=(
            "risk-demo-delete-approved"
        ),
        tool_message=(
            build_tool_call_message(
                tool_name=(
                    "delete_old_records"
                ),
                arguments={
                    "table_name": (
                        "application_logs"
                    ),
                    "days": 30,
                },
                call_id=(
                    "call-high-risk-001"
                ),
            )
        ),
    )

    print()
    print(
        "删除工具最终执行次数："
        f"{EXECUTION_COUNTERS['delete_old_records']}"
    )


def run_rejected_tool_demo() -> None:
    """高风险工具，用户拒绝后不执行。"""

    EXECUTION_COUNTERS[
        "delete_old_records"
    ] = 0

    print()
    print(
        "本案例建议输入：reject"
    )

    run_case(
        title=(
            "案例 3：高风险工具审批拒绝"
        ),
        thread_id=(
            "risk-demo-delete-rejected"
        ),
        tool_message=(
            build_tool_call_message(
                tool_name=(
                    "delete_old_records"
                ),
                arguments={
                    "table_name": (
                        "production_users"
                    ),
                    "days": 0,
                },
                call_id=(
                    "call-high-risk-002"
                ),
            )
        ),
    )

    print()
    print(
        "删除工具最终执行次数："
        f"{EXECUTION_COUNTERS['delete_old_records']}"
    )


# =============================================================================
# 程序入口
# =============================================================================


def main() -> None:
    """运行全部风险分级案例。"""

    print()
    print("=" * 80)
    print("LangGraph 工具风险分级与动态审批")
    print("=" * 80)

    print()
    print("本程序需要你输入两次：")
    print("1. 案例2输入 approve")
    print("2. 案例3输入 reject")

    run_safe_tool_demo()

    run_approved_tool_demo()

    run_rejected_tool_demo()

    print()
    print("=" * 80)
    print("全部工具风险案例执行完成")
    print("=" * 80)


if __name__ == "__main__":
    main()