"""LangGraph 外部操作幂等性演示。

本案例模拟：

1. Agent 使用稳定的 operation_id 请求创建订单；
2. 外部 SQLite 数据库成功创建订单；
3. 外部服务在返回响应前模拟网络中断；
4. LangGraph RetryPolicy 重新执行节点；
5. 节点先根据 operation_id 查询外部状态；
6. 发现订单已经创建，直接返回原结果；
7. 外部数据库中不会产生重复订单。

运行：

    python -m scripts.test_idempotent_external_operation
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from pprint import pformat
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy
from typing_extensions import TypedDict


# =============================================================================
# 路径和全局测试状态
# =============================================================================


DATABASE_PATH = Path(
    "storage/idempotency_demo.sqlite3"
)

NODE_EXECUTION_COUNTER = {
    "count": 0
}

# 用来保证每个 operation_id
# 只模拟一次“响应丢失”。
SIMULATED_RESPONSE_LOSSES: set[str] = set()


# =============================================================================
# 异常定义
# =============================================================================


class TemporaryExternalServiceError(
    Exception
):
    """模拟外部服务暂时性故障。

    例如：

    - 网络超时；
    - 响应丢失；
    - 临时连接失败；
    - 服务端暂时不可用。
    """


# =============================================================================
# LangGraph State
# =============================================================================


class OrderAgentState(
    TypedDict
):
    """订单 Agent 的内部图状态。"""

    operation_id: str
    product: str
    quantity: int

    operation_status: str
    external_order_id: str
    result: str


# =============================================================================
# 通用辅助函数
# =============================================================================


def print_title(
    title: str,
) -> None:
    """打印标题。"""

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def utc_now() -> str:
    """返回当前 UTC 时间字符串。"""

    return datetime.now(
        timezone.utc
    ).isoformat()


# =============================================================================
# 模拟外部订单系统
# =============================================================================


def initialize_external_database(
    *,
    reset: bool,
) -> None:
    """初始化模拟外部系统数据库。

    reset=True 时删除旧测试数据库，
    保证每次运行结果一致。
    """

    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if reset and DATABASE_PATH.exists():
        DATABASE_PATH.unlink()

    with sqlite3.connect(
        DATABASE_PATH
    ) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS operations (
                operation_id TEXT PRIMARY KEY,
                action_name TEXT NOT NULL,
                status TEXT NOT NULL,
                external_resource_id TEXT,
                request_json TEXT NOT NULL,
                result_json TEXT,
                last_error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,

                operation_id TEXT NOT NULL UNIQUE,

                product TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                created_at TEXT NOT NULL,

                FOREIGN KEY(operation_id)
                    REFERENCES operations(operation_id)
            )
            """
        )

        connection.commit()


def get_external_operation(
    operation_id: str,
) -> dict[str, Any] | None:
    """根据 operation_id 查询外部操作状态。"""

    with sqlite3.connect(
        DATABASE_PATH
    ) as connection:
        connection.row_factory = (
            sqlite3.Row
        )

        row = connection.execute(
            """
            SELECT
                operation_id,
                action_name,
                status,
                external_resource_id,
                request_json,
                result_json,
                last_error,
                created_at,
                updated_at
            FROM operations
            WHERE operation_id = ?
            """,
            (
                operation_id,
            ),
        ).fetchone()

    if row is None:
        return None

    return dict(
        row
    )


def parse_external_result(
    operation: dict[str, Any],
) -> dict[str, Any]:
    """解析外部操作保存的结果。"""

    raw_result = operation.get(
        "result_json"
    )

    if not raw_result:
        raise RuntimeError(
            "外部操作已经成功，"
            "但没有保存 result_json。"
        )

    parsed_result = json.loads(
        str(raw_result)
    )

    if not isinstance(
        parsed_result,
        dict,
    ):
        raise TypeError(
            "外部 result_json "
            "解析后不是字典。"
        )

    return parsed_result


def create_order_in_external_service(
    *,
    operation_id: str,
    product: str,
    quantity: int,
) -> dict[str, Any]:
    """在模拟外部服务中幂等创建订单。

    关键保证：

    operations.operation_id 是主键；

    orders.operation_id 有 UNIQUE 约束。

    因此同一个 operation_id
    不能创建两个订单。
    """

    print()
    print(
        "[外部服务] "
        "收到创建订单请求"
    )

    print(
        "[外部服务] "
        f"operation_id={operation_id}"
    )

    # -------------------------------------------------------------------------
    # 第一层幂等检查：
    #
    # 外部服务自己先查询 operation_id。
    # 即使调用方没有先查，外部也不能盲目重复执行。
    # -------------------------------------------------------------------------
    existing_operation = (
        get_external_operation(
            operation_id
        )
    )

    if existing_operation is not None:
        existing_status = str(
            existing_operation["status"]
        )

        print(
            "[外部服务] "
            "发现已有 operation_id，"
            f"status={existing_status}"
        )

        if existing_status == "succeeded":
            return parse_external_result(
                existing_operation
            )

        if existing_status == "running":
            raise TemporaryExternalServiceError(
                "相同 operation_id "
                "对应的操作仍在执行中。"
            )

        raise RuntimeError(
            "相同 operation_id "
            f"已经处于不可继续状态："
            f"{existing_status}"
        )

    request_data = {
        "product": product,
        "quantity": quantity,
    }

    now = utc_now()

    # 模拟外部系统生成的订单 ID。
    order_id = (
        "ORDER-"
        f"{uuid.uuid4().hex[:10].upper()}"
    )

    result_data = {
        "operation_id": operation_id,
        "order_id": order_id,
        "product": product,
        "quantity": quantity,
        "status": "created",
    }

    try:
        with sqlite3.connect(
            DATABASE_PATH
        ) as connection:
            # ---------------------------------------------------------------
            # 使用同一个 SQLite 事务完成：
            #
            # 1. 写入操作记录；
            # 2. 创建订单；
            # 3. 将操作标记为 succeeded。
            #
            # 中途任何 SQL 失败，
            # 整个事务都会回滚。
            # ---------------------------------------------------------------
            connection.execute(
                """
                INSERT INTO operations (
                    operation_id,
                    action_name,
                    status,
                    external_resource_id,
                    request_json,
                    result_json,
                    last_error,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    operation_id,
                    "create_order",
                    "running",
                    None,
                    json.dumps(
                        request_data,
                        ensure_ascii=False,
                    ),
                    None,
                    None,
                    now,
                    now,
                ),
            )

            connection.execute(
                """
                INSERT INTO orders (
                    order_id,
                    operation_id,
                    product,
                    quantity,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    order_id,
                    operation_id,
                    product,
                    quantity,
                    now,
                ),
            )

            connection.execute(
                """
                UPDATE operations
                SET
                    status = ?,
                    external_resource_id = ?,
                    result_json = ?,
                    updated_at = ?
                WHERE operation_id = ?
                """,
                (
                    "succeeded",
                    order_id,
                    json.dumps(
                        result_data,
                        ensure_ascii=False,
                    ),
                    utc_now(),
                    operation_id,
                ),
            )

            connection.commit()

    except sqlite3.IntegrityError:
        # 如果出现并发请求，
        # 另一个请求可能已经使用相同 operation_id
        # 完成创建。
        existing_operation = (
            get_external_operation(
                operation_id
            )
        )

        if (
            existing_operation is not None
            and existing_operation["status"]
            == "succeeded"
        ):
            return parse_external_result(
                existing_operation
            )

        raise

    print(
        "[外部服务] "
        "订单已经真实写入数据库"
    )

    print(
        "[外部服务] "
        f"order_id={order_id}"
    )

    # -------------------------------------------------------------------------
    # 模拟最棘手的情况：
    #
    # 外部数据库事务已经成功提交，
    # 但响应返回给 Agent 之前网络断开。
    #
    # 此时外部已经成功，
    # Agent 却只看到了异常。
    # -------------------------------------------------------------------------
    if (
        operation_id
        not in SIMULATED_RESPONSE_LOSSES
    ):
        SIMULATED_RESPONSE_LOSSES.add(
            operation_id
        )

        print(
            "[外部服务] "
            "模拟响应传输过程中网络中断"
        )

        raise TemporaryExternalServiceError(
            "订单已经创建，"
            "但返回响应时网络连接中断。"
        )

    return result_data


# =============================================================================
# LangGraph 节点
# =============================================================================


def create_order_node(
    state: OrderAgentState,
) -> dict[str, str]:
    """创建订单节点。

    每次执行节点时：

    1. 先按 operation_id 查询外部系统；
    2. 如果已经成功，直接读取原结果；
    3. 如果不存在，才发起创建请求。
    """

    NODE_EXECUTION_COUNTER[
        "count"
    ] += 1

    current_attempt = (
        NODE_EXECUTION_COUNTER[
            "count"
        ]
    )

    operation_id = state[
        "operation_id"
    ]

    product = state[
        "product"
    ]

    quantity = state[
        "quantity"
    ]

    print()
    print(
        "[Agent 节点] "
        f"第 {current_attempt} 次执行"
    )

    print(
        "[Agent 节点] "
        "先查询外部操作状态"
    )

    print(
        "[Agent 节点] "
        f"operation_id={operation_id}"
    )

    external_operation = (
        get_external_operation(
            operation_id
        )
    )

    if external_operation is not None:
        external_status = str(
            external_operation["status"]
        )

        print(
            "[Agent 节点] "
            "外部操作已经存在，"
            f"status={external_status}"
        )

        if external_status == "succeeded":
            external_result = (
                parse_external_result(
                    external_operation
                )
            )

            existing_order_id = str(
                external_result[
                    "order_id"
                ]
            )

            print(
                "[Agent 节点] "
                "读取原有订单，"
                "不再调用创建接口"
            )

            return {
                "operation_status": (
                    "succeeded"
                ),
                "external_order_id": (
                    existing_order_id
                ),
                "result": (
                    "读取到已完成的幂等操作："
                    f"订单 {existing_order_id} "
                    "已经创建成功。"
                ),
            }

        if external_status == "running":
            raise TemporaryExternalServiceError(
                "外部操作仍在执行中，"
                "稍后重新查询。"
            )

        raise RuntimeError(
            "外部操作已经失败，"
            f"status={external_status}"
        )

    print(
        "[Agent 节点] "
        "外部没有找到该 operation_id"
    )

    print(
        "[Agent 节点] "
        "开始调用外部创建订单接口"
    )

    external_result = (
        create_order_in_external_service(
            operation_id=operation_id,
            product=product,
            quantity=quantity,
        )
    )

    created_order_id = str(
        external_result[
            "order_id"
        ]
    )

    return {
        "operation_status": "succeeded",
        "external_order_id": (
            created_order_id
        ),
        "result": (
            "订单创建成功："
            f"{created_order_id}"
        ),
    }


# =============================================================================
# RetryPolicy 判断函数
# =============================================================================


def should_retry_external_error(
    error: Exception,
) -> bool:
    """只重试暂时性外部错误。"""

    retryable = isinstance(
        error,
        TemporaryExternalServiceError,
    )

    decision = (
        "重试"
        if retryable
        else "不重试"
    )

    print()
    print(
        "[RetryPolicy] "
        f"{type(error).__name__}"
        f" -> {decision}"
    )

    print(
        "[RetryPolicy] "
        f"错误信息：{error}"
    )

    return retryable


# =============================================================================
# 构建 LangGraph
# =============================================================================


def build_graph() -> Any:
    """构建订单 Agent 图。"""

    builder = StateGraph(
        OrderAgentState
    )

    builder.add_node(
        "create_order",
        create_order_node,
        retry_policy=RetryPolicy(
            # 包括第一次执行，
            # 最多尝试三次。
            max_attempts=3,

            initial_interval=0.2,
            backoff_factor=2.0,
            max_interval=1.0,
            jitter=False,

            retry_on=(
                should_retry_external_error
            ),
        ),
    )

    builder.add_edge(
        START,
        "create_order",
    )

    builder.add_edge(
        "create_order",
        END,
    )

    checkpointer = InMemorySaver()

    return builder.compile(
        checkpointer=checkpointer
    )


# =============================================================================
# 数据库查看函数
# =============================================================================


def print_external_database() -> None:
    """打印模拟外部系统中的所有记录。"""

    print()
    print("-" * 80)
    print("外部 operations 表")
    print("-" * 80)

    with sqlite3.connect(
        DATABASE_PATH
    ) as connection:
        connection.row_factory = (
            sqlite3.Row
        )

        operations = connection.execute(
            """
            SELECT
                operation_id,
                status,
                external_resource_id,
                last_error
            FROM operations
            ORDER BY created_at
            """
        ).fetchall()

        orders = connection.execute(
            """
            SELECT
                order_id,
                operation_id,
                product,
                quantity
            FROM orders
            ORDER BY created_at
            """
        ).fetchall()

    for row in operations:
        print(
            pformat(
                dict(row),
                width=100,
                sort_dicts=False,
            )
        )

    print()
    print("-" * 80)
    print("外部 orders 表")
    print("-" * 80)

    for row in orders:
        print(
            pformat(
                dict(row),
                width=100,
                sort_dicts=False,
            )
        )

    print()
    print(
        "operations 记录数："
        f"{len(operations)}"
    )

    print(
        "orders 记录数："
        f"{len(orders)}"
    )


# =============================================================================
# 演示案例
# =============================================================================


def run_first_request(
    graph: Any,
) -> None:
    """第一次请求：外部成功，但响应丢失。"""

    print_title(
        "案例 1：外部成功但响应丢失，Agent 自动重试"
    )

    NODE_EXECUTION_COUNTER[
        "count"
    ] = 0

    operation_id = (
        "thread-001:"
        "create-order:"
        "operation-001"
    )

    result = graph.invoke(
        {
            "operation_id": (
                operation_id
            ),
            "product": "笔记本电脑",
            "quantity": 1,
            "operation_status": (
                "pending"
            ),
            "external_order_id": "",
            "result": "",
        },
        config={
            "configurable": {
                "thread_id": (
                    "idempotency-demo-1"
                )
            }
        },
    )

    print()
    print("Agent 最终 State：")

    print(
        pformat(
            result,
            width=100,
            sort_dicts=False,
        )
    )

    print()
    print(
        "Agent 节点实际执行次数："
        f"{NODE_EXECUTION_COUNTER['count']}"
    )

    print_external_database()


def run_same_operation_again(
    graph: Any,
) -> None:
    """再次提交相同 operation_id。"""

    print_title(
        "案例 2：再次提交相同 operation_id"
    )

    NODE_EXECUTION_COUNTER[
        "count"
    ] = 0

    result = graph.invoke(
        {
            "operation_id": (
                "thread-001:"
                "create-order:"
                "operation-001"
            ),
            "product": "笔记本电脑",
            "quantity": 1,
            "operation_status": (
                "pending"
            ),
            "external_order_id": "",
            "result": "",
        },
        config={
            "configurable": {
                "thread_id": (
                    "idempotency-demo-2"
                )
            }
        },
    )

    print()
    print("Agent 最终 State：")

    print(
        pformat(
            result,
            width=100,
            sort_dicts=False,
        )
    )

    print()
    print(
        "Agent 节点实际执行次数："
        f"{NODE_EXECUTION_COUNTER['count']}"
    )

    print_external_database()


def run_new_operation(
    graph: Any,
) -> None:
    """使用新的 operation_id 创建新订单。"""

    print_title(
        "案例 3：新的 operation_id 可以创建新订单"
    )

    NODE_EXECUTION_COUNTER[
        "count"
    ] = 0

    result = graph.invoke(
        {
            "operation_id": (
                "thread-001:"
                "create-order:"
                "operation-002"
            ),
            "product": "无线鼠标",
            "quantity": 2,
            "operation_status": (
                "pending"
            ),
            "external_order_id": "",
            "result": "",
        },
        config={
            "configurable": {
                "thread_id": (
                    "idempotency-demo-3"
                )
            }
        },
    )

    print()
    print("Agent 最终 State：")

    print(
        pformat(
            result,
            width=100,
            sort_dicts=False,
        )
    )

    print()
    print(
        "Agent 节点实际执行次数："
        f"{NODE_EXECUTION_COUNTER['count']}"
    )

    print_external_database()


# =============================================================================
# 程序入口
# =============================================================================


def main() -> None:
    """运行所有幂等案例。"""

    print()
    print("=" * 80)
    print("LangGraph 外部操作幂等性演示")
    print("=" * 80)

    initialize_external_database(
        reset=True
    )

    SIMULATED_RESPONSE_LOSSES.clear()

    graph = build_graph()

    run_first_request(
        graph
    )

    run_same_operation_again(
        graph
    )

    run_new_operation(
        graph
    )

    print()
    print("=" * 80)
    print("全部幂等操作案例执行完成")
    print("=" * 80)


if __name__ == "__main__":
    main()