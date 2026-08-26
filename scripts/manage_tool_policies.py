"""Tool Policy Registry 管理 CLI。

示例：

查看全部：

    python -m scripts.manage_tool_policies list

统计：

    python -m scripts.manage_tool_policies stats

查看单个 Tool：

    python -m scripts.manage_tool_policies show send_email

给新 Tool 分类：

    python -m scripts.manage_tool_policies set send_email \
        --effect-type IRREVERSIBLE_WRITE \
        --external yes \
        --replay-policy REQUIRE_APPROVAL \
        --approval yes

禁用：

    python -m scripts.manage_tool_policies disable send_email

启用：

    python -m scripts.manage_tool_policies enable send_email

永久阻止：

    python -m scripts.manage_tool_policies block send_email
"""

from __future__ import annotations

import argparse
import json

from typing import Any

from raglab.control.tool_policy import (
    ReplayPolicy,
    ToolEffectType,
    ToolPolicyStatus,
)

from raglab.control.tool_policy_repository import (
    ToolPolicyRepository,
)


# ============================================================
# 参数工具
# ============================================================


def parse_yes_no(
    value: str,
) -> bool:
    """解析 yes / no。"""

    normalized = str(
        value
    ).strip().lower()

    if normalized in {
        "yes",
        "y",
        "true",
        "1",
        "on",
    }:
        return True

    if normalized in {
        "no",
        "n",
        "false",
        "0",
        "off",
    }:
        return False

    raise argparse.ArgumentTypeError(
        "必须填写 yes 或 no。"
    )


# ============================================================
# Parser
# ============================================================


def build_parser(
) -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "管理 RAGLab Tool Policy Registry。"
        )
    )

    subparsers = (
        parser.add_subparsers(
            dest="command",
            required=True,
        )
    )

    # --------------------------------------------------------
    # list
    # --------------------------------------------------------

    subparsers.add_parser(
        "list",
        help="查看全部 Tool Policy。",
    )

    # --------------------------------------------------------
    # pending
    # --------------------------------------------------------

    subparsers.add_parser(
        "pending",
        help="查看待分类 Tool。",
    )

    # --------------------------------------------------------
    # stats
    # --------------------------------------------------------

    subparsers.add_parser(
        "stats",
        help="查看 Tool Policy 统计。",
    )

    # --------------------------------------------------------
    # bootstrap
    # --------------------------------------------------------

    subparsers.add_parser(
        "bootstrap",
        help="补充当前项目初始 Tool Policy。",
    )

    # --------------------------------------------------------
    # show
    # --------------------------------------------------------

    show_parser = (
        subparsers.add_parser(
            "show",
            help="查看一个 Tool。",
        )
    )

    show_parser.add_argument(
        "tool_name",
    )

    # --------------------------------------------------------
    # set
    # --------------------------------------------------------

    set_parser = (
        subparsers.add_parser(
            "set",
            help="配置或重新分类 Tool。",
        )
    )

    set_parser.add_argument(
        "tool_name",
    )

    set_parser.add_argument(
        "--effect-type",
        required=True,
        choices=[
            item.value
            for item
            in ToolEffectType
        ],
    )

    set_parser.add_argument(
        "--external",
        required=True,
        type=parse_yes_no,
        help=(
            "是否修改 LangGraph State "
            "之外的状态：yes/no"
        ),
    )

    set_parser.add_argument(
        "--replay-policy",
        required=True,
        choices=[
            item.value
            for item
            in ReplayPolicy
        ],
    )

    set_parser.add_argument(
        "--approval",
        type=parse_yes_no,
        default=False,
        help="是否要求人工审批。",
    )

    set_parser.add_argument(
        "--enabled",
        type=parse_yes_no,
        default=True,
    )

    set_parser.add_argument(
        "--status",
        choices=[
            item.value
            for item
            in ToolPolicyStatus
        ],
        default=(
            ToolPolicyStatus.ACTIVE.value
        ),
    )

    set_parser.add_argument(
        "--source",
        default=None,
    )

    set_parser.add_argument(
        "--source-id",
        default=None,
    )

    set_parser.add_argument(
        "--idempotency-strategy",
        default=None,
    )

    set_parser.add_argument(
        "--compensation-tool",
        default=None,
    )

    set_parser.add_argument(
        "--description",
        default=None,
    )

    # --------------------------------------------------------
    # disable
    # --------------------------------------------------------

    disable_parser = (
        subparsers.add_parser(
            "disable",
            help="暂时禁用 Tool。",
        )
    )

    disable_parser.add_argument(
        "tool_name",
    )

    # --------------------------------------------------------
    # enable
    # --------------------------------------------------------

    enable_parser = (
        subparsers.add_parser(
            "enable",
            help="启用已分类 Tool。",
        )
    )

    enable_parser.add_argument(
        "tool_name",
    )

    # --------------------------------------------------------
    # block
    # --------------------------------------------------------

    block_parser = (
        subparsers.add_parser(
            "block",
            help="阻止 Tool。",
        )
    )

    block_parser.add_argument(
        "tool_name",
    )

    return parser


# ============================================================
# 输出
# ============================================================


def print_record(
    record: Any,
) -> None:
    """输出 Tool Policy。"""

    data = record.to_dict()

    print()
    print("=" * 80)

    print(
        f"Tool：{data['tool_name']}"
    )

    print("=" * 80)

    print(
        "source："
        f"{data['tool_source']}"
    )

    print(
        "source_id："
        f"{data['source_id'] or '无'}"
    )

    print(
        "status："
        f"{data['status']}"
    )

    print(
        "enabled："
        f"{data['enabled']}"
    )

    print(
        "effect_type："
        f"{data['effect_type'] or '未分类'}"
    )

    print(
        "external_side_effect："
        f"{data['has_external_side_effect']}"
    )

    print(
        "replay_policy："
        f"{data['replay_policy'] or '未分类'}"
    )

    print(
        "requires_approval："
        f"{data['requires_approval']}"
    )

    print(
        "idempotency_strategy："
        f"{data['idempotency_strategy'] or '无'}"
    )

    print(
        "compensation_tool："
        f"{data['compensation_tool'] or '无'}"
    )

    print(
        "description："
        f"{data['description'] or '无'}"
    )

    print(
        "last_seen_at："
        f"{data['last_seen_at']}"
    )


def print_records(
    records: list[Any],
) -> None:
    """列表输出。"""

    if not records:
        print(
            "没有 Tool Policy。"
        )
        return

    print()

    print(
        f"{'Tool':35}"
        f"{'Status':12}"
        f"{'Enabled':10}"
        f"{'Effect Type':24}"
        f"{'External':10}"
        f"{'Replay'}"
    )

    print("-" * 120)

    for record in records:

        effect_type = (
            record.effect_type.value
            if record.effect_type
            is not None
            else "UNCLASSIFIED"
        )

        replay_policy = (
            record.replay_policy.value
            if record.replay_policy
            is not None
            else "UNCLASSIFIED"
        )

        print(
            f"{record.tool_name:35}"
            f"{record.status.value:12}"
            f"{str(record.enabled):10}"
            f"{effect_type:24}"
            f"{str(record.has_external_side_effect):10}"
            f"{replay_policy}"
        )


# ============================================================
# Main
# ============================================================


def main(
) -> None:

    parser = build_parser()

    args = parser.parse_args()

    repository = (
        ToolPolicyRepository()
    )

    repository.setup()

    command = args.command

    if command == "bootstrap":

        repository.bootstrap_known_tools()

        print(
            "Bootstrap 完成。"
        )

        return

    if command == "list":

        print_records(
            repository.list_all()
        )

        return

    if command == "pending":

        print_records(
            repository.list_pending()
        )

        return

    if command == "stats":

        print(
            json.dumps(
                repository.statistics(),
                ensure_ascii=False,
                indent=2,
            )
        )

        return

    if command == "show":

        record = repository.get(
            args.tool_name
        )

        if record is None:
            print(
                "未找到 Tool："
                f"{args.tool_name}"
            )
            return

        print_record(
            record
        )

        return

    if command == "set":

        record = (
            repository.set_policy(

                tool_name=(
                    args.tool_name
                ),

                effect_type=(
                    args.effect_type
                ),

                has_external_side_effect=(
                    args.external
                ),

                replay_policy=(
                    args.replay_policy
                ),

                requires_approval=(
                    args.approval
                ),

                enabled=(
                    args.enabled
                ),

                status=(
                    args.status
                ),

                tool_source=(
                    args.source
                ),

                source_id=(
                    args.source_id
                ),

                idempotency_strategy=(
                    args.idempotency_strategy
                ),

                compensation_tool=(
                    args.compensation_tool
                ),

                description=(
                    args.description
                ),
            )
        )

        print_record(
            record
        )

        return

    if command == "disable":

        print_record(
            repository.disable(
                args.tool_name
            )
        )

        return

    if command == "enable":

        print_record(
            repository.enable(
                args.tool_name
            )
        )

        return

    if command == "block":

        print_record(
            repository.block(
                args.tool_name
            )
        )

        return

    raise RuntimeError(
        f"未知命令：{command}"
    )


if __name__ == "__main__":
    main()