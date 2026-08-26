"""人工管理 External Effect Remediation Case。

常用命令：

查看未处理工单：

    python -m scripts.manage_remediations list --status OPEN

查看工单：

    python -m scripts.manage_remediations show <case_id>

开始处理：

    python -m scripts.manage_remediations start <case_id> \
        --actor huangwu \
        --note "开始检查实际 GitHub 情报数据"

添加处理记录：

    python -m scripts.manage_remediations note <case_id> \
        --actor huangwu \
        --message "确认 SQLite 已被更新，但日报尚未发布"

处理完成：

    python -m scripts.manage_remediations resolve <case_id> \
        --actor huangwu \
        --resolution "已重新执行正确版本并重建 RAG 索引"

接受风险：

    python -m scripts.manage_remediations accept-risk <case_id> \
        --actor huangwu \
        --reason "该外部影响可接受，不再继续处理"

重新打开：

    python -m scripts.manage_remediations reopen <case_id> \
        --actor huangwu \
        --reason "发现之前的修复仍有遗漏"
"""

from __future__ import annotations

import argparse
import json

from typing import Any

from raglab.control.external_effect_repository import (
    ExternalEffectRepository,
)

from raglab.control.remediation import (
    RemediationStatus,
)

from raglab.control.remediation_repository import (
    RemediationRepository,
)


# ============================================================
# Parser
# ============================================================


def build_parser(
) -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "管理无法自动补偿的"
            " External Effect 人工修复工单。"
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

    list_parser = (
        subparsers.add_parser(
            "list",
            help="查看 Remediation Cases。",
        )
    )

    list_parser.add_argument(
        "--status",
        choices=[
            item.value
            for item
            in RemediationStatus
        ],
        default=None,
    )

    list_parser.add_argument(
        "--thread-id",
        default=None,
    )

    list_parser.add_argument(
        "--tool",
        default=None,
    )

    list_parser.add_argument(
        "--limit",
        type=int,
        default=50,
    )

    # --------------------------------------------------------
    # show
    # --------------------------------------------------------

    show_parser = (
        subparsers.add_parser(
            "show",
            help="查看一个 Remediation Case。",
        )
    )

    show_parser.add_argument(
        "case_id"
    )

    # --------------------------------------------------------
    # stats
    # --------------------------------------------------------

    subparsers.add_parser(
        "stats",
        help="查看 Remediation 统计。",
    )

    # --------------------------------------------------------
    # start
    # --------------------------------------------------------

    start_parser = (
        subparsers.add_parser(
            "start",
            help="开始处理一个 Case。",
        )
    )

    start_parser.add_argument(
        "case_id"
    )

    start_parser.add_argument(
        "--actor",
        required=True,
    )

    start_parser.add_argument(
        "--note",
        default=None,
    )

    # --------------------------------------------------------
    # note
    # --------------------------------------------------------

    note_parser = (
        subparsers.add_parser(
            "note",
            help="追加人工处理反馈。",
        )
    )

    note_parser.add_argument(
        "case_id"
    )

    note_parser.add_argument(
        "--actor",
        required=True,
    )

    note_parser.add_argument(
        "--message",
        required=True,
    )

    # --------------------------------------------------------
    # resolve
    # --------------------------------------------------------

    resolve_parser = (
        subparsers.add_parser(
            "resolve",
            help="确认人工修复完成。",
        )
    )

    resolve_parser.add_argument(
        "case_id"
    )

    resolve_parser.add_argument(
        "--actor",
        required=True,
    )

    resolve_parser.add_argument(
        "--resolution",
        required=True,
    )

    # --------------------------------------------------------
    # accept-risk
    # --------------------------------------------------------

    risk_parser = (
        subparsers.add_parser(
            "accept-risk",
            help=(
                "无法进一步修复时"
                "明确接受剩余风险。"
            ),
        )
    )

    risk_parser.add_argument(
        "case_id"
    )

    risk_parser.add_argument(
        "--actor",
        required=True,
    )

    risk_parser.add_argument(
        "--reason",
        required=True,
    )

    # --------------------------------------------------------
    # reopen
    # --------------------------------------------------------

    reopen_parser = (
        subparsers.add_parser(
            "reopen",
            help="重新打开已关闭工单。",
        )
    )

    reopen_parser.add_argument(
        "case_id"
    )

    reopen_parser.add_argument(
        "--actor",
        required=True,
    )

    reopen_parser.add_argument(
        "--reason",
        required=True,
    )

    return parser


# ============================================================
# Output
# ============================================================


def print_case_list(
    cases: list[Any],
) -> None:

    if not cases:

        print(
            "没有 Remediation Case。"
        )

        return

    print()

    print(
        f"{'Case ID':34}"
        f"{'Status':16}"
        f"{'Priority':12}"
        f"{'Action':20}"
        f"{'Tool'}"
    )

    print(
        "-" * 130
    )

    for case in cases:

        print(
            f"{case.case_id:34}"
            f"{case.status.value:16}"
            f"{case.priority.value:12}"
            f"{case.action_type.value:20}"
            f"{case.tool_name}"
        )


def print_case(
    case: Any,
    *,
    effect: Any | None,
    feedback: list[Any],
) -> None:

    print()
    print("=" * 80)
    print(
        "Remediation Case"
    )
    print("=" * 80)

    print(
        f"case_id：{case.case_id}"
    )

    print(
        f"status：{case.status.value}"
    )

    print(
        f"priority：{case.priority.value}"
    )

    print(
        "action_type："
        f"{case.action_type.value}"
    )

    print(
        f"owner：{case.owner or '未分配'}"
    )

    print(
        f"tool：{case.tool_name}"
    )

    print(
        f"thread_id：{case.thread_id}"
    )

    print(
        f"effect_id：{case.effect_id}"
    )

    print(
        f"plan_id：{case.plan_id}"
    )

    print()

    print(
        "问题摘要："
    )

    print(
        case.summary
    )

    print()

    print(
        "系统判断原因："
    )

    print(
        case.reason
    )

    # --------------------------------------------------------
    # 原 External Effect
    # --------------------------------------------------------

    print()
    print("-" * 80)
    print(
        "原 External Effect"
    )
    print("-" * 80)

    if effect is None:

        print(
            "Effect Ledger 中未找到记录。"
        )

    else:

        print(
            "effect_type："
            f"{effect.effect_type.value}"
        )

        print(
            "effect_status："
            f"{effect.status.value}"
        )

        print(
            "execution_mode："
            f"{effect.execution_mode}"
        )

        print(
            "checkpoint_id："
            f"{effect.checkpoint_id}"
        )

        print()

        print(
            "原 Tool 参数："
        )

        print(
            effect.args_json
        )

        print()

        print(
            "原 Tool 结果："
        )

        print(
            effect.result_text
            or ""
        )

        if effect.error_text:

            print()

            print(
                "原 Tool 错误："
            )

            print(
                effect.error_text
            )

    # --------------------------------------------------------
    # 人工处理结果
    # --------------------------------------------------------

    if case.resolution_note:

        print()
        print("-" * 80)
        print(
            "最终处理结果"
        )
        print("-" * 80)

        print(
            case.resolution_note
        )

    # --------------------------------------------------------
    # Feedback Timeline
    # --------------------------------------------------------

    print()
    print("-" * 80)
    print(
        "人工处理时间线"
    )
    print("-" * 80)

    if not feedback:

        print(
            "暂无反馈记录。"
        )

    else:

        for item in feedback:

            print(
                f"[{item.created_at}] "
                f"{item.feedback_type.value} "
                f"actor={item.actor}"
            )

            print(
                f"    {item.message}"
            )


# ============================================================
# Main
# ============================================================


def main(
) -> None:

    args = (
        build_parser()
        .parse_args()
    )

    repository = (
        RemediationRepository()
    )

    repository.setup()

    effect_repository = (
        ExternalEffectRepository(
            database_path=(
                repository.database_path
            )
        )
    )

    effect_repository.setup()

    # --------------------------------------------------------
    # list
    # --------------------------------------------------------

    if args.command == "list":

        cases = (
            repository.list_cases(
                status=(
                    args.status
                ),
                thread_id=(
                    args.thread_id
                ),
                tool_name=(
                    args.tool
                ),
                limit=(
                    args.limit
                ),
            )
        )

        print_case_list(
            cases
        )

        return

    # --------------------------------------------------------
    # stats
    # --------------------------------------------------------

    if args.command == "stats":

        print(
            json.dumps(
                repository.statistics(),
                ensure_ascii=False,
                indent=2,
            )
        )

        return

    # --------------------------------------------------------
    # show
    # --------------------------------------------------------

    if args.command == "show":

        case = repository.get_case(
            args.case_id
        )

        if case is None:

            print(
                "Remediation Case 不存在："
                f"{args.case_id}"
            )

            return

        effect = (
            effect_repository.get(
                case.effect_id
            )
        )

        feedback = (
            repository.list_feedback(
                case.case_id
            )
        )

        print_case(
            case,
            effect=effect,
            feedback=feedback,
        )

        return

    # --------------------------------------------------------
    # start
    # --------------------------------------------------------

    if args.command == "start":

        case = (
            repository.start_case(
                case_id=(
                    args.case_id
                ),
                actor=(
                    args.actor
                ),
                note=(
                    args.note
                ),
            )
        )

        print(
            "已开始处理："
            f"{case.case_id}"
        )

        print(
            f"status={case.status.value}"
        )

        print(
            f"owner={case.owner}"
        )

        return

    # --------------------------------------------------------
    # note
    # --------------------------------------------------------

    if args.command == "note":

        feedback = (
            repository.add_note(
                case_id=(
                    args.case_id
                ),
                actor=(
                    args.actor
                ),
                message=(
                    args.message
                ),
            )
        )

        print(
            "反馈已记录："
            f"{feedback.feedback_id}"
        )

        return

    # --------------------------------------------------------
    # resolve
    # --------------------------------------------------------

    if args.command == "resolve":

        case = (
            repository.resolve_case(
                case_id=(
                    args.case_id
                ),
                actor=(
                    args.actor
                ),
                resolution=(
                    args.resolution
                ),
            )
        )

        print(
            "Remediation 已完成："
            f"{case.case_id}"
        )

        print(
            f"status={case.status.value}"
        )

        return

    # --------------------------------------------------------
    # accept-risk
    # --------------------------------------------------------

    if (
        args.command
        == "accept-risk"
    ):

        case = (
            repository.accept_risk(
                case_id=(
                    args.case_id
                ),
                actor=(
                    args.actor
                ),
                reason=(
                    args.reason
                ),
            )
        )

        print(
            "已记录风险接受："
            f"{case.case_id}"
        )

        print(
            f"status={case.status.value}"
        )

        return

    # --------------------------------------------------------
    # reopen
    # --------------------------------------------------------

    if args.command == "reopen":

        case = (
            repository.reopen_case(
                case_id=(
                    args.case_id
                ),
                actor=(
                    args.actor
                ),
                reason=(
                    args.reason
                ),
            )
        )

        print(
            "Remediation 已重新打开："
            f"{case.case_id}"
        )

        print(
            f"status={case.status.value}"
        )

        return

    raise RuntimeError(
        f"未知命令：{args.command}"
    )


if __name__ == "__main__":
    main()