"""启动带安全控制面的 RAGLab Agent CLI。"""

from __future__ import annotations

from raglab.application.policy_agent_factory import (
    build_agent as build_policy_agent,
)

from scripts import (
    chat_automatic_memory_agent
    as original_cli,
)


def main(
) -> None:

    # --------------------------------------------------------
    # 使用 Policy-aware Factory
    # --------------------------------------------------------

    original_cli.build_agent = (
        build_policy_agent
    )

    # --------------------------------------------------------
    # 在原 CLI Help 上增加 HITL 命令。
    # --------------------------------------------------------

    original_print_help = (
        original_cli.print_help
    )

    def print_help_with_hitl(
    ) -> None:

        original_print_help()

        print(
            "/approve              "
            "批准当前等待中的高风险 Tool"
        )

        print(
            "/reject <原因>         "
            "拒绝当前等待中的高风险 Tool"
        )

    original_cli.print_help = (
        print_help_with_hitl
    )

    original_cli.main()


if __name__ == "__main__":
    main()