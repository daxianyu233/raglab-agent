"""Policy + Effect + HITL + Remediation Agent Factory。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from raglab.application.agent_factory import (
    build_agent as build_base_agent,
)

from raglab.control.external_effect_repository import (
    ExternalEffectRepository,
)

from raglab.control.human_approval_repository import (
    HumanApprovalAuditRepository,
)

from raglab.control.runtime_guard import (
    ToolPolicyRuntimeGuard,
)

from raglab.control.tool_policy_repository import (
    ToolPolicyRepository,
)


def build_agent(
    config_path: Path,
) -> Any:
    """创建带完整安全控制面的 Agent。"""

    # --------------------------------------------------------
    # 原 Agent
    # --------------------------------------------------------

    agent = build_base_agent(
        config_path
    )

    # --------------------------------------------------------
    # Tool Policy
    # --------------------------------------------------------

    policy_repository = (
        ToolPolicyRepository()
    )

    policy_repository.setup()

    policy_repository.bootstrap_known_tools()

    # --------------------------------------------------------
    # External Effect
    # --------------------------------------------------------

    effect_repository = (
        ExternalEffectRepository(
            database_path=(
                policy_repository
                .database_path
            )
        )
    )

    effect_repository.setup()

    # --------------------------------------------------------
    # Human Approval Audit
    # --------------------------------------------------------

    approval_repository = (
        HumanApprovalAuditRepository(
            database_path=(
                policy_repository
                .database_path
            )
        )
    )

    approval_repository.setup()

    # --------------------------------------------------------
    # Runtime Guard
    # --------------------------------------------------------

    guard = ToolPolicyRuntimeGuard(

        agent=agent,

        repository=(
            policy_repository
        ),

        effect_repository=(
            effect_repository
        ),

        approval_repository=(
            approval_repository
        ),
    )

    guard.install()

    # --------------------------------------------------------
    # Startup
    # --------------------------------------------------------

    policy_stats = (
        policy_repository.statistics()
    )

    effect_stats = (
        effect_repository.statistics()
    )

    approval_stats = (
        approval_repository.statistics()
    )

    print(
        "Tool Policy Registry："
        f"{policy_stats['database_path']}"
    )

    print(
        "Tool Policy："
        f"total={policy_stats['total']}, "
        f"active={policy_stats['active']}, "
        f"pending={policy_stats['pending']}, "
        f"blocked={policy_stats['blocked']}"
    )

    print(
        "External Effect Ledger："
        f"total={effect_stats['total']}"
    )

    print(
        "Human Approval Audit："
        f"total={approval_stats['total']}"
    )

    return agent