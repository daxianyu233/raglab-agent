"""Build the security-enabled RAG-LAB Agent Runtime."""

from __future__ import annotations

from pathlib import Path

from raglab.application.agent_factory import (
    build_agent as build_base_agent,
)

from raglab.control.runtime_security import (
    SecureAgentRuntime,
    SQLiteToolPolicyStore,
)


def build_secure_agent(
    config_path: Path,
) -> SecureAgentRuntime:
    """构建带 Policy + HITL 的 Agent Runtime。"""

    base_agent = (
        build_base_agent(
            config_path
        )
    )

    policy_store = (
        SQLiteToolPolicyStore()
    )

    secure_agent = (
        SecureAgentRuntime(
            base_agent,
            policy_store=policy_store,
        )
    )

    print(
        "Agent Runtime Security："
        "ENABLED"
    )

    print(
        "  Tool Policy Registry：ENABLED"
    )

    print(
        "  Fail-Closed：ENABLED"
    )

    print(
        "  HITL：ENABLED"
    )

    print(
        "  Job Single-Flight："
        "由 JobExecutionService 管理"
    )

    return secure_agent