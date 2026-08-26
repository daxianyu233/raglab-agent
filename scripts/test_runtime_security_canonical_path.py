"""Verify Runtime Security canonical module wiring.

运行：
    python -m scripts.test_runtime_security_canonical_path
"""

from raglab.agent.runtime_security import (
    SQLiteToolPolicyStore as AgentSQLiteToolPolicyStore,
    SecureAgentRuntime as AgentSecureAgentRuntime,
    SecureToolNode as AgentSecureToolNode,
)

from raglab.control.runtime_security import (
    SQLiteToolPolicyStore as ControlSQLiteToolPolicyStore,
    SecureAgentRuntime as ControlSecureAgentRuntime,
    SecureToolNode as ControlSecureToolNode,
)


def main() -> None:
    print("=" * 80)
    print("Runtime Security Canonical Path 回归")
    print("=" * 80)

    assert AgentSecureAgentRuntime is ControlSecureAgentRuntime
    print("[PASS] agent.SecureAgentRuntime -> control.SecureAgentRuntime")

    assert AgentSecureToolNode is ControlSecureToolNode
    print("[PASS] agent.SecureToolNode -> control.SecureToolNode")

    assert AgentSQLiteToolPolicyStore is ControlSQLiteToolPolicyStore
    print("[PASS] agent.SQLiteToolPolicyStore -> control.SQLiteToolPolicyStore")

    assert (
        ControlSecureAgentRuntime.__module__
        == "raglab.control.runtime_security"
    )
    print("[PASS] 唯一正式实现位于 raglab.control.runtime_security")

    print("=" * 80)
    print("Runtime Security Canonical Path 回归测试通过")
    print("=" * 80)


if __name__ == "__main__":
    main()