from __future__ import annotations


from pathlib import Path


from raglab.application.secure_agent_factory import (
    build_secure_agent,
)


def build_real_agent():

    config_path = (
        Path("configs")
        /
        "agent.yaml"
    )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Agent config not found: {config_path}"
        )

    return build_secure_agent(
        config_path=config_path
    )


def main():

    print(
        "=" * 80
    )

    print(
        "Phase 8D-2 Real Agent Adapter Test"
    )

    print(
        "=" * 80
    )


    agent = build_real_agent()


    print(
        "\nAgent Type:"
    )

    print(
        type(agent)
    )


    print(
        "\nRun minimal query..."
    )


    result = agent.run(
        "只回复：REAL_AGENT_TEST"
    )


    print(
        "\nResult:"
    )

    print(
        result
    )


    print(
        "\nReal Agent Adapter Test PASS"
    )


if __name__ == "__main__":
    main()