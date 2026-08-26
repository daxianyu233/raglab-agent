from pathlib import Path


from raglab.application.secure_agent_factory import (
    build_secure_agent,
)


from raglab.evaluation.adapters.full_agent_adapter import (
    FullAgentAdapter,
)



def main():


    config_path = Path(
        "config/agent.yaml"
    )


    runtime = build_secure_agent(
        config_path
    )


    adapter = FullAgentAdapter(

        runtime,

        thread_id="adapter-test",

        user_id="test_user",

    )


    result = adapter.send(

        "查询知识库中的相关内容"

    )


    print("="*80)


    print(
        "Answer:"
    )

    print(
        result.answer
    )


    print()

    print(
        "Completed:",
        result.completed_normally
    )


    print()

    print(
        "Tools:",
        result.tool_calls
    )


    print()

    print(
        "Capabilities:",
        result.capability_groups_used
    )


    print()

    print(
        "Latency:",
        result.total_latency_ms
    )


    print()

    print(
        "Pending HITL:",
        result.pending_human_approval
    )


    print()

    print(
        "Secure Agent Adapter Test PASS"
    )



if __name__ == "__main__":

    main()