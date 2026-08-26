from pathlib import Path


from raglab.evaluation.adapters.secure_runtime_adapter import (
    SecureRuntimeAdapter,
)



def main():

    config_path = Path(
        "config/agent.yaml"
    )


    adapter = SecureRuntimeAdapter(
        config_path
    )


    result = adapter.send(

        "从知识库查询 RRF_TEST_FACT_731",

        thread_id="adapter-real-test",

    )


    print("=" * 80)


    print(
        "Answer:"
    )

    print(
        result.answer
    )


    print()


    print(
        "Tools:"
    )

    print(
        result.tool_calls
    )


    print()


    print(
        "Capabilities:"
    )

    print(
        result.capability_groups_used
    )


    print()


    print(
        "State:"
    )

    print(
        result.state
    )



    assert (
        result.completed_normally
    )


    assert (
        "knowledge_retrieval"
        in
        result.capability_groups_used
    )


    print()


    print(
        "Secure Runtime Adapter PASS"
    )



if __name__ == "__main__":

    main()