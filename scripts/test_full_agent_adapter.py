from raglab.evaluation.adapters.full_agent_adapter import (
    FullAgentAdapter
)


class FakeAgent:


    def run(
        self,
        message,
        thread_id,
        user_id
    ):

        return {

            "answer":
                "测试回答",


            "tool_trace":

                [

                    {
                        "tool":
                        "search_knowledge_base"
                    },

                    {
                        "tool":
                        "query_github_intelligence_sql"
                    }

                ],


            "approval_state":
                "",


            "side_effect_count":
                0
        }



def main():


    adapter = FullAgentAdapter(
        FakeAgent()
    )


    result = adapter.run(

        "测试",

        thread_id="adapter-test"

    )


    print("="*80)

    print(
        "Answer:",
        result.answer
    )

    print(
        "Tools:",
        result.tool_names
    )

    print(
        "Capabilities:",
        result.capability_groups
    )

    print(
        "Retrieval:",
        result.retrieval_calls
    )


    assert (
        result.total_tool_calls
        ==
        2
    )


    assert (
        "knowledge_retrieval"
        in
        result.capability_groups
    )


    assert (
        "structured_query"
        in
        result.capability_groups
    )


    print()

    print(
        "FullAgent Adapter Test PASS"
    )



if __name__ == "__main__":

    main()
    