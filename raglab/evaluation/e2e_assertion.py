from __future__ import annotations

from typing import Any

from raglab.evaluation.models.e2e_observation import (
    E2ETurnObservation,
)


def evaluate_case(
    observation: E2ETurnObservation,
    assertions: dict[str, Any],
) -> tuple[bool, list[str]]:


    errors = []


    if assertions.get(
        "answer_nonempty",
        False,
    ):

        if not observation.answer.strip():

            errors.append(
                "answer empty"
            )



    if (
        "max_tool_calls"
        in assertions
    ):

        if len(
            observation.tool_calls
        ) > assertions["max_tool_calls"]:

            errors.append(
                "tool calls exceed limit"
            )



    required_caps = assertions.get(
        "must_use_capability_groups",
        [],
    )


    for cap in required_caps:

        if (
            cap
            not in
            observation.capability_groups_used
        ):

            errors.append(
                f"missing capability: {cap}"
            )



    forbidden_caps = assertions.get(
        "must_not_use_capability_groups",
        [],
    )


    for cap in forbidden_caps:

        if (
            cap
            in
            observation.capability_groups_used
        ):

            errors.append(
                f"forbidden capability used: {cap}"
            )



    if assertions.get(
        "pending_human_approval",
        False,
    ):

        if not observation.pending_human_approval:

            errors.append(
                "approval not pending"
            )

    if "pending_human_approval_after" in assertions:
        expected_pending = bool(assertions["pending_human_approval_after"])
        if observation.pending_human_approval != expected_pending:
            errors.append("approval pending state mismatch")

    if assertions.get("dynamic_capability_loaded", False):
        loaded_skills = observation.state.get("loaded_skills", [])
        used_skill_tool = "skill_management" in observation.capability_groups_used
        if not loaded_skills and not used_skill_tool:
            errors.append("dynamic capability not loaded")

    if "approval_decision" in assertions:
        actual_decision = observation.state.get("approval_decision", "")
        if actual_decision != assertions["approval_decision"]:
            errors.append("approval decision mismatch")

    if assertions.get("external_effect_ledger_entry_created", False):
        if not observation.state.get("external_effect_ledger_entry_created", False):
            errors.append("external effect ledger entry not created")

    if assertions.get("no_duplicate_side_effect", False):
        if not observation.state.get("no_duplicate_side_effect", False):
            errors.append("duplicate side effect detected")



    if (
        "write_side_effect_count"
        in assertions
    ):

        expected = assertions[
            "write_side_effect_count"
        ]

        if (
            observation.write_side_effect_count
            !=
            expected
        ):

            errors.append(
                "side effect count mismatch"
            )



    return (
        len(errors) == 0,
        errors,
    )
