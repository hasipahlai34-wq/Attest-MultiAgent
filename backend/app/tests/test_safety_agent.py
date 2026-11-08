import json
from pathlib import Path
from typing import Any

import pytest

from app.services.agents.safety_agent import SafetyAgent
from app.services.safety_policy import POLICY_RULES, classify_action

FIXTURES = Path(__file__).parent / "fixtures" / "safety_attack_cases.json"
CASES: list[dict[str, Any]] = json.loads(FIXTURES.read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
@pytest.mark.asyncio
async def test_safety_agent_matches_policy_table(case: dict[str, Any]) -> None:
    result = await SafetyAgent().run(
        {"action": case["action"], "scope": case["scope"], "effect": case["effect"], "draft": "draft body"},
        {},
    )

    assert result["requires_confirmation"] is case["expect"]["requires_confirmation"]
    assert result["risk_level"] == case["expect"]["risk_level"]
    assert result["draft"] == "draft body"


@pytest.mark.parametrize("case", [item for item in CASES if item["effect"] == "external_write"])
def test_external_writes_require_confirmation(case: dict[str, Any]) -> None:
    decision = classify_action(case["action"], case["scope"], case["effect"])

    assert decision.requires_confirmation is True
    assert decision.risk_level in {"medium", "high"}


@pytest.mark.parametrize("case", [item for item in CASES if item["effect"] == "read_only" and item["scope"] == "self_repo"])
def test_read_only_analysis_needs_no_confirmation(case: dict[str, Any]) -> None:
    decision = classify_action(case["action"], case["scope"], case["effect"])

    assert decision.requires_confirmation is False
    assert decision.risk_level == "low"


@pytest.mark.parametrize("case", [item for item in CASES if item["expect"]["denied"]])
def test_cross_repo_and_out_of_boundary_actions_are_denied(case: dict[str, Any]) -> None:
    decision = classify_action(case["action"], case["scope"], case["effect"])

    assert decision.policy_id.startswith("deny_")
    assert decision.requires_confirmation is True
    assert decision.risk_level == "high"


def test_unknown_action_only_uses_keywords_as_fallback() -> None:
    hit = classify_action("写回说明")
    miss = classify_action("look")

    assert hit.policy_id == "fallback_keyword_hit"
    assert hit.requires_confirmation is True
    assert miss.policy_id == "fallback_keyword_miss"
    assert miss.requires_confirmation is False


def test_known_action_is_not_decided_by_keywords() -> None:
    # "read_file" contains no write keyword, but even if it did, the policy table
    # must decide first: a read-only action stays low risk.
    decision = classify_action("read_file", "self_repo", "read_only")

    assert decision.policy_id == "read_only"
    assert decision.requires_confirmation is False


@pytest.mark.asyncio
async def test_safety_agent_output_fields_are_stable() -> None:
    result = await SafetyAgent().run({"action": "comment", "draft": "x"}, {})

    assert set(result) == {"requires_confirmation", "risk_level", "reason", "draft"}


def test_policy_table_has_no_duplicate_rule_ids() -> None:
    ids = [rule.rule_id for rule in POLICY_RULES]

    assert len(ids) == len(set(ids))
