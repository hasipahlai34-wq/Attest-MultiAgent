import uuid

import pytest

from app.services.security_scan import parse_sarif, to_action_draft_candidates

SARIF = {
    "version": "2.1.0",
    "runs": [
        {
            "tool": {"driver": {"name": "aig-skill-scan"}},
            "results": [
                {
                    "ruleId": "T04",
                    "level": "error",
                    "message": {"text": "Shell injection\n\nUse a fixed command."},
                    "locations": [
                        {
                            "physicalLocation": {
                                "artifactLocation": {"uri": "scripts/setup.sh"},
                                "region": {"startLine": 12, "endLine": 18},
                            }
                        }
                    ],
                    "properties": {"severity": "High", "suggestion": "Remove curl | bash."},
                },
                {
                    "ruleId": "T02",
                    "level": "warning",
                    "message": {"text": "Broad file read"},
                    "locations": [{"physicalLocation": {"artifactLocation": {"uri": "SKILL.md"}}}],
                },
            ],
        }
    ],
}


def test_parse_sarif_extracts_location_and_severity() -> None:
    findings = parse_sarif(SARIF)

    assert [item.rule_id for item in findings] == ["T04", "T02"]
    assert findings[0].severity == "high"
    assert findings[0].file_path == "scripts/setup.sh"
    assert findings[0].start_line == 12
    assert findings[0].suggestion == "Remove curl | bash."
    assert findings[1].severity == "medium"
    assert findings[1].start_line is None


def test_candidates_carry_rule_file_line_and_severity() -> None:
    repo_id = uuid.uuid4()

    candidates = to_action_draft_candidates(parse_sarif(SARIF), repo_id=repo_id, scan_id="20261028-1500")

    assert len(candidates) == 2
    top = candidates[0]
    assert top["draft_type"] == "create_issue"
    assert top["risk_level"] == "high"
    assert top["status"] == "pending_confirmation"
    assert top["meta"]["rule_id"] == "T04"
    assert top["meta"]["file_path"] == "scripts/setup.sh"
    assert top["meta"]["start_line"] == 12
    assert "T04" in top["title"] and "scripts/setup.sh:12" in top["title"]


def test_min_severity_filters_low_findings() -> None:
    findings = parse_sarif(SARIF)

    assert len(to_action_draft_candidates(findings, repo_id=uuid.uuid4(), scan_id="s", min_severity="high")) == 1


def test_invalid_sarif_raises() -> None:
    with pytest.raises(ValueError, match="runs"):
        parse_sarif({"version": "2.1.0"})
