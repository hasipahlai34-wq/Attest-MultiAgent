"""Read-only SARIF ingestion for Attest security baselines.

The parser never writes to the database and never calls GitHub. It turns SARIF
2.1.0 results into ``ActionDraft`` candidates (``draft_type=create_issue``) that
the existing draft review flow can persist or reject.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

_SEVERITY_BY_LEVEL = {"error": "high", "warning": "medium", "note": "low", "none": "low"}
_SEVERITY_BY_TEXT = {
    "critical": "high",
    "high": "high",
    "error": "high",
    "medium": "medium",
    "warning": "medium",
    "moderate": "medium",
    "low": "low",
    "note": "low",
    "info": "low",
}


@dataclass(frozen=True)
class SarifFinding:
    scanner: str
    rule_id: str
    level: str
    severity: str
    message: str
    file_path: str | None
    start_line: int | None
    end_line: int | None
    suggestion: str | None


def _first_line(text: str, limit: int = 160) -> str:
    line = text.strip().splitlines()[0].strip() if text.strip() else "未提供描述"
    return line if len(line) <= limit else line[: limit - 1] + "…"


def _normalize_severity(level: str, properties: dict[str, Any]) -> str:
    raw = str(properties.get("severity") or "").strip().lower()
    if raw in _SEVERITY_BY_TEXT:
        return _SEVERITY_BY_TEXT[raw]
    return _SEVERITY_BY_LEVEL.get(level.lower(), "medium")


def parse_sarif(document: dict[str, Any]) -> list[SarifFinding]:
    """Parse one SARIF document into findings. Raises ValueError on bad input."""

    if not isinstance(document, dict) or not isinstance(document.get("runs"), list):
        raise ValueError("SARIF 文档缺少 runs 数组")
    findings: list[SarifFinding] = []
    for run in document["runs"]:
        if not isinstance(run, dict):
            raise ValueError("SARIF run 必须是对象")
        driver = (run.get("tool") or {}).get("driver") or {}
        scanner = str(driver.get("name") or "unknown-sarif")
        for result in run.get("results") or []:
            if not isinstance(result, dict):
                raise ValueError(f"{scanner} 的 SARIF result 不是对象")
            message = str((result.get("message") or {}).get("text") or "")
            level = str(result.get("level") or "warning")
            properties = result.get("properties") or {}
            location = ((result.get("locations") or [{}])[0] or {}).get("physicalLocation") or {}
            region = location.get("region") or {}
            findings.append(
                SarifFinding(
                    scanner=scanner,
                    rule_id=str(result.get("ruleId") or "unknown"),
                    level=level,
                    severity=_normalize_severity(level, properties),
                    message=message or "未提供描述",
                    file_path=(location.get("artifactLocation") or {}).get("uri"),
                    start_line=region.get("startLine"),
                    end_line=region.get("endLine"),
                    suggestion=properties.get("suggestion"),
                )
            )
    return findings


def load_sarif(path: Path) -> list[SarifFinding]:
    if not path.is_file():
        raise ValueError(f"SARIF 文件不存在：{path}")
    return parse_sarif(json.loads(path.read_text(encoding="utf-8")))


def _location_text(finding: SarifFinding) -> str:
    if not finding.file_path:
        return "未定位"
    if finding.start_line is None:
        return finding.file_path
    return f"{finding.file_path}:{finding.start_line}"


def to_action_draft_candidates(
    findings: list[SarifFinding],
    *,
    repo_id: UUID,
    scan_id: str,
    min_severity: str = "low",
) -> list[dict[str, Any]]:
    """Build ActionDraft kwargs for each finding. Pure data, no side effects."""

    rank = {"low": 0, "medium": 1, "high": 2}
    floor = rank.get(min_severity, 0)
    candidates: list[dict[str, Any]] = []
    for finding in findings:
        if rank.get(finding.severity, 0) < floor:
            continue
        location = _location_text(finding)
        content_lines = [
            f"- 扫描器：`{finding.scanner}`",
            f"- 规则：`{finding.rule_id}`",
            f"- 等级：`{finding.severity}`（SARIF level: `{finding.level}`）",
            f"- 位置：`{location}`",
            f"- 说明：{finding.message.strip()}",
        ]
        if finding.suggestion:
            content_lines.append(f"- 建议：{finding.suggestion.strip()}")
        content_lines.append(f"- 扫描批次：`{scan_id}`")
        candidates.append(
            {
                "repo_id": repo_id,
                "draft_type": "create_issue",
                "target_type": "repository",
                "title": f"[{finding.scanner}] {finding.rule_id} @ {location}"[:300],
                "content": "\n".join(content_lines),
                "risk_level": finding.severity,
                "status": "pending_confirmation",
                "meta": {
                    "source": "aig_sarif",
                    "scan_id": scan_id,
                    "scanner": finding.scanner,
                    "rule_id": finding.rule_id,
                    "severity": finding.severity,
                    "sarif_level": finding.level,
                    "file_path": finding.file_path,
                    "start_line": finding.start_line,
                    "end_line": finding.end_line,
                },
            }
        )
    return candidates
