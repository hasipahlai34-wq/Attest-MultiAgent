"""Deterministic safety policy for Attest agent actions.

Three dimensions decide the risk:

* ``action`` — what the agent wants to do (enum, translated from tool/action names)
* ``scope``  — whose target it touches (current repo, another repo, external system)
* ``effect`` — read-only, internal write, or external write

The policy is an ordered table of pure predicates. Keyword matching is kept only
as the last resort for actions that cannot be mapped to any enum value.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum


class ActionType(StrEnum):
    READ = "read"
    ANALYZE = "analyze"
    DRAFT = "draft"
    COMMENT = "comment"
    CREATE_ISSUE = "create_issue"
    LABEL = "label"
    CLOSE_ISSUE = "close_issue"
    MERGE = "merge"
    SEND_REPORT = "send_report"
    WRITE_FILE = "write_file"
    EXECUTE = "execute"
    UNKNOWN = "unknown"


class TargetScope(StrEnum):
    SELF_REPO = "self_repo"
    OTHER_REPO = "other_repo"
    EXTERNAL = "external"
    UNKNOWN = "unknown"


class WriteEffect(StrEnum):
    READ_ONLY = "read_only"
    INTERNAL_WRITE = "internal_write"
    EXTERNAL_WRITE = "external_write"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SafetyDecision:
    risk_level: str
    requires_confirmation: bool
    reason: str
    policy_id: str
    action_type: str
    scope: str
    effect: str


@dataclass(frozen=True)
class PolicyRule:
    rule_id: str
    matches: Callable[[ActionType, TargetScope, WriteEffect], bool]
    risk_level: str
    requires_confirmation: bool
    reason: str


_ACTION_ALIASES: dict[str, ActionType] = {
    "read": ActionType.READ,
    "read_file": ActionType.READ,
    "list": ActionType.READ,
    "search": ActionType.READ,
    "analyze": ActionType.ANALYZE,
    "analysis": ActionType.ANALYZE,
    "review": ActionType.ANALYZE,
    "summarize": ActionType.ANALYZE,
    "draft": ActionType.DRAFT,
    "draft_only": ActionType.DRAFT,
    "comment": ActionType.COMMENT,
    "reply": ActionType.COMMENT,
    "post_comment": ActionType.COMMENT,
    "create_issue": ActionType.CREATE_ISSUE,
    "open_issue": ActionType.CREATE_ISSUE,
    "label": ActionType.LABEL,
    "add_label": ActionType.LABEL,
    "close_issue": ActionType.CLOSE_ISSUE,
    "close": ActionType.CLOSE_ISSUE,
    "merge": ActionType.MERGE,
    "merge_pr": ActionType.MERGE,
    "send_report": ActionType.SEND_REPORT,
    "send": ActionType.SEND_REPORT,
    "email": ActionType.SEND_REPORT,
    "write_file": ActionType.WRITE_FILE,
    "write": ActionType.WRITE_FILE,
    "write_back": ActionType.WRITE_FILE,
    "execute": ActionType.EXECUTE,
    "run_code": ActionType.EXECUTE,
    "shell": ActionType.EXECUTE,
}

_SCOPE_ALIASES: dict[str, TargetScope] = {
    "repo": TargetScope.SELF_REPO,
    "self": TargetScope.SELF_REPO,
    "self_repo": TargetScope.SELF_REPO,
    "current": TargetScope.SELF_REPO,
    "current_repo": TargetScope.SELF_REPO,
    "other": TargetScope.OTHER_REPO,
    "other_repo": TargetScope.OTHER_REPO,
    "cross_repo": TargetScope.OTHER_REPO,
    "external_repo": TargetScope.OTHER_REPO,
    "external": TargetScope.EXTERNAL,
    "email": TargetScope.EXTERNAL,
    "webhook": TargetScope.EXTERNAL,
    "third_party": TargetScope.EXTERNAL,
}

_EFFECT_ALIASES: dict[str, WriteEffect] = {
    "read_only": WriteEffect.READ_ONLY,
    "read": WriteEffect.READ_ONLY,
    "none": WriteEffect.READ_ONLY,
    "internal_write": WriteEffect.INTERNAL_WRITE,
    "draft": WriteEffect.INTERNAL_WRITE,
    "local_write": WriteEffect.INTERNAL_WRITE,
    "external_write": WriteEffect.EXTERNAL_WRITE,
    "write": WriteEffect.EXTERNAL_WRITE,
}

_DEFAULT_EFFECT: dict[ActionType, WriteEffect] = {
    ActionType.READ: WriteEffect.READ_ONLY,
    ActionType.ANALYZE: WriteEffect.READ_ONLY,
    ActionType.DRAFT: WriteEffect.INTERNAL_WRITE,
    ActionType.WRITE_FILE: WriteEffect.INTERNAL_WRITE,
    ActionType.EXECUTE: WriteEffect.INTERNAL_WRITE,
    ActionType.COMMENT: WriteEffect.EXTERNAL_WRITE,
    ActionType.CREATE_ISSUE: WriteEffect.EXTERNAL_WRITE,
    ActionType.LABEL: WriteEffect.EXTERNAL_WRITE,
    ActionType.CLOSE_ISSUE: WriteEffect.EXTERNAL_WRITE,
    ActionType.MERGE: WriteEffect.EXTERNAL_WRITE,
    ActionType.SEND_REPORT: WriteEffect.EXTERNAL_WRITE,
    ActionType.UNKNOWN: WriteEffect.UNKNOWN,
}

POLICY_RULES: tuple[PolicyRule, ...] = (
    PolicyRule(
        rule_id="deny_cross_repo",
        matches=lambda action, scope, effect: scope is TargetScope.OTHER_REPO,
        risk_level="high",
        requires_confirmation=True,
        reason="跨仓库操作超出当前授权范围，已拒绝执行，只保留待人工确认的草稿。",
    ),
    PolicyRule(
        rule_id="deny_external_resource",
        matches=lambda action, scope, effect: scope is TargetScope.EXTERNAL
        and action in {ActionType.READ, ActionType.WRITE_FILE, ActionType.EXECUTE},
        risk_level="high",
        requires_confirmation=True,
        reason="目标路径或资源在当前工作区之外，已拒绝执行，避免越权读取或写入。",
    ),
    PolicyRule(
        rule_id="deny_unknown_scope_write",
        matches=lambda action, scope, effect: (
            scope is TargetScope.UNKNOWN and effect in {WriteEffect.INTERNAL_WRITE, WriteEffect.EXTERNAL_WRITE}
        ),
        risk_level="high",
        requires_confirmation=True,
        reason="写操作的目标范围无法确认，按最坏情况拦截，等待人工确认。",
    ),
    PolicyRule(
        rule_id="external_send",
        matches=lambda action, scope, effect: (
            effect is WriteEffect.EXTERNAL_WRITE and action is ActionType.SEND_REPORT
        ),
        risk_level="high",
        requires_confirmation=True,
        reason="内容将发送到外部渠道，可能不可撤回，必须人工确认。",
    ),
    PolicyRule(
        rule_id="external_write",
        matches=lambda action, scope, effect: effect is WriteEffect.EXTERNAL_WRITE,
        risk_level="medium",
        requires_confirmation=True,
        reason="动作会写入 GitHub 等外部系统，只生成草稿并等待人工确认。",
    ),
    PolicyRule(
        rule_id="internal_write",
        matches=lambda action, scope, effect: effect is WriteEffect.INTERNAL_WRITE,
        risk_level="medium",
        requires_confirmation=True,
        reason="动作会修改本地工作区或草稿，需要人工确认后再继续。",
    ),
    PolicyRule(
        rule_id="read_only",
        matches=lambda action, scope, effect: effect is WriteEffect.READ_ONLY,
        risk_level="low",
        requires_confirmation=False,
        reason="只读分析操作不改变仓库或外部系统，不需要人工确认。",
    ),
)

# Last-resort hints, only consulted when the action cannot be mapped to an enum.
_LEGACY_KEYWORDS = (
    "comment",
    "create_issue",
    "label",
    "close_issue",
    "send_report",
    "write",
    "post",
    "send",
    "评论",
    "创建",
    "新建",
    "发给",
    "发送",
    "改标签",
    "关闭",
    "写回",
)


def _action_type(action: str) -> ActionType:
    return _ACTION_ALIASES.get(action.strip().lower(), ActionType.UNKNOWN)


def _target_scope(scope: str | None) -> TargetScope:
    if not scope:
        return TargetScope.SELF_REPO
    return _SCOPE_ALIASES.get(str(scope).strip().lower(), TargetScope.UNKNOWN)


def _write_effect(effect: str | None, action: ActionType) -> WriteEffect:
    if effect:
        return _EFFECT_ALIASES.get(str(effect).strip().lower(), WriteEffect.UNKNOWN)
    return _DEFAULT_EFFECT[action]


def classify_action(action: str, scope: str | None = None, effect: str | None = None) -> SafetyDecision:
    """Return the policy decision for an action. Pure function, no I/O."""

    action_type = _action_type(str(action or ""))
    target_scope = _target_scope(scope)
    write_effect = _write_effect(effect, action_type)
    for rule in POLICY_RULES:
        if rule.matches(action_type, target_scope, write_effect):
            return SafetyDecision(
                risk_level=rule.risk_level,
                requires_confirmation=rule.requires_confirmation,
                reason=rule.reason,
                policy_id=rule.rule_id,
                action_type=action_type.value,
                scope=target_scope.value,
                effect=write_effect.value,
            )
    # Nothing matched: fail closed and fall back to the legacy keyword hints.
    lowered = str(action or "").lower()
    if any(token in lowered for token in _LEGACY_KEYWORDS):
        return SafetyDecision(
            risk_level="medium",
            requires_confirmation=True,
            reason="动作未映射到策略表，关键词兜底命中写操作特征，按需要确认处理。",
            policy_id="fallback_keyword_hit",
            action_type=action_type.value,
            scope=target_scope.value,
            effect=write_effect.value,
        )
    return SafetyDecision(
        risk_level="low",
        requires_confirmation=False,
        reason="动作未映射到策略表，关键词兜底也未命中写操作特征，按只读处理。",
        policy_id="fallback_keyword_miss",
        action_type=action_type.value,
        scope=target_scope.value,
        effect=write_effect.value,
    )
