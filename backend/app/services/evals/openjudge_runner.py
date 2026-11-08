"""OpenJudge behaviour grading for Attest agents.

Three upstream graders from ``py-openjudge`` are wired in:

* ``ToolSelectionGrader`` — did ChatAgent pick the right production tool?
* ``PlanFeasibilityGrader`` — is the PlannerAgent spec executable?
* ``ResponseCompletenessGrader`` — do PR review / issue analysis outputs cover
  every required field?

The result payload mirrors ``ragas_runner._finalize_result``: ``status`` (passed /
failed / error / no_cases), ``quality_status``, per-metric ``{score, threshold,
passed}``, ``hard_rules`` and ``diagnoses`` with ``severity`` + ``category``. Judge
failures are execution errors, never quality failures.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import math
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Issue, PullRequest, Repository, WorkflowRun
from app.services.agents.chat_agent import ChatAgent
from app.services.agents.issue_agent import IssueAgent
from app.services.agents.planner_agent import PlannerAgent
from app.services.agents.pr_review_agent import PRReviewAgent
from app.services.evals.ragas_runner import (
    ProgressCallback,
    RagEvalCase,
    _emit_progress,
    _run_chatagent_case,
)
from app.services.evals.suites import AUTO_SMOKE_SUITE_ID, load_eval_suite

OPENJUDGE_METRICS = ("tool_selection", "plan_feasibility", "response_completeness")
_METRIC_SCALE = {"tool_selection": 5.0, "plan_feasibility": 1.0, "response_completeness": 5.0}

ISSUE_REQUIRED_FIELDS = (
    "summary",
    "conclusion",
    "conclusion_reason",
    "category",
    "priority",
    "complexity",
    "suggested_owner",
    "owner_reason",
    "evidence",
    "checklist",
    "action_items",
    "confidence",
)
PR_REQUIRED_FIELDS = (
    "summary",
    "merge_recommendation",
    "recommendation_reason",
    "review_findings",
    "key_changes",
    "risk_points",
    "blocking_issues",
    "review_checklist",
    "test_suggestions",
    "confidence",
)
COMPLETENESS_QUERIES = {
    "issue": (
        "请分析这条 Issue，输出摘要、结论、结论依据、分类、优先级、复杂度、"
        "建议负责人及理由、证据、检查清单、行动项和置信度。"
    ),
    "pull_request": (
        "请审查这个 PR，输出摘要、合入建议及理由、审查发现、关键变更、"
        "风险点、阻塞问题、审查清单、测试建议和置信度。"
    ),
}

AgentFactory = Callable[[], ChatAgent]


class OpenJudgeError(RuntimeError):
    """OpenJudge package, judge model, or a judge call is unavailable."""


class OpenJudgeMappingError(RuntimeError):
    """A Attest field required by a grader is missing or malformed."""


class JudgeScorer(Protocol):
    version: str
    judge_model: str

    async def score(self, kind: str, payload: dict[str, Any]) -> tuple[float, str]: ...


@dataclass(frozen=True)
class OpenJudgeCase:
    id: str
    kind: str
    target: str
    question: str
    target_id: str | None = None


class OpenJudgeScorer:
    """Adapter from Attest's OpenAI-compatible endpoint to OpenJudge graders."""

    def __init__(self) -> None:
        try:
            from openjudge.graders.agent import (
                PlanFeasibilityGrader,
                ResponseCompletenessGrader,
                ToolSelectionGrader,
            )
            from openjudge.models.openai_chat_model import OpenAIChatModel
            from openjudge.models.schema.prompt_template import LanguageEnum
        except Exception as exc:  # pragma: no cover - depends on optional package
            raise OpenJudgeError(f"Unable to import OpenJudge graders: {exc}") from exc

        api_key = settings.openjudge_judge_api_key or settings.llm_api_key
        base_url = (settings.openjudge_judge_base_url or settings.llm_base_url or "").rstrip("/")
        model_name = settings.openjudge_judge_model or settings.llm_model
        if not api_key:
            raise OpenJudgeError("OpenJudge judge API key is not configured")
        if not base_url or not model_name:
            raise OpenJudgeError("OpenJudge judge base URL or model is not configured")

        try:
            model = OpenAIChatModel(
                model=model_name,
                api_key=api_key,
                base_url=base_url,
                timeout=settings.openjudge_timeout_seconds,
                temperature=0,
            )
            self.graders = {
                "tool_selection": ToolSelectionGrader(model=model, language=LanguageEnum.ZH),
                "plan_feasibility": PlanFeasibilityGrader(model=model, language=LanguageEnum.ZH),
                "response_completeness": ResponseCompletenessGrader(model=model, language=LanguageEnum.ZH),
            }
            self.version = importlib.metadata.version("py-openjudge")
        except Exception as exc:
            raise OpenJudgeError(f"Unable to initialize OpenJudge judge: {exc}") from exc
        self.judge_model = model_name

    async def score(self, kind: str, payload: dict[str, Any]) -> tuple[float, str]:
        grader = self.graders.get(kind)
        if grader is None:
            raise OpenJudgeError(f"Unknown OpenJudge grader: {kind}")
        result = await grader.aevaluate(**payload)
        reason = str(getattr(result, "reason", "") or "")
        # Upstream graders swallow their own exceptions and return score 0.0 with
        # this prefix. Treating it as a score would fake a quality failure.
        if reason.startswith("Evaluation error:"):
            if "response_format" in reason or "structured output" in reason.casefold():
                raise OpenJudgeError(
                    f"{reason}；judge 模型不支持 json_schema 结构化输出，"
                    "请用 OPENJUDGE_JUDGE_* 指向支持该能力的模型。"
                )
            raise OpenJudgeError(reason)
        value = getattr(result, "score", None)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise OpenJudgeError(f"{kind} returned non-numeric score: {value!r}")
        return float(value), reason


def _require_text(mapping: dict[str, Any], key: str, where: str) -> str:
    if key not in mapping:
        raise OpenJudgeMappingError(f"{where} 缺少字段 {key}")
    value = mapping[key]
    if not isinstance(value, str) or not value.strip():
        raise OpenJudgeMappingError(f"{where} 的字段 {key} 为空或不是字符串")
    return value.strip()


def _require_list(mapping: dict[str, Any], key: str, where: str) -> list[Any]:
    if key not in mapping:
        raise OpenJudgeMappingError(f"{where} 缺少字段 {key}")
    value = mapping[key]
    if not isinstance(value, list):
        raise OpenJudgeMappingError(f"{where} 的字段 {key} 不是列表")
    return value


def _render_fields(output: dict[str, Any], fields: tuple[str, ...], where: str) -> str:
    missing = [name for name in fields if name not in output]
    if missing:
        raise OpenJudgeMappingError(f"{where} 缺少字段: {', '.join(missing)}")
    lines: list[str] = []
    for name in fields:
        value = output[name]
        text = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else str(value)
        lines.append(f"{name}: {text}")
    return "\n".join(lines)


def _runtime_tool_names(result: dict[str, Any], case_id: str) -> list[str]:
    for event in result.get("agent_events") or []:
        if not isinstance(event, dict) or event.get("type") != "agent_loop_start":
            continue
        names = event.get("available_tools")
        if not isinstance(names, list) or not names:
            raise OpenJudgeMappingError(f"ChatAgent 结果 {case_id} 的 agent_loop_start 缺少 available_tools")
        return [str(name) for name in names if str(name).strip()]
    raise OpenJudgeMappingError(f"ChatAgent 结果 {case_id} 缺少 agent_loop_start 事件，无法确定工具集")


def _tool_definition(agent: ChatAgent, name: str) -> dict[str, str]:
    return {"name": name, "description": agent._tool_description(name)}


def _map_tool_selection(
    case: OpenJudgeCase,
    result: dict[str, Any],
    tool_definitions: list[dict[str, str]],
) -> dict[str, Any]:
    if not case.question.strip():
        raise OpenJudgeMappingError(f"用例 {case.id} 缺少 query")
    if not tool_definitions:
        raise OpenJudgeMappingError("tool_definitions 为空，无法评估工具选择")
    raw_calls = _require_list(result, "tool_calls", f"ChatAgent 结果 {case.id}")
    calls: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw_calls):
        if not isinstance(item, dict):
            raise OpenJudgeMappingError(f"ChatAgent 结果 {case.id} 的第 {index + 1} 个 tool_call 不是对象")
        name = item.get("tool_name")
        if not isinstance(name, str) or not name.strip():
            raise OpenJudgeMappingError(f"ChatAgent 结果 {case.id} 的第 {index + 1} 个 tool_call 缺少 tool_name")
        # workflow.run_engineering_review writes nested task traces back into
        # tool_calls with the parent tool_use_id. They are telemetry, not model
        # tool selections, so keep only the first entry per tool_use_id.
        call_id = item.get("tool_use_id")
        if isinstance(call_id, str) and call_id:
            if call_id in seen_ids:
                continue
            seen_ids.add(call_id)
        calls.append({"name": name, "arguments": dict(item.get("arguments") or {})})
    _require_text(result, "answer", f"ChatAgent 结果 {case.id}")
    return {"query": case.question, "tool_definitions": tool_definitions, "tool_calls": calls}


def _map_completeness(target: str, output: dict[str, Any], case_id: str) -> dict[str, Any]:
    fields = ISSUE_REQUIRED_FIELDS if target == "issue" else PR_REQUIRED_FIELDS
    rendered = _render_fields(output, fields, f"{target} 输出 {case_id}")
    return {"query": COMPLETENESS_QUERIES[target], "response": rendered}


def _render_plan(spec: dict[str, Any]) -> str:
    goal = _require_text(spec, "goal", "Planner 计划")
    tasks = _require_list(spec, "tasks", "Planner 计划")
    if not tasks:
        raise OpenJudgeMappingError("Planner 计划没有任何 task")
    lines = [f"目标：{goal}"]
    for index, task in enumerate(tasks, start=1):
        if not isinstance(task, dict):
            raise OpenJudgeMappingError(f"Planner 计划的第 {index} 个 task 不是对象")
        task_type = _require_text(task, "task_type", f"Planner 计划 task-{index}")
        agent_name = _require_text(task, "agent_name", f"Planner 计划 task-{index}")
        objective = _require_text(task, "objective", f"Planner 计划 task-{index}")
        dependencies = _require_list(task, "dependencies", f"Planner 计划 task-{index}")
        lines.append(f"{index}. {task_type} / {agent_name}：{objective}（依赖：{dependencies or '无'}）")
    return "\n".join(lines)


def _render_memory(spec: dict[str, Any]) -> str:
    criteria = _require_list(spec, "acceptance_criteria", "Planner 计划")
    if not criteria:
        raise OpenJudgeMappingError("Planner 计划缺少 acceptance_criteria")
    return "\n".join(f"- {item}" for item in criteria)


def _repo_signals(db: Session, repo: Repository) -> dict[str, Any]:
    issue = (
        db.query(Issue)
        .filter(Issue.repo_id == repo.id, Issue.state == "open")
        .order_by(Issue.updated_at.desc(), Issue.created_at.desc())
        .first()
    )
    pull_request = (
        db.query(PullRequest)
        .filter(PullRequest.repo_id == repo.id, PullRequest.state == "open")
        .order_by(PullRequest.updated_at.desc(), PullRequest.created_at.desc())
        .first()
    )
    failed_ci = (
        db.query(WorkflowRun)
        .filter(WorkflowRun.repo_id == repo.id, WorkflowRun.conclusion == "failure")
        .order_by(WorkflowRun.created_at.desc())
        .first()
    )
    return {
        "issue": issue,
        "pull_request": pull_request,
        "failed_ci": failed_ci,
        "issue_count": db.query(Issue).filter(Issue.repo_id == repo.id).count(),
        "pull_request_count": db.query(PullRequest).filter(PullRequest.repo_id == repo.id).count(),
        "failed_ci_count": db.query(WorkflowRun)
        .filter(WorkflowRun.repo_id == repo.id, WorkflowRun.conclusion == "failure")
        .count(),
    }


def _planner_context(repo: Repository, signals: dict[str, Any]) -> dict[str, Any]:
    issue: Issue | None = signals["issue"]
    pull_request: PullRequest | None = signals["pull_request"]
    failed_ci: WorkflowRun | None = signals["failed_ci"]
    return {
        "repo_id": str(repo.id),
        "has_repo": True,
        "has_issue": issue is not None,
        "issue_id": str(issue.id) if issue else None,
        "issue_number": issue.number if issue else None,
        "has_pr": pull_request is not None,
        "pr_id": str(pull_request.id) if pull_request else None,
        "pr_number": pull_request.number if pull_request else None,
        "has_ci_run": failed_ci is not None,
        "workflow_run_id": str(failed_ci.id) if failed_ci else None,
        "workflow_run_number": failed_ci.github_run_id if failed_ci else None,
        "enable_rag_task": True,
    }


def _render_observation(repo: Repository, signals: dict[str, Any]) -> str:
    issue: Issue | None = signals["issue"]
    pull_request: PullRequest | None = signals["pull_request"]
    parts = [
        f"仓库 {repo.full_name}：Issue {signals['issue_count']} 个",
        f"PR {signals['pull_request_count']} 个",
        f"失败 CI {signals['failed_ci_count']} 次",
    ]
    if issue:
        parts.append(f"最新 Issue #{issue.number} {issue.title}")
    if pull_request:
        parts.append(f"最新 PR #{pull_request.number} {pull_request.title}")
    return "；".join(parts)


def _auto_question(repo: Repository) -> str:
    return f"查看仓库 {repo.full_name} 当前的工程状态：有哪些未关闭的 Issue、待审查的 PR，以及失败的 CI 运行？"


def _open_issues(db: Session, repo: Repository, limit: int) -> list[Issue]:
    return (
        db.query(Issue)
        .filter(Issue.repo_id == repo.id, Issue.state == "open")
        .order_by(Issue.number.desc())
        .limit(limit)
        .all()
    )


def _open_pull_requests(db: Session, repo: Repository, limit: int) -> list[PullRequest]:
    return (
        db.query(PullRequest)
        .filter(PullRequest.repo_id == repo.id, PullRequest.state == "open")
        .order_by(PullRequest.number.desc())
        .limit(limit)
        .all()
    )


def _build_cases(
    db: Session,
    repo: Repository,
    suite_id: str | None,
    limit: int,
) -> tuple[list[OpenJudgeCase], dict[str, Any]]:
    evaluation_set: dict[str, Any] = {
        "id": AUTO_SMOKE_SUITE_ID,
        "name": AUTO_SMOKE_SUITE_ID,
        "label": "OpenJudge 自动行为冒烟",
        "source": "auto_smoke",
        "sha256": None,
        "case_count": None,
    }
    questions = [_auto_question(repo)]
    if suite_id and suite_id != AUTO_SMOKE_SUITE_ID:
        suite = load_eval_suite(db, repo.id, suite_id)
        evaluation_set = suite.metadata
        questions = [case.question for case in suite.cases][: max(1, limit - 2)]

    cases = [
        OpenJudgeCase(id=f"tool-selection-{index}", kind="tool_selection", target="chat", question=question)
        for index, question in enumerate(questions, start=1)
    ]
    cases.append(
        OpenJudgeCase(
            id="plan-feasibility-1",
            kind="plan_feasibility",
            target="planner",
            question="为当前仓库制定一次工程审查计划：先建立仓库基线，再按现有信号安排 Issue 分析、PR 审查和 CI 检查，最后给出下一步动作。",
        )
    )
    # The remaining budget alternates Issue and PR samples so the regression
    # surface covers several real objects instead of only the newest one.
    budget = max(0, limit - len(cases))
    issues = _open_issues(db, repo, budget)
    pull_requests = _open_pull_requests(db, repo, budget)
    while budget > 0 and (issues or pull_requests):
        if issues:
            issue = issues.pop(0)
            cases.append(
                OpenJudgeCase(
                    id=f"issue-completeness-{issue.number}",
                    kind="response_completeness",
                    target="issue",
                    question=COMPLETENESS_QUERIES["issue"],
                    target_id=str(issue.id),
                )
            )
            budget -= 1
        if pull_requests and budget > 0:
            pull_request = pull_requests.pop(0)
            cases.append(
                OpenJudgeCase(
                    id=f"pr-completeness-{pull_request.number}",
                    kind="response_completeness",
                    target="pull_request",
                    question=COMPLETENESS_QUERIES["pull_request"],
                    target_id=str(pull_request.id),
                )
            )
            budget -= 1
    evaluation_set["case_count"] = min(len(cases), limit)
    return cases[:limit], evaluation_set


async def _tool_selection_payload(
    db: Session,
    repo: Repository,
    case: OpenJudgeCase,
    agent_factory: AgentFactory,
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    agent = agent_factory()
    result = await _run_chatagent_case(
        db,
        repo,
        RagEvalCase(
            id=case.id,
            question=case.question,
            reference=case.question,
            expected_source_ids=[],
        ),
        limit=5,
        agent_factory=lambda: agent,
    )
    payload = _map_tool_selection(
        case,
        result,
        [_tool_definition(agent, name) for name in _runtime_tool_names(result, case.id)],
    )
    tool_calls = [dict(item) for item in result.get("tool_calls") or [] if isinstance(item, dict)]
    return payload, str(result.get("answer") or ""), tool_calls


async def _plan_feasibility_payload(
    db: Session,
    repo: Repository,
    case: OpenJudgeCase,
    agent_factory: AgentFactory,
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    signals = _repo_signals(db, repo)
    spec = await PlannerAgent().run({"message": case.question}, _planner_context(repo, signals))
    payload = {
        "plan": _render_plan(spec),
        "observation": _render_observation(repo, signals),
        "memory": _render_memory(spec),
        "context": _require_text(spec, "goal", "Planner 计划"),
    }
    return payload, payload["plan"], []


async def _response_completeness_payload(
    db: Session,
    repo: Repository,
    case: OpenJudgeCase,
    agent_factory: AgentFactory,
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    if case.target == "issue":
        if not case.target_id:
            raise OpenJudgeMappingError(f"用例 {case.id} 缺少 Issue id")
        issue = db.get(Issue, uuid.UUID(case.target_id))
        if issue is None:
            raise OpenJudgeMappingError(f"用例 {case.id} 指向的 Issue 不存在")
        output = await IssueAgent().run(
            {
                "number": issue.number,
                "title": issue.title,
                "body": issue.body or "",
                "state": issue.state,
                "labels": list(issue.labels or []),
            },
            {"repo_id": str(repo.id), "team_members": []},
        )
    elif case.target == "pull_request":
        if not case.target_id:
            raise OpenJudgeMappingError(f"用例 {case.id} 缺少 PR id")
        pull_request = db.get(PullRequest, uuid.UUID(case.target_id))
        if pull_request is None:
            raise OpenJudgeMappingError(f"用例 {case.id} 指向的 PR 不存在")
        output = await PRReviewAgent().run(
            {
                "number": pull_request.number,
                "title": pull_request.title,
                "body": pull_request.body or "",
                "files": [
                    {"filename": item.filename, "status": item.status, "patch": item.patch or ""}
                    for item in pull_request.files
                ],
                "comments": [],
            },
            {"repo_id": str(repo.id)},
        )
    else:
        raise OpenJudgeMappingError(f"未知的 completeness 目标: {case.target}")
    payload = _map_completeness(case.target, output, case.id)
    return payload, payload["response"], []


async def _run_case(
    db: Session,
    repo: Repository,
    case: OpenJudgeCase,
    *,
    judge_requested: bool,
    scorer: JudgeScorer | None,
    judge_error: str | None,
    agent_factory: AgentFactory,
) -> dict[str, Any]:
    started = time.perf_counter()
    row: dict[str, Any] = {
        "id": case.id,
        "kind": case.kind,
        "target": case.target,
        "question": case.question,
        "response": None,
        "tool_calls": [],
        "retrieved_contexts": [],
        "retrieved_sources": [],
        "hard_rules": {"passed": False},
        "openjudge": None,
        "openjudge_errors": {},
        "openjudge_reasons": {},
        "latency_ms": 0.0,
    }
    try:
        if case.kind == "tool_selection":
            payload, response, tool_calls = await _tool_selection_payload(db, repo, case, agent_factory)
        elif case.kind == "plan_feasibility":
            payload, response, tool_calls = await _plan_feasibility_payload(db, repo, case, agent_factory)
        elif case.kind == "response_completeness":
            payload, response, tool_calls = await _response_completeness_payload(db, repo, case, agent_factory)
        else:
            raise OpenJudgeMappingError(f"未注册的评分维度: {case.kind}")
    except OpenJudgeMappingError as exc:
        row["openjudge_errors"][case.kind] = f"OpenJudgeMappingError: {exc}"
        row["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return row
    except Exception as exc:
        row["openjudge_errors"][case.kind] = f"{type(exc).__name__}: {exc}"
        row["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return row

    row["response"] = response
    row["tool_calls"] = tool_calls
    row["hard_rules"] = {"passed": True, "target": case.target}
    if not judge_requested:
        row["openjudge_skipped_reason"] = "judge_disabled"
    elif scorer is None:
        row["openjudge_errors"][case.kind] = f"OpenJudgeError: {judge_error or 'judge unavailable'}"
    else:
        try:
            raw_score, reason = await asyncio.wait_for(
                scorer.score(case.kind, payload),
                timeout=settings.openjudge_timeout_seconds,
            )
        except (OpenJudgeError, TimeoutError) as exc:
            row["openjudge_errors"][case.kind] = f"{type(exc).__name__}: {exc}"
        else:
            row["openjudge"] = {case.kind: _normalize(case.kind, raw_score)}
            row["openjudge_raw"] = {case.kind: round(raw_score, 4)}
            row["openjudge_reasons"] = {case.kind: reason}
    row["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return row


def _normalize(kind: str, value: float) -> float:
    scale = _METRIC_SCALE[kind]
    return round(min(max(value / scale, 0.0), 1.0), 4)


def _thresholds() -> dict[str, float]:
    return {
        "tool_selection": settings.openjudge_tool_selection_min,
        "plan_feasibility": settings.openjudge_plan_feasibility_min,
        "response_completeness": settings.openjudge_response_completeness_min,
    }


def _mean(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return round(sum(finite) / len(finite), 4) if finite else None


def _metric_details(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    thresholds = _thresholds()
    values: dict[str, list[float]] = {name: [] for name in OPENJUDGE_METRICS}
    for row in rows:
        for name, value in (row.get("openjudge") or {}).items():
            if name in values and isinstance(value, (int, float)) and math.isfinite(float(value)):
                values[name].append(float(value))
    details: dict[str, dict[str, Any]] = {}
    for name, metric_values in values.items():
        score = _mean(metric_values)
        details[name] = {
            "score": score,
            "threshold": thresholds[name],
            "passed": None if score is None else score >= thresholds[name],
            "scored_cases": len(metric_values),
        }
    return details


_DIAGNOSIS_COPY = {
    "tool_selection": (
        "ChatAgent 没有选到合适的工具",
        "检查工具描述、工具集裁剪和系统提示里的路由规则。",
    ),
    "plan_feasibility": (
        "Planner 计划不可执行",
        "检查任务目录、依赖约束和计划提示，避免计划依赖不存在的信号。",
    ),
    "response_completeness": (
        "输出没有覆盖必要项",
        "补齐必需字段，或在 Agent 提示中显式要求逐项输出。",
    ),
}


def _case_diagnoses(row: dict[str, Any]) -> list[dict[str, str]]:
    scores = row.get("openjudge") or {}
    errors = row.get("openjudge_errors") or {}
    thresholds = _thresholds()
    diagnoses: list[dict[str, str]] = []
    if errors:
        names = "、".join(errors)
        diagnoses.append(
            {
                "category": "judge_error",
                "severity": "error",
                "title": "OpenJudge 评分异常",
                "summary": f"{names} 没有得到可信分数；这是执行问题，不代表质量结论。",
                "action": "重试评分；若仍失败，检查 Judge 端点、超时和模型结构化输出。",
            }
        )
    if not scores:
        return diagnoses
    for name, value in scores.items():
        if not isinstance(value, (int, float)) or value >= thresholds[name]:
            continue
        title, action = _DIAGNOSIS_COPY[name]
        diagnoses.append(
            {
                "category": name,
                "severity": "failed",
                "title": title,
                "summary": f"{name} 得分 {value:.4f}，低于门槛 {thresholds[name]:.2f}。",
                "action": action,
            }
        )
    if not diagnoses:
        diagnoses.append(
            {
                "category": "passed",
                "severity": "passed",
                "title": "该用例未发现评分问题",
                "summary": "被测 Agent 的输出完整，评分达到当前门槛。",
                "action": "保留为回归基线。",
            }
        )
    return diagnoses


def _diagnostic_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    failed_cases = 0
    error_cases = 0
    for row in rows:
        row["diagnoses"] = _case_diagnoses(row)
        categories = {item["category"] for item in row["diagnoses"] if item["category"] != "passed"}
        if categories:
            failed_cases += 1
        if row.get("openjudge_errors"):
            error_cases += 1
        for category in categories:
            counts[category] = counts.get(category, 0) + 1
    return {"failed_cases": failed_cases, "error_cases": error_cases, "category_counts": counts}


def _finalize_result(
    *,
    rows: list[dict[str, Any]],
    performance: dict[str, Any],
    judge_requested: bool,
    scorer: JudgeScorer | None,
    judge_error: str | None,
    repo_id: str,
    limit: int,
    evaluation_set: dict[str, Any],
) -> dict[str, Any]:
    metric_rows = _metric_details(rows)
    hard_passed_cases = sum(1 for row in rows if (row.get("hard_rules") or {}).get("passed"))
    hard_pass_rate = round(hard_passed_cases / len(rows), 4) if rows else 0.0
    case_metric_errors = sum(bool(row.get("openjudge_errors")) for row in rows)
    required_scored = all(metric_rows[name]["score"] is not None for name in OPENJUDGE_METRICS)
    judge_available = bool(judge_requested and scorer is not None and required_scored)
    evaluation_complete = bool(judge_available and case_metric_errors == 0)
    semantic_failed = any(metric_rows[name]["passed"] is False for name in OPENJUDGE_METRICS)
    hard_passed = bool(rows) and hard_passed_cases == len(rows)
    if not rows:
        quality_status = "not_evaluated"
        status = "no_cases"
    elif not hard_passed or semantic_failed:
        quality_status = "failed"
        status = "error" if judge_requested and not evaluation_complete else "failed"
    elif not judge_requested:
        quality_status = "partial"
        status = "partial"
    elif not evaluation_complete:
        quality_status = "incomplete"
        status = "error"
    else:
        quality_status = "passed"
        status = "passed"

    summary_metrics = {
        name: value["score"] for name, value in metric_rows.items() if value["score"] is not None
    }
    summary_metrics["overall"] = _mean(list(summary_metrics.values()))
    summary_metrics["hard_gate_pass_rate"] = hard_pass_rate
    diagnostics = _diagnostic_summary(rows)
    return {
        "kind": "openjudge",
        "status": status,
        "execution_status": "completed",
        "quality_status": quality_status,
        "evaluation_complete": evaluation_complete,
        "passed": status == "passed",
        "run_mode": "full",
        "target": {
            "name": "Attest Agents",
            "repo_id": str(repo_id),
            "execution": "production_agents_with_isolated_state",
            "evaluation_mode": True,
        },
        "evaluation_set": evaluation_set,
        "pipeline": {"max_cases": limit},
        **summary_metrics,
        "summary_metrics": summary_metrics,
        "performance": performance,
        "hard_gate": {
            "passed": hard_passed,
            "pass_rate": hard_pass_rate,
            "passed_cases": hard_passed_cases,
            "total_cases": len(rows),
        },
        "openjudge": {
            "requested": judge_requested,
            "available": judge_available,
            "complete": evaluation_complete,
            "passed": (not semantic_failed) if evaluation_complete else None,
            "version": scorer.version if scorer is not None else None,
            "judge_model": scorer.judge_model if scorer is not None else None,
            "error": judge_error or (
                f"{case_metric_errors} evaluation case(s) contain OpenJudge errors"
                if case_metric_errors
                else None
            ),
            "case_errors": case_metric_errors,
            "metrics": metric_rows,
        },
        "diagnostics": diagnostics,
        "cases": rows,
    }


async def run_openjudge_eval(
    db: Session,
    repo_id: str,
    suite_id: str | None = None,
    run_judge: bool = True,
    *,
    scorer: JudgeScorer | None = None,
    agent_factory: AgentFactory | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Grade Attest agent behaviour with OpenJudge's built-in agent graders."""

    repo = db.get(Repository, uuid.UUID(str(repo_id)))
    if repo is None:
        raise ValueError(f"Repository {repo_id} was not found")
    limit = max(0, settings.openjudge_max_cases)
    cases, evaluation_set = _build_cases(db, repo, suite_id, limit)
    judge_requested = bool(run_judge)
    total_cases = len(cases)
    await _emit_progress(
        progress_callback,
        stage="preparing",
        message=f"已准备 {total_cases} 条 OpenJudge 用例，正在初始化 Judge。",
        completed_cases=0,
        total_cases=total_cases,
    )
    judge_error: str | None = None
    if judge_requested and scorer is None:
        try:
            scorer = OpenJudgeScorer()
        except OpenJudgeError as exc:
            judge_error = str(exc)

    factory: AgentFactory = agent_factory or ChatAgent
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, case in enumerate(cases, start=1):
        await _emit_progress(
            progress_callback,
            stage="openjudge",
            message=f"正在运行第 {index}/{total_cases} 条用例：{case.kind}。",
            current_case=index,
            case_id=case.id,
            completed_cases=index - 1,
            total_cases=total_cases,
        )
        rows.append(
            await _run_case(
                db,
                repo,
                case,
                judge_requested=judge_requested,
                scorer=scorer,
                judge_error=judge_error,
                agent_factory=factory,
            )
        )
        case_errors = rows[-1].get("openjudge_errors") or {}
        if case_errors:
            await _emit_progress(
                progress_callback,
                stage="judge_error",
                message=(
                    f"第 {index}/{total_cases} 条用例评分失败："
                    f"{next(iter(case_errors.values()))}；状态将记为 error。"
                ),
                current_case=index,
                case_id=case.id,
                completed_cases=index,
                total_cases=total_cases,
            )
        await _emit_progress(
            progress_callback,
            stage="case_complete",
            message=f"第 {index}/{total_cases} 条用例已完成。",
            current_case=index,
            case_id=case.id,
            completed_cases=index,
            total_cases=total_cases,
        )

    await _emit_progress(
        progress_callback,
        stage="finalizing",
        message="正在汇总 OpenJudge 指标并生成最终判定。",
        completed_cases=total_cases,
        total_cases=total_cases,
    )
    latencies = [float(row["latency_ms"]) for row in rows]
    performance = {
        "cases": len(rows),
        "total_latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "max_latency_ms": max(latencies) if latencies else 0.0,
    }
    return _finalize_result(
        rows=rows,
        performance=performance,
        judge_requested=judge_requested,
        scorer=scorer,
        judge_error=judge_error,
        repo_id=str(repo.id),
        limit=limit,
        evaluation_set=evaluation_set,
    )
