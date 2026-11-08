import asyncio
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.db.models import Base, Issue, PRFile, PullRequest, Repository
from app.services.evals import openjudge_runner
from app.services.evals.openjudge_runner import (
    OpenJudgeCase,
    OpenJudgeError,
    OpenJudgeMappingError,
    _map_tool_selection,
    run_openjudge_eval,
)


class FakeScorer:
    version = "test-openjudge"
    judge_model = "fake-judge"

    def __init__(
        self,
        scores: dict[str, float] | None = None,
        error: Exception | None = None,
        delay: float = 0.0,
    ) -> None:
        self.scores = scores or {
            "tool_selection": 5.0,
            "plan_feasibility": 1.0,
            "response_completeness": 4.0,
        }
        self.error = error
        self.delay = delay

    async def score(self, kind: str, payload: dict) -> tuple[float, str]:
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return self.scores[kind], "fake reason"


async def _fake_payload(db, repo, case, agent_factory):
    return {"stub": case.kind}, f"{case.kind} output", []


@pytest.fixture
def db_with_repo():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    repo = Repository(owner="local", name="openjudge", full_name="local/openjudge")
    db.add(repo)
    db.flush()
    db.add(
        Issue(
            repo_id=repo.id,
            number=7,
            title="Login API returns 500",
            body="Token refresh fails when the session expires",
            state="open",
            labels=["bug"],
        )
    )
    pull_request = PullRequest(
        repo_id=repo.id,
        number=12,
        title="Refactor auth middleware",
        body="Split token validation and session refresh",
        state="open",
    )
    db.add(pull_request)
    db.flush()
    db.add(PRFile(pr_id=pull_request.id, filename="backend/app/auth.py", status="modified", patch="+refresh"))
    db.commit()
    yield db, repo
    db.close()


@pytest.fixture
def stub_targets(monkeypatch):
    for name in (
        "_tool_selection_payload",
        "_plan_feasibility_payload",
        "_response_completeness_payload",
    ):
        monkeypatch.setattr(openjudge_runner, name, _fake_payload)


@pytest.mark.asyncio
async def test_openjudge_returns_three_normalized_dimensions(db_with_repo, stub_targets) -> None:
    db, repo = db_with_repo

    result = await run_openjudge_eval(db, str(repo.id), scorer=FakeScorer())

    assert result["kind"] == "openjudge"
    assert result["status"] == "passed"
    assert result["quality_status"] == "passed"
    assert result["evaluation_complete"] is True
    assert result["openjudge"]["version"] == "test-openjudge"
    metrics = result["openjudge"]["metrics"]
    assert metrics["tool_selection"]["score"] == 1.0
    assert metrics["plan_feasibility"]["score"] == 1.0
    assert metrics["response_completeness"]["score"] == 0.8
    assert result["summary_metrics"]["overall"] == 0.9333
    assert {case["kind"] for case in result["cases"]} == {
        "tool_selection",
        "plan_feasibility",
        "response_completeness",
    }


@pytest.mark.asyncio
async def test_openjudge_low_scores_are_quality_failures(db_with_repo, stub_targets) -> None:
    db, repo = db_with_repo
    scorer = FakeScorer(
        {"tool_selection": 2.0, "plan_feasibility": 0.0, "response_completeness": 2.0}
    )

    result = await run_openjudge_eval(db, str(repo.id), scorer=scorer)

    assert result["status"] == "failed"
    assert result["quality_status"] == "failed"
    assert result["openjudge"]["case_errors"] == 0
    categories = {item["category"] for item in result["cases"][0]["diagnoses"]}
    assert "tool_selection" in categories


@pytest.mark.asyncio
async def test_openjudge_judge_failure_reports_error_not_quality(db_with_repo, stub_targets) -> None:
    db, repo = db_with_repo

    result = await run_openjudge_eval(db, str(repo.id), scorer=FakeScorer(error=OpenJudgeError("judge down")))

    assert result["status"] == "error"
    assert result["quality_status"] == "incomplete"
    assert result["evaluation_complete"] is False
    assert result["openjudge"]["case_errors"] == len(result["cases"])
    assert "OpenJudgeError" in result["cases"][0]["openjudge_errors"]["tool_selection"]
    assert result["cases"][0]["diagnoses"][0]["category"] == "judge_error"


@pytest.mark.asyncio
async def test_openjudge_judge_timeout_reports_error(db_with_repo, stub_targets, monkeypatch) -> None:
    db, repo = db_with_repo
    monkeypatch.setattr(settings, "openjudge_timeout_seconds", 0.01)

    result = await run_openjudge_eval(db, str(repo.id), scorer=FakeScorer(delay=0.2))

    assert result["status"] == "error"
    assert all("TimeoutError" in case["openjudge_errors"][case["kind"]] for case in result["cases"])
    assert result["summary_metrics"] == {"overall": None, "hard_gate_pass_rate": 1.0}


@pytest.mark.asyncio
async def test_openjudge_missing_field_is_mapping_error(db_with_repo, monkeypatch) -> None:
    db, repo = db_with_repo

    async def broken_payload(db_arg, repo_arg, case, agent_factory):
        raise OpenJudgeMappingError("ChatAgent 结果 tool-call 缺少字段 tool_calls")

    monkeypatch.setattr(openjudge_runner, "_tool_selection_payload", broken_payload)
    monkeypatch.setattr(openjudge_runner, "_plan_feasibility_payload", _fake_payload)
    monkeypatch.setattr(openjudge_runner, "_response_completeness_payload", _fake_payload)

    result = await run_openjudge_eval(db, str(repo.id), scorer=FakeScorer())

    assert result["status"] == "error"
    assert "OpenJudgeMappingError" in result["cases"][0]["openjudge_errors"]["tool_selection"]
    assert result["cases"][0]["hard_rules"]["passed"] is False


def test_tool_selection_mapper_requires_tool_name() -> None:
    case = OpenJudgeCase(id="tool-1", kind="tool_selection", target="chat", question="查一下 Issue")
    definitions = [{"name": "rag.search_similar_documents", "description": "搜索知识"}]

    with pytest.raises(OpenJudgeMappingError, match="tool_name"):
        _map_tool_selection(case, {"tool_calls": [{"arguments": {}}], "answer": "ok"}, definitions)

    with pytest.raises(OpenJudgeMappingError, match="tool_calls"):
        _map_tool_selection(case, {"answer": "ok"}, definitions)


def test_tool_selection_mapper_drops_nested_workflow_traces() -> None:
    case = OpenJudgeCase(id="tool-1", kind="tool_selection", target="chat", question="查一下 Issue")
    definitions = [{"name": "workflow.run_engineering_review", "description": "工作流"}]
    result = {
        "answer": "ok",
        "tool_calls": [
            {"tool_name": "workflow.run_engineering_review", "arguments": {}, "tool_use_id": "call_1"},
            {"tool_name": "workflow.repo_health", "arguments": {}, "tool_use_id": "call_1"},
            {"tool_name": "workflow.issue_analysis", "arguments": {}, "tool_use_id": "call_1"},
        ],
    }

    payload = _map_tool_selection(case, result, definitions)

    assert payload["tool_calls"] == [{"name": "workflow.run_engineering_review", "arguments": {}}]


def test_completeness_mapper_requires_every_required_field() -> None:
    output = {
        "summary": "登录失败",
        "conclusion": "进入开发",
        "conclusion_reason": "可复现",
        "category": "Bug",
        "priority": "P1",
        "complexity": "S",
        "suggested_owner": "alice",
        "owner_reason": "熟悉鉴权",
        "evidence": [],
        "checklist": [],
        "action_items": [],
    }

    with pytest.raises(OpenJudgeMappingError, match="confidence"):
        openjudge_runner._map_completeness("issue", output, "issue-7")


@pytest.mark.asyncio
async def test_openjudge_no_cases_and_judge_disabled(db_with_repo, stub_targets, monkeypatch) -> None:
    db, repo = db_with_repo
    monkeypatch.setattr(settings, "openjudge_max_cases", 0)
    empty = await run_openjudge_eval(db, str(repo.id), scorer=FakeScorer())
    assert empty["status"] == "no_cases"
    assert empty["quality_status"] == "not_evaluated"

    monkeypatch.setattr(settings, "openjudge_max_cases", 4)
    partial = await run_openjudge_eval(db, str(repo.id), run_judge=False, scorer=FakeScorer())
    assert partial["status"] == "partial"
    assert partial["quality_status"] == "partial"
    assert all(case["openjudge_skipped_reason"] == "judge_disabled" for case in partial["cases"])


def test_openjudge_cases_cover_repo_signals(db_with_repo) -> None:
    db, repo = db_with_repo

    cases, evaluation_set = openjudge_runner._build_cases(db, repo, None, 4)

    assert [case.kind for case in cases] == [
        "tool_selection",
        "plan_feasibility",
        "response_completeness",
        "response_completeness",
    ]
    assert {case.target for case in cases[2:]} == {"issue", "pull_request"}
    assert uuid.UUID(cases[2].target_id or "")
    assert evaluation_set["case_count"] == 4


def test_openjudge_cases_balance_multiple_issues_and_prs() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    repo = Repository(owner="local", name="multi", full_name="local/multi")
    db.add(repo)
    db.flush()
    for number in (1, 2, 3):
        db.add(Issue(repo_id=repo.id, number=number, title=f"issue {number}", body="b", state="open", labels=[]))
    for number in (8, 9):
        db.add(PullRequest(repo_id=repo.id, number=number, title=f"pr {number}", body="b", state="open"))
    db.commit()

    cases, evaluation_set = openjudge_runner._build_cases(db, repo, None, 7)

    assert [case.kind for case in cases[:2]] == ["tool_selection", "plan_feasibility"]
    assert [case.target for case in cases[2:]] == [
        "issue",
        "pull_request",
        "issue",
        "pull_request",
        "issue",
    ]
    assert [case.id for case in cases[2:]] == [
        "issue-completeness-3",
        "pr-completeness-9",
        "issue-completeness-2",
        "pr-completeness-8",
        "issue-completeness-1",
    ]
    assert evaluation_set["case_count"] == 7

    trimmed, _ = openjudge_runner._build_cases(db, repo, None, 4)
    assert [case.target for case in trimmed[2:]] == ["issue", "pull_request"]
    db.close()


class StubAgent:
    def _tool_description(self, name: str) -> str:
        return f"{name} 描述"


@pytest.mark.asyncio
async def test_tool_selection_payload_reads_chatagent_trace(db_with_repo, monkeypatch) -> None:
    db, repo = db_with_repo

    async def fake_run(db_arg, repo_arg, case, *, limit, agent_factory):
        return {
            "answer": "库存没有回滚",
            "agent_events": [
                {
                    "type": "agent_loop_start",
                    "available_tools": ["rag.search_similar_documents", "workspace.read_file"],
                }
            ],
            "tool_calls": [{"tool_name": "rag.search_similar_documents", "arguments": {"query": "库存"}}],
        }

    monkeypatch.setattr(openjudge_runner, "_run_chatagent_case", fake_run)
    case = OpenJudgeCase(id="tool-1", kind="tool_selection", target="chat", question="查一下库存问题")
    payload, response, calls = await openjudge_runner._tool_selection_payload(
        db, repo, case, lambda: StubAgent()
    )

    assert payload["query"] == "查一下库存问题"
    assert payload["tool_calls"] == [{"name": "rag.search_similar_documents", "arguments": {"query": "库存"}}]
    assert [item["name"] for item in payload["tool_definitions"]] == [
        "rag.search_similar_documents",
        "workspace.read_file",
    ]
    assert response == "库存没有回滚"
    assert calls[0]["tool_name"] == "rag.search_similar_documents"
