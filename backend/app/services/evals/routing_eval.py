"""Hold-out routing evaluation for ChatAgent (v2).

Deterministic layer only: expected/forbidden tool sets, call budget, refusal
keywords. No LLM judge yet, so the result is reproducible and cheap to re-run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any

from app.db.models import Repository
from app.db.session import SessionLocal
from app.services.agents.chat_agent import ChatAgent
from app.services.evals.ragas_runner import RagEvalCase, _run_chatagent_case

DATASET = Path(__file__).resolve().parents[3] / "evals" / "routing_holdout.jsonl"
CURRENT_WORKFLOW = (
    "工作流硬性规则：当用户明确要求多 Agent 审查，或要求决策结论（能不能合、是否阻塞、优先级、跨 Agent 综合判断）时，"
    "才调用 workflow.run_engineering_review。若用户只是盘点现状（有哪些未关闭 Issue、待审 PR、失败 CI），"
    "先调用 repo_health.generate_summary，拿到结果后直接回答，不要启动 workflow。"
    "工具预算：状态盘点最多 2 个工具、合计不超过 3 次调用；禁止用 workspace.* 查 Issue/PR/CI 状态；"
    "禁止对 attest_search_evidence 或 rag.search_similar_documents 重复调用超过 1 次。"
)
ORIGINAL_WORKFLOW = (
    "工作流硬性规则：如果用户请求多 Agent 审查、合并就绪度、当前阻塞、工程状态、跨 Agent 综合，"
    "或依赖 Issue、PR、CI 与仓库证据共同判断的决策，必须先调用 workflow.run_engineering_review，"
    "再给出最终建议。"
)


class OriginalRuleAgent(ChatAgent):
    def _tool_calling_user_prompt(self, state, *, aggressive: bool = False) -> str:  # type: ignore[override]
        text = super()._tool_calling_user_prompt(state, aggressive=aggressive)
        return text.replace(CURRENT_WORKFLOW, ORIGINAL_WORKFLOW)


@dataclass(frozen=True)
class Case:
    id: str
    intent: str
    question: str
    repo: str
    required_any: tuple[str, ...]
    allowed: tuple[str, ...]
    max_calls: int
    expect: str
    refuse_any: tuple[str, ...]


def load_cases(limit: int | None = None, only: str | None = None, intents: set[str] | None = None) -> list[Case]:
    cases: list[Case] = []
    for line in DATASET.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if only and item["intent"] != only:
            continue
        if intents and item["intent"] not in intents:
            continue
        cases.append(
            Case(
                id=item["id"],
                intent=item["intent"],
                question=item["question"],
                repo=item.get("repo", "default"),
                required_any=tuple(item.get("required_any") or ()),
                allowed=tuple(item.get("allowed") or ()),
                max_calls=int(item.get("max_calls") or 0),
                expect=item.get("expect", "answer"),
                refuse_any=tuple(item.get("refuse_any") or ()),
            )
        )
    return cases[:limit] if limit else cases


def wilson(passed: int, total: int) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    z = 1.96
    p = passed / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def evaluate(case: Case, calls: list[dict[str, Any]], answer: str) -> dict[str, Any]:
    names = [str(call.get("tool_name") or "") for call in calls]
    outside = sorted({name for name in names if name not in case.allowed})
    hit = sorted(set(names) & set(case.required_any))
    budget_ok = len(names) <= case.max_calls
    required_ok = bool(hit) if case.required_any else True
    answer_ok = len(answer.strip()) >= 20
    refusal_ok = True
    if case.expect == "refuse":
        refusal_ok = any(token in answer for token in case.refuse_any)
    passed = not outside and budget_ok and required_ok and answer_ok and refusal_ok
    reasons: list[str] = []
    if outside:
        reasons.append(f"越界工具 {outside}")
    if not budget_ok:
        reasons.append(f"调用 {len(names)}>{case.max_calls}")
    if not required_ok:
        reasons.append("未命中期望工具")
    if not answer_ok:
        reasons.append("回答为空")
    if not refusal_ok:
        reasons.append("未明确拒绝")
    return {
        "passed": passed,
        "reasons": reasons,
        "tools": names,
        "outside": outside,
        "calls": len(names),
        "budget_ok": budget_ok,
    }


def _resolve_repos(db) -> tuple[Repository, Repository]:
    default_repo = (
        db.query(Repository)
        .filter(Repository.full_name.like("%sample-api"))
        .order_by(Repository.created_at.asc())
        .first()
    ) or db.query(Repository).first()
    if default_repo is None:
        raise RuntimeError("数据库里没有可用的仓库，请先执行 seed_demo_data.py")
    empty_repo = db.query(Repository).filter(Repository.full_name == "course-demo/empty-repo").first()
    if empty_repo is None:
        empty_repo = Repository(
            owner="course-demo",
            name="empty-repo",
            full_name="course-demo/empty-repo",
            default_branch="main",
        )
        db.add(empty_repo)
        db.commit()
    return default_repo, empty_repo


async def run(agent_name: str, cases: list[Case]) -> dict[str, Any]:
    db = SessionLocal()
    default_repo, empty_repo = _resolve_repos(db)
    factory = OriginalRuleAgent if agent_name == "original" else ChatAgent
    rows: list[dict[str, Any]] = []
    try:
        for case in cases:
            repo = empty_repo if case.repo == "empty" else default_repo
            started = time.perf_counter()
            try:
                result = await _run_chatagent_case(
                    db,
                    repo,
                    RagEvalCase(id=case.id, question=case.question, reference=case.question, expected_source_ids=[]),
                    limit=5,
                    agent_factory=factory,
                )
                calls = [item for item in result.get("tool_calls") or [] if isinstance(item, dict)]
                answer = str(result.get("answer") or "")
                verdict = evaluate(case, calls, answer)
            except Exception as exc:  # noqa: BLE001 - record and continue the batch
                verdict = {"passed": False, "reasons": [f"{type(exc).__name__}: {exc}"], "tools": [], "outside": [], "calls": 0}
                answer = ""
            rows.append(
                {
                    "id": case.id,
                    "intent": case.intent,
                    "question": case.question,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                    "answer_preview": answer[:200],
                    **verdict,
                }
            )
    finally:
        db.close()
    return summarize(agent_name, rows)


def summarize(agent_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    passed = sum(1 for row in rows if row["passed"])
    low, high = wilson(passed, total)
    by_intent: dict[str, dict[str, Any]] = {}
    for row in rows:
        bucket = by_intent.setdefault(row["intent"], {"total": 0, "passed": 0, "calls": []})
        bucket["total"] += 1
        bucket["passed"] += 1 if row["passed"] else 0
        bucket["calls"].append(row["calls"])
    for bucket in by_intent.values():
        bucket["calls"].sort()
        bucket["pass_rate"] = round(bucket["passed"] / bucket["total"], 4)
        bucket["p50_calls"] = bucket["calls"][len(bucket["calls"]) // 2]
        bucket["p95_calls"] = bucket["calls"][max(0, math.ceil(0.95 * len(bucket["calls"])) - 1)]
        bucket.pop("calls")
    calls = sorted(row["calls"] for row in rows)
    latencies = sorted(row["latency_ms"] for row in rows)
    return {
        "agent": agent_name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "ci95": [round(low, 4), round(high, 4)],
        "forbidden_rate": round(sum(1 for row in rows if row["outside"]) / total, 4) if total else 0.0,
        "p50_calls": calls[len(calls) // 2] if calls else 0,
        "p95_calls": calls[max(0, math.ceil(0.95 * len(calls)) - 1)] if calls else 0,
        "p50_latency_ms": round(median(latencies), 1) if latencies else 0,
        "by_intent": by_intent,
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["current", "original"], default="current")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", default=None)
    parser.add_argument("--intents", default=None, help="逗号分隔的意图列表，如 status,single,decision")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    result = asyncio.run(
        run(
            args.agent,
            load_cases(args.limit, args.only, {item.strip() for item in (args.intents or "").split(",") if item.strip()}),
        )
    )
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) if args.out else Path("out") / "evals" / stamp
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"routing-{args.agent}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("agent", "total", "passed", "pass_rate", "ci95", "forbidden_rate", "p50_calls", "p95_calls")}, ensure_ascii=False))
    print("saved", path)


if __name__ == "__main__":
    main()
