"""运维端点：指标导出与深度健康检查。

指标直接从既有业务表聚合，不引入额外时序库；单机/小规模足够，规模化时把
``collect_metrics`` 换成 Prometheus client 的 Counter/Gauge 即可。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse
from sqlalchemy import func, text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import ActionDraft, AgentRun, AgentWorkflowRun, AuditLog, User
from app.db.session import get_db
from app.services.permissions import require_permission

router = APIRouter()


def _percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(ratio * (len(ordered) - 1)))))
    return float(ordered[index])


def collect_metrics(db: Session) -> dict[str, object]:
    workflow_status = dict(
        db.query(AgentWorkflowRun.status, func.count()).group_by(AgentWorkflowRun.status).all()
    )
    durations = [
        float((row.metrics or {}).get("duration_ms") or 0)
        for row in db.query(AgentWorkflowRun).order_by(AgentWorkflowRun.created_at.desc()).limit(200).all()
    ]
    durations = [value for value in durations if value > 0]
    draft_status = dict(db.query(ActionDraft.status, func.count()).group_by(ActionDraft.status).all())
    agent_status = dict(db.query(AgentRun.status, func.count()).group_by(AgentRun.status).all())
    since = datetime.now(timezone.utc) - timedelta(hours=24)
    audit_failures = (
        db.query(func.count()).select_from(AuditLog).filter(AuditLog.created_at >= since, AuditLog.status == "failed").scalar()
        or 0
    )
    return {
        "workflow_status": workflow_status,
        "workflow_duration_p50": _percentile(durations, 0.50),
        "workflow_duration_p95": _percentile(durations, 0.95),
        "draft_status": draft_status,
        "agent_status": agent_status,
        "audit_failures_24h": int(audit_failures),
    }


def render_prometheus(metrics: dict[str, object]) -> str:
    lines: list[str] = []

    def emit(name: str, help_text: str, kind: str, samples: list[tuple[dict[str, str], float]]) -> None:
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} {kind}")
        for labels, value in samples:
            label_text = ",".join(f'{key}="{val}"' for key, val in labels.items())
            suffix = f"{{{label_text}}}" if label_text else ""
            lines.append(f"{name}{suffix} {value}")

    emit(
        "attest_workflow_runs_total",
        "Agent 工作流运行数，按状态分组。",
        "counter",
        [({"status": str(key)}, float(value)) for key, value in (metrics["workflow_status"] or {}).items()],
    )
    emit(
        "attest_workflow_duration_ms",
        "最近 200 次工作流耗时分位（毫秒）。",
        "gauge",
        [({"quantile": "0.5"}, float(metrics["workflow_duration_p50"])), ({"quantile": "0.95"}, float(metrics["workflow_duration_p95"]))],
    )
    emit(
        "attest_action_drafts",
        "待执行/已执行草稿数量，按状态分组。",
        "gauge",
        [({"status": str(key)}, float(value)) for key, value in (metrics["draft_status"] or {}).items()],
    )
    emit(
        "attest_agent_runs_total",
        "对话 Agent 运行数，按状态分组。",
        "counter",
        [({"status": str(key)}, float(value)) for key, value in (metrics["agent_status"] or {}).items()],
    )
    emit(
        "attest_audit_failures_24h",
        "最近 24 小时失败的审计动作数。",
        "gauge",
        [({}, float(metrics["audit_failures_24h"]))],
    )
    return "\n".join(lines) + "\n"


@router.get("/metrics", response_class=PlainTextResponse)
async def metrics(
    db: Session = Depends(get_db),
    _user: User = Depends(require_permission("metrics:read")),
) -> PlainTextResponse:
    if not settings.metrics_enabled:
        return PlainTextResponse("# metrics disabled\n", status_code=200)
    return PlainTextResponse(render_prometheus(collect_metrics(db)))


@router.get("/health/deep")
async def deep_health(
    db: Session = Depends(get_db),
    _user: User = Depends(require_permission("metrics:read")),
) -> dict:
    checks: dict[str, str] = {}
    try:
        db.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:
        checks["database"] = f"error: {exc}"
    checks["llm"] = "configured" if settings.llm_api_key else "missing_key"
    try:
        from app.services.rag.vector_store import milvus_runtime_status

        status = milvus_runtime_status()
        checks["milvus"] = str(status.get("status") or status)
    except Exception as exc:
        checks["milvus"] = f"error: {exc}"
    degraded = any(value not in {"ok", "configured", "disabled"} for value in checks.values())
    return {"status": "degraded" if degraded else "ok", "checks": checks}
