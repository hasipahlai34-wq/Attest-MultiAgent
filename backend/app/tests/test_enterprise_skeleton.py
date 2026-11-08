"""可观测性骨架的契约测试：错误契约、请求追踪、指标与健康检查。

不涉及认证与多用户——当前定位是单人本地演示项目，这两项目标已明确不做。
"""

import os
import uuid

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

from app.db.models import Base, Repository
from app.db.session import get_db
from app.main import create_app


def _client():
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False, autocommit=False)()
    app = create_app()

    def _override():
        return session

    app.dependency_overrides[get_db] = _override
    return TestClient(app), session


def test_not_found_uses_unified_error_contract():
    client, _session = _client()

    response = client.get(f"/api/action-drafts/{uuid.uuid4()}")

    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "NOT_FOUND"
    assert body["error"]["request_id"]
    assert body["detail"] == "未找到安全草稿"


def test_validation_error_is_machine_readable():
    client, _session = _client()

    response = client.post("/api/action-drafts", json={"repo_id": "not-a-uuid", "content": "x"})

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "VALIDATION_FAILED"
    assert body["error"]["details"]["errors"]


def test_request_id_is_propagated_from_client_header():
    client, session = _client()
    repo = Repository(owner="attest", name="demo", full_name=f"attest/demo-{uuid.uuid4().hex[:8]}")
    session.add(repo)
    session.commit()

    response = client.get("/api/repos", headers={"X-Request-ID": "trace-local-001"})

    assert response.headers["X-Request-ID"] == "trace-local-001"
    assert response.status_code == 200
    assert any(item["id"] == str(repo.id) for item in response.json())


def test_request_id_is_generated_when_absent():
    client, _session = _client()

    response = client.get("/health")

    assert len(response.headers["X-Request-ID"]) == 32


def test_metrics_endpoint_exposes_prometheus_text():
    client, _session = _client()

    response = client.get("/api/ops/metrics")

    assert response.status_code == 200
    assert "# TYPE attest_workflow_runs_total counter" in response.text
    assert "attest_audit_failures_24h" in response.text


def test_deep_health_reports_component_status():
    client, _session = _client()

    response = client.get("/api/ops/health/deep")

    assert response.status_code == 200
    body = response.json()
    assert body["checks"]["database"] == "ok"
    assert body["status"] in {"ok", "degraded"}
