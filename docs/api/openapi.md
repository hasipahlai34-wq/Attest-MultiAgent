# Attest AI API

启动后端后访问 `http://localhost:8000/docs` 查看交互式 OpenAPI。

## Core Endpoints

- `POST /api/repos/connect`
- `GET /api/repos`
- `POST /api/repos/{repo_id}/sync`
- `GET /api/repos/{repo_id}/sync-status`
- `GET /api/repos/{repo_id}/issues`
- `POST /api/issues/{issue_id}/analyze`
- `GET /api/issues/{issue_id}/similar`
- `GET /api/repos/{repo_id}/pull-requests`
- `POST /api/pull-requests/{pr_id}/analyze`
- `POST /api/pull-requests/{pr_id}/review-checklist`
- `GET /api/repos/{repo_id}/workflow-runs`
- `GET /api/workflow-runs/{run_id}/failed-jobs`
- `POST /api/workflow-runs/{run_id}/analyze`
- `POST /api/chat`
- `POST /api/search`
- `POST /api/reports/weekly`
- `POST /api/evals/run`
- `GET /api/ops/metrics`
- `GET /api/ops/health/deep`

## Error Contract

所有失败响应结构一致，`detail` 为人读信息，`error` 为机器可读信息：

```json
{
  "detail": "未找到安全草稿",
  "error": {
    "code": "NOT_FOUND",
    "message": "未找到安全草稿",
    "request_id": "b6e843b391fb43bebb6cd39ae99ef191",
    "details": {}
  }
}
```

`code` 取值：`AUTH_REQUIRED`、`INVALID_CREDENTIAL`、`FORBIDDEN`、`NOT_FOUND`、`CONFLICT`、`VALIDATION_FAILED`、`RATE_LIMITED`、`UPSTREAM_UNAVAILABLE`、`INTERNAL`。
校验失败时 `details.errors` 为 FastAPI 原始错误列表。

## Request Tracing

每个响应都带 `X-Request-ID`；请求头传入则透传，否则服务端生成 32 位 hex。该值同时出现在日志与错误体中。

