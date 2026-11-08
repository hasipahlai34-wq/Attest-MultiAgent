"""只读 Agent 能力自述端点。

外部安全扫描器（如 AI-Infra-Guard agent-scan）需要一份快速、只读的能力清单，
否则只能靠对话去套系统提示词——既慢、又会诱发真实的提示注入风险。

本端点只返回静态元数据：工具名、读写属性、作用域、防护策略摘要。
不触发模型推理、不访问数据库、不返回任何密钥或环境变量取值。
"""

from fastapi import APIRouter

from app.core.config import settings
from app.services.agents.chat_agent import TOOL_DESCRIPTIONS
from app.services.safety_policy import (
    ActionType,
    POLICY_RULES,
    TargetScope,
    WriteEffect,
)

router = APIRouter()

# 工具读写属性：read=只读；guarded_write=写操作但只落待确认草稿；workflow=多 Agent 编排
TOOL_ACCESS: dict[str, tuple[str, str]] = {
    "attest_search_evidence": ("read", "current_repo"),
    "attest_get_thread_context": ("read", "current_repo"),
    "attest_read_session_events": ("read", "current_repo"),
    "workspace.list_files": ("read", "current_repo_checkout"),
    "workspace.read_file": ("read", "current_repo_checkout"),
    "workspace.search_code": ("read", "current_repo_checkout"),
    "repo_health.generate_summary": ("read", "current_repo"),
    "rag.search_similar_documents": ("read", "current_repo"),
    "issue_agent.analyze_issue": ("read", "current_repo"),
    "pr_review_agent.analyze_pr": ("read", "current_repo"),
    "ci_debug_agent.analyze_workflow_run": ("read", "current_repo"),
    "report_agent.generate_weekly_report": ("read", "current_repo"),
    "safety_agent.classify_action_risk": ("guarded_write", "current_repo"),
    "workflow.run_engineering_review": ("workflow", "current_repo"),
}


@router.get("/agent/capabilities")
async def agent_capabilities() -> dict:
    """返回 Agent 的能力清单与防护边界（只读、静态、无敏感值）。"""
    return {
        "agent": {
            "name": "Attest ChatAgent",
            "type": "code_review_agent",
            "runtime_model": settings.llm_model,
            "transport": "http_json",
        },
        "capabilities": [
            {
                "name": name,
                "access": TOOL_ACCESS.get(name, ("read", "current_repo"))[0],
                "scope": TOOL_ACCESS.get(name, ("read", "current_repo"))[1],
                "description": TOOL_DESCRIPTIONS.get(name, "Attest 工具。"),
            }
            for name in TOOL_DESCRIPTIONS
        ],
        "guardrails": {
            "write_actions": "draft_only",
            "write_approval": "pending_confirmation",
            "default_deny_scopes": [
                TargetScope.OTHER_REPO.value,
                TargetScope.EXTERNAL.value,
                TargetScope.UNKNOWN.value,
            ],
            "policy_rule_count": len(POLICY_RULES),
            "action_types": [item.value for item in ActionType],
            "target_scopes": [item.value for item in TargetScope],
            "write_effects": [item.value for item in WriteEffect],
        },
        "data_handling": {
            "secrets": "environment_variables_only_never_returned",
            "cross_repo_access": "blocked_by_scope_check",
            "system_prompt_disclosure": "refused",
            "credential_disclosure": "refused",
        },
    }
