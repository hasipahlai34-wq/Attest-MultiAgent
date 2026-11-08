from typing import Any

from app.schemas.analysis import SafetyOutput
from app.services.agents.base import BaseAgent
from app.services.safety_policy import classify_action


class SafetyAgent(BaseAgent):
    name = "safety_agent"
    description = "检查某个操作是否需要人工确认。"
    available_tools = ["classify_action_risk"]

    async def run(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        decision = classify_action(
            action=str(input_data.get("action") or input_data.get("tool_name") or ""),
            scope=input_data.get("scope") or input_data.get("target_scope"),
            effect=input_data.get("effect") or input_data.get("write_effect"),
        )
        return SafetyOutput(
            requires_confirmation=decision.requires_confirmation,
            risk_level=decision.risk_level,
            reason=decision.reason,
            draft=input_data.get("draft"),
        ).model_dump()
