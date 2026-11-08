"""上下文压缩效果实测：量化 token 降幅与关键事实保留率。

用法（后端不必启动，纯函数级实验）：
    python scripts/eval_context_compression.py

两个场景：
A. 工具结果投影（ChatAgent._runtime_tool_context）：30 条大工具结果 → 有界 prompt 视图
B. 历史消息压缩（compress_messages）：40 条长消息 → 预算内最近消息
每个场景在原文中均匀埋入 10 个唯一事实标记，压缩后按标记统计保留率。
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

from app.services.agents.chat_agent import ChatAgent  # noqa: E402
from app.services.context_compression import (  # noqa: E402
    compress_messages,
    default_context_budget,
    estimate_tokens,
)

FACTS = [f"FACT-{index:02d}" for index in range(1, 11)]
FILLER = "接口返回结构包含状态码、耗时与错误上下文，需要结合日志进一步核对。" * 6


def build_text(seed: str = "") -> str:
    """约 1000 字，10 个事实均匀分布。"""
    blocks = []
    for index, fact in enumerate(FACTS):
        blocks.append(f"{fact}：{seed}结论 {index + 1}，涉及模块 {seed}-{index + 1}。{FILLER}")
    return "\n".join(blocks)


def kept_facts(payload: str) -> int:
    return sum(1 for fact in FACTS if fact in payload)


def report(label: str, raw_text: str, compacted_text: str, recent_text: str = "") -> dict:
    raw_tokens = estimate_tokens(raw_text)
    compacted_tokens = estimate_tokens(compacted_text)
    reduction = (raw_tokens - compacted_tokens) / raw_tokens * 100 if raw_tokens else 0.0
    kept = kept_facts(compacted_text)
    recent_facts = [fact for fact in FACTS if fact in recent_text] if recent_text else []
    kept_recent = sum(1 for fact in recent_facts if fact in compacted_text)
    print(
        f"{label:<14} 原始 {raw_tokens:>6} tok → 压缩后 {compacted_tokens:>6} tok "
        f"（-{reduction:5.1f}%）　全局保留 {kept}/{len(FACTS)}　近期保留 {kept_recent}/{len(recent_facts) or 0}"
    )
    return {
        "raw_tokens": raw_tokens,
        "compacted_tokens": compacted_tokens,
        "reduction_pct": round(reduction, 2),
        "facts_kept": kept,
        "facts_total": len(FACTS),
        "recent_facts_kept": kept_recent,
        "recent_facts_total": len(recent_facts),
    }


def scenario_observations() -> dict:
    observations = [
        {
            "tool_name": "rag.search_similar_documents",
            "arguments": {"question": f"第 {index} 个查询"},
            "observation": {"answer": build_text(f"TOOL{index}")},
        }
        for index in range(1, 31)
    ]
    state = {"observations": observations, "agent_events": []}
    agent = ChatAgent(model=object())  # 只为调用投影方法，不触发任何模型调用
    projected, _events, stats = agent._runtime_tool_context(state)
    recent = json.dumps(observations[-3:], ensure_ascii=False)  # 投影设计目标是保住近期证据
    return report(
        "工具结果投影",
        json.dumps(observations, ensure_ascii=False),
        json.dumps(projected, ensure_ascii=False),
        recent,
    ) | {"projection_stats": stats}


def scenario_history() -> dict:
    messages: list[dict] = []
    for index in range(20):
        messages.append({"role": "user", "content": f"第 {index + 1} 轮：{build_text(f'U{index}')}"})
        messages.append({"role": "assistant", "content": f"第 {index + 1} 轮回答：{build_text(f'A{index}')}"})
    budget = default_context_budget().available_prompt_tokens
    kept, stats = compress_messages(messages, budget)
    recent = json.dumps(messages[-4:], ensure_ascii=False)  # 压缩保留最近的完整消息
    result = report(
        "历史消息压缩",
        json.dumps(messages, ensure_ascii=False),
        json.dumps(kept, ensure_ascii=False),
        recent,
    )
    return result | {"budget_tokens": budget, "compress_stats": stats}


def main() -> int:
    print(f"模型上下文预算：{default_context_budget().available_prompt_tokens} tok\n")
    result = {"observations": scenario_observations(), "history": scenario_history()}
    output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".eval", "context-compression.json")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(f"\n明细已写入 {os.path.normpath(output)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
