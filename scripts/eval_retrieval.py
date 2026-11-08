"""检索质量评测：在固定评测集上扫描 Top-K，输出 Precision / Recall / F1。

用法（后端需已运行）：
    python scripts/eval_retrieval.py --repo-id <uuid> [--top-k 2,3,4,5]

判定口径：
- 结果按「文档」去重，不按切片计数：knowledge_file 取文件名（`a.md · 章节` → `a.md`），
  issue / pull_request / workflow_run 取记录标题。
- Precision@K = 命中相关文档数 / 实际召回的不同文档数
- Recall@K    = 命中相关文档数 / 该题标注的相关文档总数
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import httpx

BASE_URL = os.environ.get("ATTEST_BASE_URL", "http://127.0.0.1:8000")

# 评测集：question -> 相关文档（按上面的判定口径写 key）
EVAL_CASES: list[dict] = [
    {
        "question": "token 过期时登录接口返回了什么错误？",
        "relevant": ["Login API returns 500 when token expires"],
    },
    {
        "question": "Issue #1 指派给谁处理？",
        "relevant": ["Login API returns 500 when token expires"],
    },
    {
        "question": "GitHub workflow 同步功能需要实现什么？",
        "relevant": ["Add GitHub workflow sync"],
    },
    {
        "question": "没有 GitHub token 时怎么在本地启动演示？",
        "relevant": ["Document local demo mode", "README.md"],
    },
    {
        "question": "PR #8 改动了哪些文件？",
        "relevant": ["Refactor GitHub sync service"],
    },
    {
        "question": "PR #8 的评审意见提了哪些要求？",
        "relevant": ["Refactor GitHub sync service"],
    },
    {
        "question": "CI 中失败的测试用例和失败原因是什么？",
        "relevant": ["backend-test"],
    },
    {
        "question": "Agent 工作流的执行流程是怎样的？",
        "relevant": ["agent_workflow.md"],
    },
    {
        "question": "数据库有哪些核心表？",
        "relevant": ["database_schema.md"],
    },
    {
        "question": "如何接入一个 GitHub 仓库？",
        "relevant": ["README.md"],
    },
    {
        "question": "知识库索引出问题后如何重建？",
        "relevant": ["README.md"],
    },
]


def doc_key(result: dict) -> str:
    title = str(result.get("title") or "")
    if str(result.get("source_type") or "") == "knowledge_file":
        return title.split(" · ")[0].strip()
    return title.strip()


def search(client: httpx.Client, repo_id: str, question: str, top_k: int) -> list[dict]:
    response = client.post(
        f"{BASE_URL}/api/rag/{repo_id}/search",
        json={"question": question, "top_k": top_k},
    )
    response.raise_for_status()
    return response.json().get("results") or []


def set_retrieval_method(client: httpx.Client, repo_id: str, method: str, rerank: bool) -> None:
    response = client.patch(
        f"{BASE_URL}/api/rag/repositories/{repo_id}/config",
        json={"retrieval_method": method, "rerank_enabled": rerank},
    )
    response.raise_for_status()


def evaluate(client: httpx.Client, repo_id: str, top_k: int) -> dict:
    rows = []
    for case in EVAL_CASES:
        results = search(client, repo_id, case["question"], top_k)
        seen: list[str] = []
        for item in results:
            key = doc_key(item)
            if key and key not in seen:
                seen.append(key)
        relevant = set(case["relevant"])
        hits = [key for key in seen if key in relevant]
        precision = len(hits) / len(seen) if seen else 0.0
        recall = len(hits) / len(relevant) if relevant else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        rows.append(
            {
                "question": case["question"],
                "returned": seen,
                "hits": hits,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }
        )
    n = len(rows)
    return {
        "top_k": top_k,
        "precision": round(sum(r["precision"] for r in rows) / n, 4),
        "recall": round(sum(r["recall"] for r in rows) / n, 4),
        "f1": round(sum(r["f1"] for r in rows) / n, 4),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--top-k", default="2,3,4,5")
    parser.add_argument("--methods", default="hybrid", help="逗号分隔：hybrid,vector,fulltext")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    top_ks = [int(value) for value in args.top_k.split(",") if value.strip()]
    methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    # 本地回环不能走 http_proxy，否则请求被代理吞掉。
    client = httpx.Client(trust_env=False, timeout=120.0)

    health = client.get(f"{BASE_URL}/health")
    health.raise_for_status()

    results: dict[str, list[dict]] = {}
    for method in methods:
        set_retrieval_method(client, args.repo_id, method, rerank=True)
        results[method] = [evaluate(client, args.repo_id, top_k) for top_k in top_ks]

    print(f"评测集：{len(EVAL_CASES)} 题　（Precision / Recall / F1）")
    header = "方法".ljust(10) + "".join(f"K={k}".rjust(22) for k in top_ks)
    print(header)
    for method in methods:
        cells = "".join(
            f"{item['precision']:.2f}/{item['recall']:.2f}/{item['f1']:.2f}".rjust(22)
            for item in results[method]
        )
        print(method.ljust(10) + cells)

    # 恢复默认配置，避免影响后续使用
    set_retrieval_method(client, args.repo_id, "hybrid", rerank=True)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump({"cases": len(EVAL_CASES), "results": results}, handle, ensure_ascii=False, indent=2)
        print(f"\n明细已写入 {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
