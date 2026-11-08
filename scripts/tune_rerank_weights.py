"""重排权重扫描：在固定评测集上标定 rerank 各分项权重。

用法（后端不必启动）：
    python scripts/tune_rerank_weights.py

背景：hybrid 模式下最终排序由 heuristic rerank 决定，而词面类权重
（keyword 0.16 + coverage 0.17 + phrase 0.12 + source 0.08 = 0.53）远高于向量 0.20，
长文档（切片多、词覆盖高）容易被顶到前排。这里扫描几组配比找更优解。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import app.services.rag.rerank as rerank  # noqa: E402
import app.services.rag.retrieval as retrieval  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from app.services.rag.embeddings import embed_knowledge_query  # noqa: E402
from app.core.config import settings  # noqa: E402
from eval_retrieval import EVAL_CASES, doc_key  # noqa: E402

REPO_ID = os.environ.get("ATTEST_TUNE_REPO_ID", "")
if not REPO_ID:
    raise SystemExit("请先设置 ATTEST_TUNE_REPO_ID 为要评测的仓库 UUID")
TOP_KS = (2, 3, 5)

PRESETS: dict[str, dict[str, float]] = {
    "当前(默认)": {"base": 0.31, "vector": 0.50, "keyword": 0.06, "coverage": 0.06,
                "title": 0.04, "path": 0.02, "phrase": 0.04, "source": 0.03},
    "原版权重": {"base": 0.31, "vector": 0.20, "keyword": 0.16, "coverage": 0.17,
              "title": 0.04, "path": 0.02, "phrase": 0.12, "source": 0.08},
    "弱化coverage-保精确": {"base": 0.31, "vector": 0.35, "keyword": 0.16, "coverage": 0.08,
                       "title": 0.10, "path": 0.08, "phrase": 0.08, "source": 0.06},
    "弱化coverage-强精确": {"base": 0.31, "vector": 0.30, "keyword": 0.18, "coverage": 0.06,
                       "title": 0.12, "path": 0.10, "phrase": 0.08, "source": 0.06},
}


def score_rows(rows: list[list[dict]]) -> tuple[float, float, float]:
    precision_sum = recall_sum = f1_sum = 0.0
    for case, row in zip(EVAL_CASES, rows):
        seen: list[str] = []
        for item in row:
            key = doc_key(item)
            if key and key not in seen:
                seen.append(key)
        relevant = set(case["relevant"])
        hits = [key for key in seen if key in relevant]
        precision = len(hits) / len(seen) if seen else 0.0
        recall = len(hits) / len(relevant) if relevant else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        precision_sum += precision
        recall_sum += recall
        f1_sum += f1
    n = len(rows)
    return precision_sum / n, recall_sum / n, f1_sum / n


def main() -> int:
    db = SessionLocal()
    try:
        print(f"预计算 {len(EVAL_CASES)} 题的 query embedding …")
        embeddings = [
            embed_knowledge_query(
                case["question"],
                provider=settings.embedding_provider,
                model=settings.embedding_model,
                dimensions=settings.embedding_dimensions,
            )
            for case in EVAL_CASES
        ]

        print("".ljust(18) + "".join(f"K={k}".rjust(22) for k in TOP_KS))
        summary = []
        for label, weights in PRESETS.items():
            rerank.RERANK_WEIGHTS.update(weights)
            cells = []
            f1_by_k = {}
            for top_k in TOP_KS:
                rows = [
                    retrieval.search_similar_documents(
                        db, REPO_ID, case["question"],
                        limit=top_k, retrieval_method="hybrid", apply_rerank=True,
                        query_embedding=embedding,
                    )
                    for case, embedding in zip(EVAL_CASES, embeddings)
                ]
                precision, recall, f1 = score_rows(rows)
                f1_by_k[top_k] = f1
                cells.append(f"{precision:.2f}/{recall:.2f}/{f1:.2f}".rjust(22))
            print(label.ljust(18) + "".join(cells))
            summary.append((label, weights, f1_by_k))

        best = max(summary, key=lambda item: item[2][5])
        default = summary[0]
        print(f"\nF1@5 最优：{best[0]} → {best[2][5]:.4f}（默认 {default[2][5]:.4f}）")
        print(f"最优权重：{best[1]}")
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
