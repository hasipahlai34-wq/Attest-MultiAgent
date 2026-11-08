"""Hybrid 融合权重扫描：在固定评测集上找最优 向量/关键词 权重配比。

用法（后端不必启动，直接读库 + 打 embedding 接口）：
    python scripts/tune_fusion_weights.py

原理：直接 monkeypatch `retrieval.VECTOR_FUSION_WEIGHT` / `KEYWORD_FUSION_WEIGHT`，
绕过 HTTP 与进程重启，逐个配比算 Precision / Recall / F1。
评测集与判定口径复用 eval_retrieval.py。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import app.services.rag.retrieval as retrieval  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from app.services.rag.embeddings import embed_knowledge_query  # noqa: E402
from app.core.config import settings  # noqa: E402
from eval_retrieval import EVAL_CASES, doc_key  # noqa: E402

REPO_ID = os.environ.get("ATTEST_TUNE_REPO_ID", "")
if not REPO_ID:
    raise SystemExit("请先设置 ATTEST_TUNE_REPO_ID 为要评测的仓库 UUID")
TOP_KS = (2, 3, 5)
# 权重对之和保持 0.92（= 原 0.62 + 0.30），dual_recall_bonus 0.08 不变
COMBOS = [
    (0.92, 0.00),
    (0.82, 0.10),
    (0.72, 0.20),
    (0.62, 0.30),  # 项目默认
    (0.52, 0.40),
    (0.42, 0.50),
    (0.30, 0.62),
]


def score_rows(rows: list[dict]) -> tuple[float, float, float]:
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

        header = "vector/keyword".ljust(16) + "".join(f"K={k}".rjust(22) for k in TOP_KS)
        print(header)
        results = []
        for vector_weight, keyword_weight in COMBOS:
            retrieval.VECTOR_FUSION_WEIGHT = vector_weight
            retrieval.KEYWORD_FUSION_WEIGHT = keyword_weight
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
            label = f"{vector_weight:.2f}/{keyword_weight:.2f}"
            print(label.ljust(16) + "".join(cells))
            results.append((vector_weight, keyword_weight, f1_by_k))

        best = max(results, key=lambda item: item[2][5])
        print(f"\nF1@5 最优组合：vector={best[0]:.2f} / keyword={best[1]:.2f} → F1@5={best[2][5]:.4f}")
        default = next(item for item in results if item[0] == 0.62)
        print(f"默认组合 0.62/0.30 → F1@5={default[2][5]:.4f}")
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
