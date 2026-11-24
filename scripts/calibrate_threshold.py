"""Calibrate the abstention score from the evaluation cases.

Usage: uv run python scripts/calibrate_threshold.py
Reads cases.json: expect_retrieval=True -> in scope, False -> out of scope.
"""

import json
import sys
from itertools import pairwise
from pathlib import Path

from insurance_rag_assistant.config import ARTIFACTS_DIR, EVALUATION_CASES_FILE
from insurance_rag_assistant.evaluation.evaluator import load_evaluation_cases
from insurance_rag_assistant.retrieval.search import SemanticSearchService


def best_threshold(in_scores: list[float], out_scores: list[float]) -> dict[str, float]:
    """Pick the threshold maximising balanced accuracy (TPR + TNR) / 2."""
    candidates = sorted(set(in_scores + out_scores))
    best = {"threshold": 0.0, "balanced_accuracy": -1.0, "tpr": 0.0, "tnr": 0.0}
    for lo, hi in pairwise(candidates):
        t = (lo + hi) / 2
        tpr = sum(s >= t for s in in_scores) / len(in_scores)
        tnr = sum(s < t for s in out_scores) / len(out_scores)
        ba = (tpr + tnr) / 2
        if ba > best["balanced_accuracy"]:
            best = {"threshold": t, "balanced_accuracy": ba, "tpr": tpr, "tnr": tnr}
    return best


def main() -> None:
    cases = load_evaluation_cases(EVALUATION_CASES_FILE)
    service = SemanticSearchService()
    rows = []
    try:
        for case in cases:
            vector = service.embedder.embed_query(case.question)
            hits = service.vector_store.search(vector, case.filters, 5, -1.0)
            top1 = hits[0].score if hits else 0.0
            top2 = hits[1].score if len(hits) > 1 else 0.0
            rows.append(
                {
                    "case_id": case.case_id,
                    "in_scope": case.expect_retrieval,
                    "top1": top1,
                    "margin": top1 - top2,
                }
            )
    finally:
        service.close()

    in_scores = [r["top1"] for r in rows if r["in_scope"]]
    out_scores = [r["top1"] for r in rows if not r["in_scope"]]
    if not in_scores or not out_scores:
        sys.exit("Need at least one in-scope and one out-of-scope case.")

    report = {
        "n_in_scope": len(in_scores),
        "n_out_of_scope": len(out_scores),
        "in_scope_min": min(in_scores),
        "out_scope_max": max(out_scores),
        "separable": min(in_scores) > max(out_scores),
        "recommended": best_threshold(in_scores, out_scores),
        "rows": rows,
    }
    out = Path(ARTIFACTS_DIR) / "threshold_calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    for r in sorted(rows, key=lambda r: r["top1"]):
        flag = "IN " if r["in_scope"] else "OUT"
        print(f"{flag} top1={r['top1']:.3f} margin={r['margin']:.3f}  {r['case_id']}")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
