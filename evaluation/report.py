"""
Aggregates raw per-question eval results into a report worth showing
someone else. Two outputs: a full JSON dump (for your own debugging /
re-analysis) and a markdown summary (for actually sharing/presenting —
e.g. to the university official).
"""
import json
from collections import defaultdict
from datetime import datetime, timezone


def _safe_mean(values):
    clean = [v for v in values if v is not None]
    return round(sum(clean) / len(clean), 2) if clean else None


def build_report(results: list[dict]) -> dict:
    """
    `results` is a list of per-question dicts, each expected to have:
      id, category, question, reference_answer, generated_answer,
      retrieval_keyword_score, retrieval_semantic_score,
      correctness, faithfulness, judge_reasoning, is_out_of_scope_check
    """
    by_category = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r)

    category_summary = {}
    for category, items in by_category.items():
        category_summary[category] = {
            "count": len(items),
            "avg_retrieval_keyword_score": _safe_mean(
                [i["retrieval_keyword_score"] for i in items]
            ),
            "avg_retrieval_semantic_score": _safe_mean(
                [i["retrieval_semantic_score"] for i in items]
            ),
            "avg_correctness": _safe_mean([i["correctness"] for i in items]),
            "avg_faithfulness": _safe_mean([i["faithfulness"] for i in items]),
        }

    # Flag anything worth a human looking at directly, rather than
    # burying it in averages. Thresholds are deliberately conservative —
    # better to over-flag for review than let a real failure hide inside
    # a decent-looking average.
    flagged = [
        r for r in results
        if (r["correctness"] is not None and r["correctness"] <= 3)
        or (r["faithfulness"] is not None and r["faithfulness"] <= 3)
        or (r["retrieval_semantic_score"] is not None and r["retrieval_semantic_score"] < 0.5)
    ]

    judge_failures = [r for r in results if r.get("error") and r["correctness"] is None]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_questions": len(results),
        "judge_failures_count": len(judge_failures),
        "overall": {
            "avg_correctness": _safe_mean([r["correctness"] for r in results]),
            "avg_faithfulness": _safe_mean([r["faithfulness"] for r in results]),
            "avg_retrieval_keyword_score": _safe_mean(
                [r["retrieval_keyword_score"] for r in results]
            ),
            "avg_retrieval_semantic_score": _safe_mean(
                [r["retrieval_semantic_score"] for r in results]
            ),
        },
        "by_category": category_summary,
        "flagged_for_review": flagged,
        "all_results": results,
    }


def write_json_report(report: dict, path: str = "eval_results.json"):
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)


def write_markdown_report(report: dict, path: str = "eval_report.md"):
    lines = [
        f"# RAG Pipeline Evaluation Report",
        f"",
        f"Generated: {report['generated_at']}",
        f"Total questions evaluated: {report['total_questions']}",
        f"",
        f"## Overall Scores",
        f"",
        f"| Metric | Score |",
        f"|---|---|",
        f"| Correctness (1-5, judge) | {report['overall']['avg_correctness']} |",
        f"| Faithfulness (1-5, judge) | {report['overall']['avg_faithfulness']} |",
        f"| Retrieval — keyword overlap (0-1) | {report['overall']['avg_retrieval_keyword_score']} |",
        f"| Retrieval — semantic similarity (0-1) | {report['overall']['avg_retrieval_semantic_score']} |",
        f"",
        f"## By Category",
        f"",
        f"| Category | Count | Correctness | Faithfulness | Retrieval (keyword) | Retrieval (semantic) |",
        f"|---|---|---|---|---|---|",
    ]

    for category, stats in report["by_category"].items():
        lines.append(
            f"| {category} | {stats['count']} | {stats['avg_correctness']} | "
            f"{stats['avg_faithfulness']} | {stats['avg_retrieval_keyword_score']} | "
            f"{stats['avg_retrieval_semantic_score']} |"
        )

    if report["judge_failures_count"] > 0:
        lines += [
            "",
            f"## ⚠️ Judge Failures ({report['judge_failures_count']})",
            "",
            "These questions have NO correctness/faithfulness score because "
            "the judge itself failed (API error or unparseable response) — "
            "NOT because the answer was graded and found lacking. Treat "
            "these as unscored, not as passing or failing.",
            "",
        ]
        for r in report["all_results"]:
            if r.get("error") and r["correctness"] is None:
                lines += [
                    f"### {r['id']}: {r['question']}",
                    f"- **Error:** {r['error']}",
                    "",
                ]

    lines += ["", f"## Flagged for Manual Review ({len(report['flagged_for_review'])})", ""]

    if not report["flagged_for_review"]:
        lines.append("None — every question scored above the review threshold.")
    else:
        for r in report["flagged_for_review"]:
            lines += [
                f"### {r['id']}: {r['question']}",
                f"",
                f"- **Correctness:** {r['correctness']}/5",
                f"- **Faithfulness:** {r['faithfulness']}/5",
                f"- **Retrieval (keyword):** {r['retrieval_keyword_score']}",
                f"- **Retrieval (semantic):** {r['retrieval_semantic_score']}",
                f"- **Judge reasoning:** {r['judge_reasoning']}",
                f"- **Error:** {r.get('error')}",
                f"- **Generated answer:** {r['generated_answer']}",
                f"- **Reference answer:** {r['reference_answer']}",
                f"",
            ]

    with open(path, "w") as f:
        f.write("\n".join(lines))