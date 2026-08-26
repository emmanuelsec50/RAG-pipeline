"""
Entry point: runs the full eval dataset through your real RAG pipeline
and produces a report.

Usage:
    python -m evaluation.run_eval

Requires pipeline_adapter.py to be wired up to your real
retrieve_relevant_resources() / generation / embed() functions first —
see the NotImplementedError stubs there.
"""
import logging

from .dataset import EVAL_CASES
from .retrieval_metrics import retrieval_overlap_score
from .semantic_similarity import max_chunk_similarity
from .judge import grade_answer
from .report import build_report, write_json_report, write_markdown_report
from . import pipeline_adapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_single_case(case: dict, embeddings, pages_and_chunks) -> dict:
    """
    Runs one eval question through: retrieve -> generate -> score
    (keyword + semantic retrieval, then LLM-judge correctness/faithfulness).

    Returns a fully-populated result dict, or a result dict with
    error fields set if any stage fails — a single bad question
    shouldn't crash the whole run.
    """
    question = case["question"]
    logger.info("Evaluating: %s", question)

    try:
        retrieval = pipeline_adapter.retrieve(question, embeddings, pages_and_chunks)
        context_text = retrieval["context_text"]
        context_items = retrieval["context_items"]
        chunk_embeddings = retrieval["chunk_embeddings"]
    except Exception as exc:
        logger.error("Retrieval failed for %r: %s", question, exc)
        return {**case, "error": f"retrieval_failed: {exc}",
                "retrieval_keyword_score": None, "retrieval_semantic_score": None,
                "generated_answer": None, "correctness": None, "faithfulness": None,
                "judge_reasoning": None}

    try:
        generated_answer = pipeline_adapter.generate_answer(question, context_items)
    except Exception as exc:
        logger.error("Generation failed for %r: %s", question, exc)
        return {**case, "error": f"generation_failed: {exc}",
                "retrieval_keyword_score": None, "retrieval_semantic_score": None,
                "generated_answer": None, "correctness": None, "faithfulness": None,
                "judge_reasoning": None}

    keyword_result = retrieval_overlap_score(context_text, case["reference_answer"])

    try:
        reference_embedding = pipeline_adapter.embed_text(case["reference_answer"])
        chunk_texts = [item["sentence_chunk"] for item in context_items]
        semantic_result = max_chunk_similarity(reference_embedding, chunk_embeddings, chunk_texts)
    except Exception as exc:
        logger.warning("Semantic scoring failed for %r: %s", question, exc)
        semantic_result = {"score": None, "best_chunk_text": None, "error": str(exc)}

    judge_result = grade_answer(
        question=question,
        generated_answer=generated_answer,
        reference_answer=case["reference_answer"],
        retrieved_context=context_text,
    )

    return {
        **case,
        "generated_answer": generated_answer,
        "retrieved_context": context_text,
        "retrieval_keyword_score": keyword_result["score"],
        "retrieval_keyword_missing_terms": keyword_result["missing_terms"],
        "retrieval_semantic_score": semantic_result["score"],
        "retrieval_best_matching_chunk": semantic_result.get("best_chunk_text"),
        "correctness": judge_result["correctness"],
        "faithfulness": judge_result["faithfulness"],
        "judge_reasoning": judge_result["reasoning"],
        "error": judge_result.get("error") or semantic_result.get("error"),
    }


def main():
    embeddings, pages_and_chunks = pipeline_adapter.load_knowledge_base()

    results = [
        run_single_case(case, embeddings, pages_and_chunks)
        for case in EVAL_CASES
    ]

    report = build_report(results)
    write_json_report(report)
    write_markdown_report(report)

    logger.info("Done. See eval_report.md for the summary, eval_results.json for full detail.")
    logger.info("Overall: %s", report["overall"])
    logger.info("Flagged for review: %d question(s)", len(report["flagged_for_review"]))


if __name__ == "__main__":
    main()