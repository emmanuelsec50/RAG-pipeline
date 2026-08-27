"""
LLM-as-judge scoring for generated answers.
"""
import json
import re
import logging

from openai import OpenAI
from decouple import config

logger = logging.getLogger(__name__)

JUDGE_MODEL = "deepseek-v4-pro"

JUDGE_SYSTEM_PROMPT = """You are a strict evaluator grading an AI assistant's answer \
for a university's student-help chatbot. Real students rely on this being accurate.

Score two things independently, each from 1 (worst) to 5 (best):

1. correctness: Does the generated answer convey the same factual content \
as the reference answer? Penalize missing steps, wrong details, or vague \
non-answers heavily. A generated answer that is vaguer but not factually \
wrong should score in the middle, not high.

2. faithfulness: Is the generated answer actually supported by the \
provided retrieved context? Penalize any claim in the generated answer \
that is NOT backed by the retrieved context, even if that claim happens \
to be true — an unsupported claim is a hallucination risk regardless of \
whether it's correct by coincidence.

Special case: if the reference answer indicates this is an out-of-scope \
question (the assistant should say it doesn't know rather than guess), \
score correctness as 5 ONLY if the generated answer appropriately \
declines/hedges, and as 1 if it confidently fabricates an answer.

Respond with ONLY a JSON object, no other text before or after it, in \
this exact shape:
{"correctness": <int 1-5>, "faithfulness": <int 1-5>, "reasoning": "<one or two sentences>"}
"""


def _extract_json_object(raw: str) -> dict:
    """
    Robustly pulls a JSON object out of a model response that may have
    preamble/postamble text around it despite instructions not to
    include any (models don't always follow that instruction reliably).

    Strategy: find the first '{' and the LAST matching '}' in the text
    and parse only that substring. This is more robust than the
    previous approach (which only handled markdown-fence-wrapped JSON)
    because it works regardless of what surrounds the JSON object.
    """
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in judge response: {raw[:200]!r}")
    return json.loads(raw[start:end + 1])


def grade_answer(question: str, generated_answer: str, reference_answer: str,
                  retrieved_context: str) -> dict:
    """
    Returns a dict: {correctness, faithfulness, reasoning, error}.
    On any failure, correctness/faithfulness/reasoning are None and
    `error` contains the actual failure reason — this is what lets a
    human diagnose WHY a question wasn't scored, instead of just seeing
    a silent None in the report.
    """
    client = OpenAI(
        base_url="https://api.deepseek.com",
        api_key=config("DEEPSEEK_API_KEY"),
    )

    user_prompt = f"""Question: {question}

Reference answer (ground truth): {reference_answer}

Retrieved context (what the RAG pipeline had available): {retrieved_context}

Generated answer (to be graded): {generated_answer}
"""

    raw = None
    try:
        completion = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,  # deterministic grading, not creative generation
            max_tokens=300,
            stream=False,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}},
            response_format={'type': 'json_object'}
        )
        raw = completion.choices[0].message.content.strip()
        parsed = _extract_json_object(raw)

        return {
            "correctness": int(parsed["correctness"]),
            "faithfulness": int(parsed["faithfulness"]),
            "reasoning": parsed.get("reasoning", ""),
            "error": None,
        }

    except Exception as exc:
        logger.warning("Judge grading failed for question %r: %s", question, exc)
        return {
            "correctness": None,
            "faithfulness": None,
            "reasoning": None,
            # Include the raw response (truncated) in the error so a
            # human reviewing the report can see exactly what the model
            # returned that failed to parse, not just "it failed."
            "error": f"{exc} | raw_response={raw[:300] if raw else None!r}",
        }