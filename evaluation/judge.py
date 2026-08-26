"""
LLM-as-judge scoring for generated answers.

Deliberately uses a NON-streaming call, separate from the production
glm() generator in ragapp/utils.py — streaming is a UX concern for the
live chat interface, and has no benefit for offline batch evaluation.
Mixing the two would couple this harness unnecessarily to the SSE
plumbing built for the chat UI.

The judge is asked to score two DIFFERENT things, not one blended score,
because they catch different failure modes:

- correctness: does the generated answer match the reference answer's
  actual content? (catches wrong information)
- faithfulness: is the generated answer actually supported by the
  retrieved context, or did it add unsupported claims? (catches
  hallucination — this can be LOW even when correctness is high, if the
  model happened to guess right without the context actually saying so)
"""
import json
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

Respond with ONLY a JSON object, no other text, in this exact shape:
{"correctness": <int 1-5>, "faithfulness": <int 1-5>, "reasoning": "<one or two sentences>"}
"""


def grade_answer(question: str, generated_answer: str, reference_answer: str,
                  retrieved_context: str) -> dict:
    """
    Returns a dict: {correctness, faithfulness, reasoning}.
    On any failure to get a valid judgment (API error, malformed JSON),
    returns a dict with score=None and an error message rather than
    raising — a single bad grading call shouldn't crash an entire eval
    run over a full dataset.
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

        # Models occasionally wrap JSON in markdown fences despite
        # instructions not to — strip defensively rather than failing
        # the whole grading run over formatting.
        if raw.startswith("```"):
            raw = raw.strip("`").removeprefix("json").strip()

        parsed = json.loads(raw)
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
            "error": str(exc),
        }
