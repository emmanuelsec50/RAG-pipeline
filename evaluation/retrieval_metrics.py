"""
Automatic retrieval relevance scoring — no manual keyword tagging required.

Since we only have (question, reference_answer) pairs, we derive an
approximate "did retrieval pull the right content" signal by checking
how much of the reference answer's meaningful vocabulary actually shows
up in the chunks the pipeline retrieved. This is a proxy, not a perfect
measure — a chunk could contain the right words while framing them
completely wrong — but it catches the failure mode that matters most:
retrieval pulling completely unrelated content for a question.
"""
import re

# Deliberately small and conservative — the goal is filtering out noise
# words (the, is, how, do), not being a linguistically complete stopword
# list. Over-filtering here would hide real signal.
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "how", "what", "when", "where", "why", "who",
    "which", "this", "that", "these", "those", "to", "of", "in", "on",
    "for", "and", "or", "but", "if", "then", "than", "as", "at", "by",
    "with", "from", "your", "you", "my", "i", "it", "its", "can", "will",
    "should", "would", "could", "please", "also", "not", "no", "yes",
}


def extract_significant_terms(text: str) -> set[str]:
    """
    Pulls out the meaningful words from a reference answer — lowercased,
    stopwords removed, short tokens (<=2 chars) dropped since they're
    rarely meaningful on their own (except things like "ID", handled
    separately by keeping uppercase acronyms regardless of length).
    """
    # Preserve acronyms (HELB, ICT, KRA) before lowercasing everything else.
    acronyms = set(re.findall(r"\b[A-Z]{2,}\b", text))

    words = re.findall(r"[a-zA-Z]+", text.lower())
    significant = {
        w for w in words
        if w not in STOPWORDS and len(w) > 2
    }
    return significant | {a.lower() for a in acronyms}


def retrieval_overlap_score(retrieved_context: str, reference_answer: str) -> dict:
    """
    Returns the fraction of the reference answer's significant terms
    that appear somewhere in the retrieved context, plus the specific
    terms that were missing — the missing list is what makes a low
    score actionable rather than just a number.
    """
    expected_terms = extract_significant_terms(reference_answer)
    if not expected_terms:
        # Reference answer had no extractable content terms (e.g. the
        # out-of-scope control question) — nothing meaningful to score.
        return {"score": None, "missing_terms": [], "expected_terms": []}

    context_lower = retrieved_context.lower()
    found = {term for term in expected_terms if term in context_lower}
    missing = expected_terms - found

    return {
        "score": round(len(found) / len(expected_terms), 3),
        "missing_terms": sorted(missing),
        "expected_terms": sorted(expected_terms),
    }
