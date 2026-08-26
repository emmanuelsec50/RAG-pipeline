"""
Semantic similarity scoring — the companion to retrieval_metrics.py's
keyword overlap score.

Where keyword overlap catches "completely wrong chunk retrieved" cheaply
and reliably, it under-scores genuinely good retrieval when the
reference answer is long or the retrieved chunk paraphrases rather than
repeats the same wording. Semantic similarity (cosine similarity between
embeddings) catches that case, at the cost of one extra embedding API
call per eval question.

Report BOTH scores rather than picking one — they catch different
failure modes:
  - Low keyword overlap + low semantic similarity  -> genuinely wrong
    chunk retrieved, high-confidence signal.
  - Low keyword overlap + high semantic similarity  -> retrieval was
    fine, the reference answer just uses different wording. Don't
    treat this as a retrieval failure.
  - High keyword overlap + low semantic similarity  -> rare, worth
    manually inspecting — could mean the shared words are coincidental
    (e.g. both mention "semester" but about different things).

`embed_fn` is injected rather than imported directly, so this module
stays testable without a live API key, and so it can reuse whatever
embedding function the production pipeline (ragapp/utils.py's embed())
already calls — no separate embedding logic to maintain.
"""
from typing import Callable

import torch


def cos_sim(a, b) -> float:
    """
    Raw torch cosine similarity — deliberately not using
    sentence-transformers, matching the dependency decision already
    made for the production RAG pipeline (see requirements.txt cleanup).
    """
    a = torch.as_tensor(a, dtype=torch.float32)
    b = torch.as_tensor(b, dtype=torch.float32)
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)
    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1)).item()


def semantic_overlap_score(retrieved_context: str, reference_answer: str,
                            embed_fn: Callable[[str], list[float]]) -> dict:
    """
    Embeds both texts using the SAME embedding function the production
    pipeline uses for retrieval, and returns their cosine similarity.

    Returns a dict (not a bare float) to match retrieval_overlap_score()'s
    shape, and to leave room for an `error` field if embedding either
    text fails (e.g. API timeout) — a single failed embedding call
    shouldn't crash the whole eval run.
    """
    try:
        context_embedding = embed_fn(retrieved_context)
        reference_embedding = embed_fn(reference_answer)
        score = cos_sim(reference_embedding, context_embedding)
        return {"score": round(score, 3), "error": None}
    except Exception as exc:
        return {"score": None, "error": str(exc)}
