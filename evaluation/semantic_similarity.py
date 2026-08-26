"""
Semantic similarity scoring.

IMPORTANT DESIGN CHANGE from the first version: this now scores against
EACH individually retrieved chunk and takes the MAX similarity, rather
than embedding one giant concatenated block of all retrieved chunks and
comparing that single blended embedding to the reference answer.

Why the change: real eval data showed answers the judge scored 5/5
(clearly correct, clearly supported) getting semantic scores as low as
0.22 under the concatenation approach. The problem wasn't retrieval —
it was that averaging together 8 chunks about different sub-topics into
one embedding dilutes the signal, so even a genuinely well-matching
single chunk gets washed out by the other 7 chunks in the blend. Scoring
each chunk independently and taking the best match answers the actual
question we care about: "did retrieval find AT LEAST ONE chunk that
actually supports this answer?" — which is what retrieval quality
means in practice.

Reuses the knowledge base's ALREADY-COMPUTED chunk embeddings (from
embeddings5.pt) rather than re-embedding each retrieved chunk's text
via a fresh API call — this is both more efficient (no extra embedding
calls beyond the one needed for the reference answer) and more
consistent (uses the exact same vectors retrieval itself operated on).
"""
from typing import Callable

import torch


def cos_sim(a, b) -> torch.Tensor:
    """
    Raw torch cosine similarity, returns the full similarity matrix —
    a is (1, dim) or (n, dim), b is (m, dim), result is (n, m).
    """
    a = torch.as_tensor(a, dtype=torch.float32)
    b = torch.as_tensor(b, dtype=torch.float32)
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)
    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def max_chunk_similarity(reference_embedding, chunk_embeddings, chunk_texts: list[str] = None) -> dict:
    """
    Compares the reference answer's embedding against EACH retrieved
    chunk's embedding individually, returns the highest similarity
    found and (if chunk_texts provided) which chunk achieved it — the
    latter is what makes a low score actionable: you can see exactly
    which chunk retrieval considered "best" and judge for yourself
    whether it's actually relevant.
    """
    sims = cos_sim(reference_embedding, chunk_embeddings)[0]  # shape: (num_chunks,)
    best_idx = int(torch.argmax(sims).item())
    best_score = round(sims[best_idx].item(), 3)

    return {
        "score": best_score,
        "best_chunk_index": best_idx,
        "best_chunk_text": chunk_texts[best_idx] if chunk_texts else None,
        "all_scores": [round(s, 3) for s in sims.tolist()],
        "error": None,
    }


def semantic_overlap_score(retrieved_context: str, reference_answer: str,
                            embed_fn: Callable[[str], list[float]]) -> dict:
    """
    Fallback single-embedding comparison, kept for cases where you only
    have the joined context text and not per-chunk embeddings (e.g.
    quick manual spot-checks). Prefer max_chunk_similarity() in the
    actual eval run — this is not what run_eval.py uses by default
    anymore.
    """
    try:
        context_embedding = embed_fn(retrieved_context)
        reference_embedding = embed_fn(reference_answer)
        score = cos_sim(reference_embedding, context_embedding)[0][0].item()
        return {"score": round(score, 3), "error": None}
    except Exception as exc:
        return {"score": None, "error": str(exc)}