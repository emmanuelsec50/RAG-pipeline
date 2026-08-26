"""
Adapter connecting this eval harness to your REAL production RAG
pipeline (ragapp/utils.py).

ADJUST THE IMPORT BELOW to match wherever ragapp/utils.py actually lives
relative to wherever you put this evaluation/ folder. If evaluation/
sits at your project root (next to ragapp/), this import works as-is.
"""
import pickle
import torch

from ragapp.utils import (
    retrieve_relevant_resources,
    prompt_formatter,
    embed,
    PAGES_AND_CHUNKS_SAVE_PATH_PICKLE,
    EMBEDDINGS_PATH,
)
from openai import OpenAI
from decouple import config


def load_knowledge_base(embeddings_path: str = None, chunks_path: str = None):
    """
    Loads your knowledge base ONCE for the whole eval run — same
    load-once principle as the production app's startup behavior, not
    per-question. Defaults to the same paths utils.py itself uses, so
    the eval harness is always testing against the exact same knowledge
    base your live app serves from.
    """
    embeddings = torch.load(embeddings_path or EMBEDDINGS_PATH)
    with open(chunks_path or PAGES_AND_CHUNKS_SAVE_PATH_PICKLE, "rb") as f:
        pages_and_chunks = pickle.load(f)
    return embeddings, pages_and_chunks


def retrieve(query: str, embeddings, pages_and_chunks, n_resources: int = 8) -> dict:
    """
    Calls your real retrieve_relevant_resources(). Returns both the
    joined context TEXT (what the scoring functions need) and the raw
    context ITEMS (so generate_answer() can build the same prompt shape
    prompt_formatter() expects, and so you can inspect exactly which
    chunks were pulled when reviewing a flagged low-scoring question).
    """
    scores, indices = retrieve_relevant_resources(
        query=query, embeddings=embeddings, n_resources_to_return=n_resources
    )
    context_items = [pages_and_chunks[i] for i in indices]
    context_text = "\n".join(item["chunk"] for item in context_items)
    return {"context_text": context_text, "context_items": context_items}


def generate_answer(query: str, context_items: list) -> str:
    """
    Non-streaming generation for eval purposes — deliberately NOT
    utils.py's glm(), which yields SSE-formatted strings for the live
    chat UI. Batch evaluation has no use for streaming, and reusing
    glm() directly would mean reassembling SSE chunks just to get the
    full text, for no benefit.

    Uses the SAME prompt_formatter() and the SAME model/provider
    (DeepSeek) your production pipeline actually calls — this must stay
    in sync with utils.py's glm() function. If you change the model or
    prompt in production, update this to match, or the eval stops
    measuring what real users actually get.
    """
    prompt = prompt_formatter(query=query, context_items=context_items)

    client = OpenAI(
        api_key=config("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )

    completion = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=False,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "disabled"}},
    )

    return completion.choices[0].message.content


def embed_text(text: str) -> list[float]:
    """
    Reuses your production embed() function directly — same embedding
    model used for retrieval, so semantic similarity scoring measures
    against the exact same vector space your actual pipeline uses.
    """
    return embed(text)