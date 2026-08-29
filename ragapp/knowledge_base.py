"""
Holds the loaded knowledge base as module-level state, separate from
utils.py to avoid any import-order surprises between AppConfig.ready()
and the functions that consume this state.
"""
import pickle
import torch

EMBEDDINGS_PATH = './ragapp/embeddings_mongo.pt'
PAGES_AND_CHUNKS_SAVE_PATH_PICKLE = "./ragapp/pages_and_chunks_mongo.pkl"

_embeddings = None
_pages_and_chunks = None


def load():
    """Called once from RagappConfig.ready()."""
    global _embeddings, _pages_and_chunks
    _embeddings = torch.load(EMBEDDINGS_PATH)
    with open(PAGES_AND_CHUNKS_SAVE_PATH_PICKLE, "rb") as f:
        _pages_and_chunks = pickle.load(f)


def get():
    """
    Returns (embeddings, pages_and_chunks). If somehow called before
    ready() has run (e.g. a management command or test that doesn't go
    through the normal app-loading path), loads on first access as a
    safety net rather than returning None and causing a confusing
    downstream crash.
    """
    if _embeddings is None or _pages_and_chunks is None:
        load()
    return _embeddings, _pages_and_chunks