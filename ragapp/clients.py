"""
Shared AsyncOpenAI client instances, created ONCE at module import time
and reused across every request — not recreated per call, as embed(),
glm(), and prompt_rewrite() were each doing before.

Why this matters: every AsyncOpenAI client owns its own httpx.AsyncClient
connection pool under the hood. Creating a fresh client per request means
paying TLS-handshake/connection-setup overhead on every single call
instead of reusing warm, keep-alive connections to the same host — real,
avoidable overhead under concurrent load, exactly the kind of thing that
matters at launch-day scale.

TWO clients, not three: embed() and prompt_rewrite() both call
OpenRouter with the same credentials (VOYAGE_API) — they share one
client. glm() calls DeepSeek directly and needs its own, separate one.

Module-level instantiation is enough here (no AppConfig.ready() needed,
unlike the knowledge base) — creating an AsyncOpenAI client is cheap and
doesn't do blocking I/O; the actual network connections are opened
lazily by httpx on first real request, not at construction time. Python
only executes this module's top-level code once per process, on first
import, so these are naturally singletons for the lifetime of that
process.
"""
from openai import AsyncOpenAI
from decouple import config

openrouter_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=config("OPENROUTER_API"),
)

deepseek_client = AsyncOpenAI(
    base_url="https://api.deepseek.com",
    api_key=config("DEEPSEEK_API_KEY"),
)