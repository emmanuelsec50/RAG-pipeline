"""
Async-compatible, two-tier rate limiting built on Django's native async
cache API (cache.aincr / cache.aset).

TWO INDEPENDENT LIMITS, stacked together — see query_view usage below:

1. Per-chat_id (tight, e.g. 5/min): catches a single runaway
   conversation or script hammering the API repeatedly.
   LIMITATION, stated plainly: chat_id is a client-generated UUID with
   no binding to real identity — a deliberate abuser can call
   crypto.randomUUID() again to get a fresh chat_id and reset this
   counter. This raises the bar for casual abuse; it does not stop a
   determined one. The real fix for that is adding actual
   authentication later, not a rate-limiting trick.

2. Per-IP (generous, e.g. 60/min): many students share one public IP
   on university WiFi via NAT, so this must be sized high enough to
   comfortably absorb many simultaneous legitimate students — its job
   is catching a genuinely abnormal flood from one machine, not
   policing normal shared-network usage.

key_func is ASYNC because the chat_id key needs to read the request
body (request.abody(), Django's native async body read) before the
view itself parses it — a sync key_func can't do that in an async view.
"""
import functools
import json
import logging

from asgiref.sync import sync_to_async
from django.core.cache import cache
from django.http import JsonResponse

logger = logging.getLogger(__name__)


def async_ratelimit(key_func, rate: int, window_seconds: int, limit_name: str = "default"):
    """
    key_func: an ASYNC callable — `async def key_func(request) -> str`.
    Stack multiple @async_ratelimit decorators on one view to enforce
    multiple independent limits (see query_view example below).
    """
    def decorator(view_func):
        @functools.wraps(view_func)
        async def wrapped(request, *args, **kwargs):
            identity = await key_func(request)
            cache_key = f"ratelimit:{limit_name}:{view_func.__name__}:{identity}"

            try:
                count = await cache.aincr(cache_key)
            except ValueError:
                # Key doesn't exist yet. Use aadd() -- an ATOMIC "create
                # only if absent" operation (Redis SETNX under the
                # hood) -- not a plain aset(). This matters: under real
                # concurrent load, multiple requests can reach this
                # branch at nearly the same instant. A plain aset()
                # here is a genuine race -- every concurrent request
                # would independently reset the counter to 1 instead of
                # incrementing it, meaning the limit silently does
                # nothing during that window (confirmed directly: 10
                # simultaneous requests against a limit of 5 all
                # succeeded before this fix). aadd() guarantees only ONE
                # request wins the "create" race; every other
                # concurrent request correctly falls through to
                # incrementing the real, now-existing counter instead.
                created = await cache.aadd(cache_key, 1, timeout=window_seconds)
                if created:
                    count = 1
                else:
                    count = await cache.aincr(cache_key)

            if count > rate:
                logger.info(
                    "Rate limit exceeded: limit=%s key=%s count=%d rate=%d",
                    limit_name, cache_key, count, rate,
                )
                return JsonResponse(
                    {"error": "Too many requests. Please slow down and try again shortly."},
                    status=429,
                )

            return await view_func(request, *args, **kwargs)

        return wrapped
    return decorator


async def ip_key(request) -> str:
    """Generous-tier key — one bucket per source IP."""
    return request.META.get("REMOTE_ADDR", "unknown")


async def chat_id_key(request) -> str:
    """
    Tight-tier key — one bucket per conversation. Reads and parses the
    request body directly, since chat_id lives in the JSON payload, not
    in headers or META.

    Uses sync_to_async(lambda: request.body) rather than request.abody()
    deliberately: abody() only exists on genuine ASGIRequest objects,
    which aren't guaranteed in every context (confirmed directly —
    Django's own test RequestFactory produces a WSGIRequest even for
    async views, which has no abody() at all). request.body works
    correctly either way, and this matches the pattern query_view
    itself already uses to read the body asynchronously.

    Deliberately tolerant of bad input: if the body isn't valid JSON or
    has no chat_id, falls back to a shared "unidentified" bucket rather
    than raising — the view's own validation will reject a genuinely
    malformed request with a 400 anyway; the rate limiter's job is just
    to not crash before that happens.
    """
    try:
        body = await sync_to_async(lambda: request.body)()
        data = json.loads(body)
        chat_id = data.get("chat_id")
        return str(chat_id) if chat_id else "unidentified"
    except (json.JSONDecodeError, ValueError, AttributeError):
        return "unidentified"