"""
The /embed/ view — served inside the widget's iframe on embuni.ac.ke.

Deliberately a SEPARATE view/route from your main chat page, not the
same page reused. Two reasons:
1. Least privilege: only THIS specific route allows being framed by an
   external site. Your main site, admin panel, and every other page
   keep Django's default clickjacking protection untouched.
2. The embed context needs a compact layout (no header/branding chrome
   that makes sense on a full page but wastes space in a 380px widget
   panel) -- that's a template/UI decision, not shown here, but the
   view-level split is what makes it possible.
"""
from django.shortcuts import render
from django.views.decorators.clickjacking import xframe_options_exempt
from django.views.decorators.csrf import ensure_csrf_cookie

# The wildcard does NOT match the bare root domain -- both must be
# listed explicitly, or a widget embedded directly on embuni.ac.ke
# (no subdomain) would be silently blocked while every subdomain works.
ALLOWED_EMBED_ORIGINS = [
    "https://embuni.ac.ke",
    "https://vixxon.online",
]

# <script src="https://[your-r2-domain]/widget.js" async></script>
@ensure_csrf_cookie
@xframe_options_exempt
def embed_chat_view(request):
    """
    @ensure_csrf_cookie: embed.html's JS reads the csrftoken cookie to
    send as X-CSRFToken on its fetch() calls to /api/query/ -- without
    this decorator, Django has no reason to set that cookie on this
    page, and the first request would fail CSRF validation (same class
    of bug we hit and fixed on the original chat page, early on).

    @xframe_options_exempt skips Django's default X-Frame-Options
    header for this view specifically (which would otherwise block ALL
    framing, including the one we actually want).

    We don't replace it with a permissive X-Frame-Options -- that
    header's ALLOW-FROM directive is deprecated and simply ignored by
    modern Chrome/Firefox/Safari, so it wouldn't provide real
    protection anyway. Content-Security-Policy's frame-ancestors
    directive is the actual, currently-supported way to restrict
    embedding to specific origins, set explicitly below -- including
    both the root domain and a wildcard for every subdomain, since
    those are two distinct match patterns under CSP, not one.
    """
    response = render(request, "ragapp/embed.html")
    origins = " ".join(ALLOWED_EMBED_ORIGINS)
    response["Content-Security-Policy"] = f"frame-ancestors 'self' {origins}"
    return response