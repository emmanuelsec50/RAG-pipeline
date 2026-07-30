from django.shortcuts import render
from .utils import *
# Create your views here.
from django.http import StreamingHttpResponse

import json
from django.http import StreamingHttpResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_protect
from django.views.decorators.http import require_POST

@require_POST
@csrf_protect
def query_view(request):
    try:
        data = json.loads(request.body)
        prompt = data.get("prompt", "").strip()
    except json.JSONDecodeError:
        return HttpResponseBadRequest("Invalid JSON")

    if not prompt:
        return HttpResponseBadRequest("Prompt is required")

    response = StreamingHttpResponse(
        ask(prompt),
        content_type="text/event-stream"
    )
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response

def home(request):
    return render(request, 'ragapp/new.html')