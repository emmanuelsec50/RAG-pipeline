from django.shortcuts import render
from .utils import *
# Create your views here.
from django.http import StreamingHttpResponse
from django_ratelimit.decorators import ratelimit
import json
from django.http import StreamingHttpResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_protect
from django.views.decorators.http import require_POST
from django.core.cache import cache


@ratelimit(key='ip', rate='3/m', block=True)
@require_POST
@csrf_protect
def query_view(request):
    try:
        data = json.loads(request.body)
        prompt = data.get("prompt", "").strip()
        chat_id = data.get("chat_id")
        

    except json.JSONDecodeError:
        return HttpResponseBadRequest("Invalid JSON")

    if not prompt:
        return HttpResponseBadRequest("Prompt is required")
    if not chat_id:
            return HttpResponseBadRequest("Do not play around with the code")


    # get_messages(str(chat_id))
    
    # response = StreamingHttpResponse(
    #     ask(prompt, get_messages(str(chat_id))),
    #     content_type="text/event-stream"
    # )
    
    response = StreamingHttpResponse(
        stream_and_save(str(chat_id), prompt),
        content_type="text/event-stream"
    )
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    
    
    return response


@ratelimit(key='ip', rate='10/m', block=True)
def home(request):
    return render(request, 'ragapp/new.html')