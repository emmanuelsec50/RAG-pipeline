from django.shortcuts import render
from .utils import *
# Create your views here.
from django.http import HttpResponseNotAllowed, StreamingHttpResponse
from django_ratelimit.decorators import ratelimit
import json
from django.http import StreamingHttpResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_protect, csrf_exempt
from django.views.decorators.http import require_POST
from django.core.cache import cache
from asgiref.sync import sync_to_async
from .ratelimit import async_ratelimit, ip_key, chat_id_key


@csrf_exempt
@require_POST
@async_ratelimit(ip_key, rate=60, window_seconds=60, limit_name="ip")
@async_ratelimit(chat_id_key, rate=5, window_seconds=60, limit_name="chatid")
async def query_view(request):
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    try:
        body = await sync_to_async(lambda: request.body)()
        data = json.loads(body)
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


@ratelimit(key='ip', rate='150/m', block=True)
def home(request):
    return render(request, 'ragapp/new.html')