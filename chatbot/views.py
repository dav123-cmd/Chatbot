import json

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .ml.inference import ChatbotEngine


def index(request):
    return render(request, "chatbot/index.html")


@csrf_exempt
@require_http_methods(["POST"])
def chat_api(request):
    try:
        body = json.loads(request.body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JsonResponse({"error": "Invalid JSON body."}, status=400)

    message = (body.get("message") or "").strip()
    if not message:
        return JsonResponse({"error": "Message cannot be empty."}, status=400)

    try:
        engine = ChatbotEngine.get_instance(
            model_path=settings.CHATBOT_MODEL_PATH,
            intents_path=settings.CHATBOT_INTENTS_PATH,
            confidence_threshold=settings.CHATBOT_CONFIDENCE_THRESHOLD,
        )
    except FileNotFoundError as exc:
        return JsonResponse({"error": str(exc)}, status=503)

    response, tag, confidence = engine.predict(message)

    return JsonResponse({
        "response": response,
        "intent": tag,
        "confidence": round(confidence, 4),
    })
