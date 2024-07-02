from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .service.chatrag import ChatRAGService
import json

api_key_openai = '在此填入你的openapikey'
chatbot_service = ChatRAGService(api_key_openai)

@csrf_exempt
def chatbot_view(request):
    
    if  request.method == 'POST':
        body = json.loads(request.body)
        query = body.get('query', '')
        session_id = body.get('session_id', 'default_session')
        
        # Check if 'hello' is true in request
        hello_param = body.get('hello', '').lower() == 'true'
        if hello_param:
            query = "你好，请你自我介绍一下"

    elif request.method == 'GET':
        query = request.GET.get('query', '')
        session_id = request.GET.get('session_id', 'default_session')
    else:
        return JsonResponse({'error': 'Invalid request'}, status=400)

    def stream_response():
        for chunk in chatbot_service.get_response_stream(query, session_id):
            yield chunk

    return StreamingHttpResponse(stream_response(), content_type='text/plain')
