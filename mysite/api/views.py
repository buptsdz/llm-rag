from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .service.chatrag import ChatRAGService
import json
import openai

api_key_openai = ''
#填入apikey的地址
temp_url = ''
#定义默认地址
base_url = temp_url if temp_url else 'https://api.openai.com'
chatbot_service = ChatRAGService(api_key_openai,base_url)

@csrf_exempt
def chatbot_view(request):
    
    if request.method == 'POST':
        body = json.loads(request.body)
        query = body.get('query', '')
        session_id = body.get('session_id', 'default_session')
        
        # 检查请求中是否包含 'hello' 参数且为 true
        hello_param = body.get('hello', '').lower() == 'true'
        if hello_param:
            query = "你好，请你自我介绍一下"

    elif request.method == 'GET':
        query = request.GET.get('query', '')
        session_id = request.GET.get('session_id', 'default_session')
    else:
        return JsonResponse({'error': 'Invalid request'}, status=400)

    def stream_response():
        try:
            for chunk in chatbot_service.get_response_stream(query, session_id):
                yield chunk
        #部分错误抛出捕获
        except openai.RateLimitError as e:
            if e.code == 'insufficient_quota':
                yield '请检查apikey是否正确且有足够token。'
        except openai.AuthenticationError as e:
            if e.code == 'account_deactivated':
                yield '您的apikey已停用，请检查账户状态。'
        except openai.NotFoundError as e:
            if e.code == 'unknown_url':
                yield '请检查您的apikey的地址是否正确。'

    return StreamingHttpResponse(stream_response(), content_type='text/plain')
