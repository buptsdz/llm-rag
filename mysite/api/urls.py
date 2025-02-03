from django.urls import path
from .views import chatbot_view
from django.views.generic import TemplateView

app_name = "api"
urlpatterns = [
    path("chatbot/", chatbot_view, name="chatbot_view"),
    path("chat/", TemplateView.as_view(template_name="api/chat.html"), name="chat"),
]
