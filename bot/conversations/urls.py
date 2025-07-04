from django.urls import path
from . import views

app_name = 'conversations'

urlpatterns = [
    path('index/', views.index, name='conversations-index'),
    path('<int:agent_id>/', views.conversations, name='conversations-list'),
    path('<int:agent_id>/<int:conversation_id>/messages/', views.conversation_messages, name='conversation-messages'),
    path('create/', views.create_conversation, name='create-conversation'),
    path('<int:agent_id>/<int:conversation_id>/update/', views.update_conversation, name='update-conversation'),
    path('<int:agent_id>/<int:conversation_id>/<int:message_id>/update/', views.update_message, name='update-message'),
    path('chat/<int:agent_id>/<int:conversation_id>/', views.chat_with_ai, name='chat-with-ai'),
    path('<int:agent_id>/<int:conversation_id>/check-active/', views.check_conversation_active, name='check-conversation-active'),
    path('api/tokens/', views.get_api_tokens, name='get-api-tokens'),
    path('api/tokens/refresh/', views.refresh_api_token, name='refresh-api-token'),
    path('api/create/', views.create_api_conversation, name='create-api-conversation'),
    path('api/chat/<int:conversation_id>/', views.chat_with_api, name='chat-with-api'),
]
