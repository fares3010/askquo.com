from django.urls import path
from . import views

app_name = 'conversations'

urlpatterns = [
    path('index/', views.index, name='conversations-index'),
    path('list/', views.conversations, name='conversations-list'),
    path('<int:conversationId>/messages/', views.conversation_messages, name='conversation-messages'),
]
