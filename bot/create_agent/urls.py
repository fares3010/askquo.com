from django.urls import path
from . import views

app_name = 'agents'

urlpatterns = [
    # List and create endpoints
    path('', views.agent_list, name='agent-list'),  # Changed from 'agents/' to '' since it's already namespaced
    path('create/', views.create_agent, name='agent-create'),  # Moved create before detail for better organization
    
    # Detail, update and delete endpoints
    path('<int:pk>/', views.agent_detail, name='agent-detail'),
    path('<int:pk>/update/', views.update_agent, name='agent-update'),
    path('<int:pk>/delete/', views.delete_agent, name='agent-delete'),
]