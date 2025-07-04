from django.urls import path
from . import views

app_name = 'agents'

urlpatterns = [
    # Agent CRUD operations
    path('', views.agent_list, name='agent-list'),
    path('create/', views.create_agent, name='agent-create'),
    path('<int:agent_id>/', views.agent_details, name='agent-details'),
    path('<int:agent_id>/update/', views.update_agent, name='agent-update'),
    path('<int:agent_id>/delete/', views.delete_agent, name='agent-delete'),

    
    # Agent documents
    path('<int:agent_id>/documents/', views.get_agent_documents, name='agent-documents'),
    path('<int:agent_id>/documents/create/', views.create_agent_document, name='agent-document-create'),
    path('<int:agent_id>/documents/<int:document_id>/update/', views.update_agent_document, name='agent-document-update'),
    path('<int:agent_id>/documents/<int:document_id>/delete/', views.delete_agent_document, name='agent-document-delete'),
    
    # Agent texts
    path('<int:agent_id>/texts/', views.get_agent_texts, name='agent-texts'),
    path('<int:agent_id>/texts/create/', views.create_agent_text, name='agent-text-create'),
    path('<int:agent_id>/texts/<int:text_id>/update/', views.update_agent_text, name='agent-text-update'),
    path('<int:agent_id>/texts/<int:text_id>/delete/', views.delete_agent_text, name='agent-text-delete'),
    
    # Agent integrations
    path('<int:agent_id>/integrations/', views.get_agent_integrations, name='agent-integrations'),
    path('<int:agent_id>/integrations/create/', views.create_agent_integration, name='agent-integration-create'),
    path('<int:agent_id>/integrations/<int:integration_id>/update/', views.update_agent_integration, name='agent-integration-update'),
    path('<int:agent_id>/integrations/<int:integration_id>/delete/', views.delete_agent_integration, name='agent-integration-delete'),
    
    # Agent websites
    path('<int:agent_id>/websites/', views.get_agent_websites, name='agent-websites'),
    path('<int:agent_id>/websites/create/', views.create_agent_website, name='agent-website-create'),
    path('<int:agent_id>/websites/<int:website_id>/update/', views.update_agent_website, name='agent-website-update'),
    path('<int:agent_id>/websites/<int:website_id>/delete/', views.delete_agent_website, name='agent-website-delete'),
    
    # Embed agent data
    path('<int:agent_id>/<str:object_type>/<str:object_id>/embed/', views.embed_agent_data, name='agent-embed-data'),
    path('<int:agent_id>/<str:object_type>/<str:object_id>/delete/', views.delete_agent_embedding, name='agent-embedding-delete'),

    # Agent integrations frontend
    path('<int:agent_id>/integrations/frontend/', views.get_agent_integrations_frontend, name='agent-integrations-frontend'),
    path('<int:agent_id>/integrations/frontend/create/', views.create_agent_integration_frontend, name='agent-integration-frontend-create'),
    path('<int:agent_id>/integrations/frontend/<int:integration_id>/update/', views.update_agent_integration_frontend, name='agent-integration-frontend-update'),
    path('<int:agent_id>/integrations/frontend/<int:integration_id>/delete/', views.delete_agent_integration_frontend, name='agent-integration-frontend-delete'),
]