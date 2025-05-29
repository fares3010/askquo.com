from django.urls import path
from . import views

app_name = 'agents'

urlpatterns = [
    # Agent CRUD operations
    path('', views.agent_list, name='agent-list'),
    path('create/', views.create_agent, name='agent-create'),
    path('<int:agent_id>/', views.agent_detail, name='agent-detail'),
    path('<int:agent_id>/update/', views.update_agent, name='agent-update'),
    path('<int:agent_id>/delete/', views.delete_agent, name='agent-delete'),
    
    # Agent documents
    path('<int:agent_id>/documents/', views.get_agent_documents, name='agent-documents'),
    path('<int:agent_id>/documents/create/', views.create_agent_document, name='agent-document-create'),
    path('documents/<int:document_id>/update/', views.update_agent_document, name='agent-document-update'),
    path('documents/<int:document_id>/delete/', views.delete_agent_document, name='agent-document-delete'),
    
    # Agent texts
    path('<int:agent_id>/texts/', views.get_agent_texts, name='agent-texts'),
    path('<int:agent_id>/texts/create/', views.create_agent_text, name='agent-text-create'),
    path('texts/<int:text_id>/update/', views.update_agent_text, name='agent-text-update'),
    path('texts/<int:text_id>/delete/', views.delete_agent_text, name='agent-text-delete'),
    
    # Agent Q&A pairs
    path('<int:agent_id>/qa-pairs/', views.get_agent_qa_pairs, name='agent-qa-pairs'),
    path('<int:agent_id>/qa-pairs/create/', views.create_agent_qa_pair, name='agent-qa-pair-create'),
    path('qa-pairs/<int:qa_pair_id>/update/', views.update_agent_qa_pair, name='agent-qa-pair-update'),
    path('qa-pairs/<int:qa_pair_id>/delete/', views.delete_agent_qa_pair, name='agent-qa-pair-delete'),
    
    # Agent integrations
    path('<int:agent_id>/integrations/', views.get_agent_integrations, name='agent-integrations'),
    path('<int:agent_id>/integrations/create/', views.create_agent_integration, name='agent-integration-create'),
    path('integrations/<int:integration_id>/update/', views.update_agent_integration, name='agent-integration-update'),
    path('integrations/<int:integration_id>/delete/', views.delete_agent_integration, name='agent-integration-delete'),
    
    # Agent websites
    path('<int:agent_id>/websites/', views.get_agent_websites, name='agent-websites'),
    path('<int:agent_id>/websites/create/', views.create_agent_website, name='agent-website-create'),
    path('websites/<int:website_id>/update/', views.update_agent_website, name='agent-website-update'),
    path('websites/<int:website_id>/delete/', views.delete_agent_website, name='agent-website-delete'),
]