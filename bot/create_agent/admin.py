from django.contrib import admin
from .models import Agent, AgentDocuments, AgentIntegrations, AgentEmbeddings

class AgentAdmin(admin.ModelAdmin):
    list_display = ('agent_id', 'name', 'user', 'visibility', 'created_at', 'is_deleted')
    search_fields = ('name', 'description', 'user__username')
    list_filter = ('visibility', 'is_deleted', 'is_archived', 'created_at')
    ordering = ('-created_at',)
    readonly_fields = ('agent_id', 'created_at', 'updated_at')

class AgentDocumentsAdmin(admin.ModelAdmin):
    list_display = ('document_id', 'agent', 'document_name', 'document_format', 'created_at')
    search_fields = ('document_name', 'agent__name')
    list_filter = ('document_format', 'created_at')
    ordering = ('-created_at',)
    readonly_fields = ('document_id', 'created_at', 'updated_at')

class AgentIntegrationsAdmin(admin.ModelAdmin):
    list_display = ('integration_id', 'agent', 'integration_name', 'integration_category', 'integration_auth_status', 'created_at')
    search_fields = ('agent__name', 'integration_name', 'integration_category')
    list_filter = ('integration_category', 'integration_auth_status', 'created_at')
    ordering = ('-created_at',)
    readonly_fields = ('integration_id', 'created_at', 'updated_at')

class AgentEmbeddingsAdmin(admin.ModelAdmin):
    list_display = ('embedding_id', 'agent', 'embedding_model', 'created_at')
    search_fields = ('agent__name', 'embedding_model')
    list_filter = ('embedding_model', 'created_at')
    ordering = ('-created_at',)
    readonly_fields = ('embedding_id', 'created_at', 'updated_at')

admin.site.register(Agent, AgentAdmin)
admin.site.register(AgentDocuments, AgentDocumentsAdmin)
admin.site.register(AgentIntegrations, AgentIntegrationsAdmin)
admin.site.register(AgentEmbeddings, AgentEmbeddingsAdmin)
