from django.contrib import admin
from .models import (
    Conversation,
    ConversationMessages,
    ConversationAttachments,
    ConversationTag,
    ConversationNotes,
    ConversationFeedback,
)

@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ('conversation_id', 'conversation_name', 'agent', 'is_favorite')
    list_filter = ('is_favorite', 'created_at')
    search_fields = ('conversation_name', 'agent__user__username','agent__name')
    ordering = ('-conversation_id',)
    readonly_fields = ('conversation_id',)

@admin.register(ConversationMessages)
class ConversationMessagesAdmin(admin.ModelAdmin):
    list_display = ('message_id', 'conversation', 'sender_id', 'sender_type', 'message_text', 'message_time', 'is_read', 'is_deleted', 'is_archived')
    list_filter = ('is_read', 'is_deleted', 'is_archived','conversation__conversation_name', 'conversation__agent__name')
    search_fields = ('conversation__conversation_name', "conversation__agent__user__username", "conversation__agent__name")
    ordering = ('-message_id',)
    readonly_fields = ("message_id",)

@admin.register(ConversationAttachments)
class ConversationAttachmentsAdmin(admin.ModelAdmin):
    list_display = ('attachment_id', 'message', 'attachment_name', 'attachment_path', 'attachment_type', 'attachment_size')
    list_filter = ('attachment_type',)
    search_fields = ('message__conversation__conversation_name', 'attachment_name')
    ordering = ('-attachment_id',)
    readonly_fields = ('attachment_id',)

@admin.register(ConversationTag)
class ConversationTagAdmin(admin.ModelAdmin):
    list_display = ('tag_id', 'conversation', 'tag_name', 'created_at', 'updated_at', 'is_active', 'is_deleted')
    list_filter = ('is_active', 'is_deleted')
    search_fields = ('conversation__conversation_name', 'tag_name')
    ordering = ('-tag_id',)
    readonly_fields = ('tag_id',)

@admin.register(ConversationNotes)
class ConversationNotesAdmin(admin.ModelAdmin):
    list_display = ('note_id', 'conversation', 'note_text', 'created_at', 'updated_at', 'is_active', 'is_deleted')
    list_filter = ('is_active', 'is_deleted')
    search_fields = ('conversation__conversation_name', 'note_text')
    ordering = ('-note_id',) 
    readonly_fields = ('note_id',)

@admin.register(ConversationFeedback)
class ConversationFeedbackAdmin(admin.ModelAdmin):
    list_display = ('feedback_id', 'conversation', 'feedback_text', 'rating', 'created_at', 'updated_at', 'is_deleted')
    list_filter = ('is_deleted',)
    search_fields = ('conversation__conversation_name', 'feedback_text')
    ordering = ('-feedback_id',) 
    readonly_fields = ('feedback_id',)