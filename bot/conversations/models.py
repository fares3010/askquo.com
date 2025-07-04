from django.db import models
from django.utils import timezone
from create_agent.models import Agent
from create_agent.models import FrontendIntegrationClient

class Conversation(models.Model):
    agent = models.ForeignKey(Agent, on_delete=models.CASCADE, related_name='agent_conversations')
    conversation_id = models.AutoField(primary_key=True)
    conversation_name = models.CharField(max_length=255, blank=True, null=True)
    frontend_client = models.ForeignKey(FrontendIntegrationClient, on_delete=models.CASCADE, related_name='frontend_client_conversations', blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_favorite = models.BooleanField(default=False)
    is_archived = models.BooleanField(default=False)
    is_deleted = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)


    def __str__(self):
        return self.conversation_name if self.conversation_name else str("unknown")    
    
    @property
    def last_message_obj(self):
        """Get the most recent message in the conversation."""
        return self.messages.order_by('-message_time').first()

    @property
    def get_ordered_messages(self):
        """Get all messages ordered by most recent first."""
        return self.messages.order_by('-message_time')
    
    def last_message_text(self):
        """Get the text content of the most recent message."""
        last = self.last_message_obj
        return last.message_text if last else None
    
    def user_response_time(self):
        """Calculate time between conversation creation and first user message."""
        last = self.last_message_obj
        if last and last.sender_type == 'user':
            return last.message_time - self.created_at
        return None
    
    def last_message_time(self):
        """Get the timestamp of the most recent message."""
        last = self.last_message_obj
        return last.message_time if last else None
    
    def check_last_message_is_read(self):
        """Check if the most recent message has been read."""
        last = self.last_message_obj
        return last.is_read if last else False
    def unread_count(self):    
        return self.messages.filter(is_read=False).count()
    
    def feedback_rate(self):
        feedbacks = self.feedback.all()
        if feedbacks.exists():
            return sum([feedback.rating for feedback in feedbacks]) / len(feedbacks)
        return None
    
    def check_is_active(self):
        if self.last_message_time() and timezone.now() - self.last_message_time() < timezone.timedelta(minutes=1):
            return True
        else:
            return False
    def check_is_active_with_time(self, time):
        if self.last_message_time() and timezone.now() - self.last_message_time() < time:
            self.is_active = True
            self.save()
            return True
        else:
            self.is_active = False
            self.save()
            return False    
    
    def check_is_deleted(self):
        if self.messages.filter(is_deleted=True).exists():
            return True
        else:
            return False
    def check_is_archived(self):
        if self.messages.filter(is_archived=True).exists():
            return True
        else:
            return False
         
    def get_conversation_details(self):
        return {
            "conversation_id": self.conversation_id,
            "agent": self.agent,
            "conversation_name": self.conversation_name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "is_favorite": self.is_favorite,
            "last_message": self.last_message_text(),
            "last_message_time": self.last_message_time(),
            "unread_count": self.unread_count(),

        }

    def number_of_messages(self):
        return self.messages.count()
    
    def number_of_agent_messages(self):
        return self.messages.filter(sender_type='agent').count()
    
    def number_of_attachments(self):
        return self.messages.filter(attachments__isnull=False).count()
    
    def number_of_tags(self):
        return self.tags.count()
    
    def number_of_notes(self):
        return self.notes.count()
    
    def number_of_feedback(self):
        return self.feedback.count()
    
    def get_tags(self):
        return self.tags.all()

        
    def get_notes(self):
        return self.notes.all()


    def get_feedback(self):
        return self.feedback.all()
    
    def get_participants(self):
        return {
            "agent": self.agent.name
        }
    
    def get_status(self):
        return {
            "is_active": self.check_is_active(),
            "is_deleted": self.check_is_deleted(),
            "is_archived": self.check_is_archived(),
            "unread_count": self.unread_count(),
            "last_message": self.last_message_text(),
        }


class ConversationMessages(models.Model):
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name='messages')
    message_id = models.AutoField(primary_key=True)
    sender_id = models.IntegerField(blank=True, null=True)
    sender_type = models.CharField(max_length=50, blank=True, null=True)
    message_text = models.TextField(blank=True, null=True)
    message_type = models.CharField(max_length=50, blank=True, null=True)
    message_time = models.DateTimeField(auto_now_add=True)
    is_read = models.BooleanField(default=False)
    is_deleted = models.BooleanField(default=False)
    is_archived = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True, null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True, null=True, blank=True)



    def __str__(self):
        return self.message_text if self.message_text else str("unknown")
    
    def get_message_details(self):
        return {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "sender_type": self.sender_type,
            "message_text": self.message_text,
            "message_type": self.message_type,
            "message_time": self.message_time,
            "is_read": self.is_read,
            "is_deleted": self.is_deleted,
            "is_archived": self.is_archived,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    

    def get_attachments(self):
        if self.attachments.exists():
            return self.attachments.all()
        else:
            return None
        
    def attachment_count(self):
        return self.attachments.count()
    
    def mark_as_read(self):
        if not self.is_read:
            self.is_read = True
            self.save()

    def soft_delete(self):
        self.is_deleted = True
        self.save()
       

class ConversationAttachments(models.Model):
    message = models.ForeignKey(
        ConversationMessages,
        on_delete=models.CASCADE,
        related_name='attachments',
        help_text="The message this attachment belongs to."
    )
    attachment_id = models.AutoField(primary_key=True)
    attachment_name = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        help_text="The name of the attachment file."
    )
    attachment_path = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        help_text="The storage path or URL of the attachment."
    )
    attachment_type = models.CharField(
        max_length=50,
        blank=True,
        null=True,
        help_text="The MIME type or file type of the attachment."
    )
    attachment_size = models.PositiveIntegerField(
        blank=True,
        null=True,
        help_text="The size of the attachment in bytes."
    )

    class Meta:
        verbose_name = "Conversation Attachment"
        verbose_name_plural = "Conversation Attachments"
        ordering = ['-attachment_id']

    def __str__(self):
        return self.attachment_name or "unknown"

    def get_attachment_details(self):
        """
        Returns a dictionary of the attachment's details.
        """
        return {
            "attachment_id": self.attachment_id,
            "attachment_name": self.attachment_name,
            "attachment_path": self.attachment_path,
            "attachment_type": self.attachment_type,
            "attachment_size": self.attachment_size,
        }

class ConversationTag(models.Model):
    conversation = models.ForeignKey(
        Conversation,
        on_delete=models.CASCADE,
        related_name='tags',
        help_text="The conversation this tag is associated with."
    )
    tag_id = models.AutoField(primary_key=True)
    tag_name = models.CharField(
        max_length=255,
        blank=False,
        null=False,
        help_text="The name of the tag."
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="Timestamp when the tag was created."
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        help_text="Timestamp when the tag was last updated."
    )
    is_active = models.BooleanField(
        default=True,
        help_text="Indicates if the tag is active."
    )
    is_deleted = models.BooleanField(
        default=False,
        help_text="Indicates if the tag is deleted."
    )

    class Meta:
        verbose_name = "Conversation Tag"
        verbose_name_plural = "Conversation Tags"
        ordering = ['-created_at', '-tag_id']
        unique_together = ('conversation', 'tag_name')

    def __str__(self):
        return self.tag_name if self.tag_name else str("unknown")
    
    def get_tag_details(self):
        return {
            "tag_id": self.tag_id,
            "tag_name": self.tag_name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "is_active": self.is_active,
            "is_deleted": self.is_deleted,
        }


class ConversationNotes(models.Model):
    conversation = models.ForeignKey(
        Conversation,
        on_delete=models.CASCADE,
        related_name='notes',
        help_text="The conversation this note is associated with."
    )
    note_id = models.AutoField(primary_key=True)
    note_text = models.TextField(
        blank=True,
        null=True,
        help_text="The content of the note."
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="Timestamp when the note was created."
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        help_text="Timestamp when the note was last updated."
    )
    is_active = models.BooleanField(
        default=True,
        help_text="Indicates if the note is active."
    )
    is_deleted = models.BooleanField(
        default=False,
        help_text="Indicates if the note is deleted."
    )

    class Meta:
        verbose_name = "Conversation Note"
        verbose_name_plural = "Conversation Notes"
        ordering = ['-created_at', '-note_id']
        unique_together = ('conversation', 'note_text')

    def __str__(self):
        return self.note_text if self.note_text else "unknown"

    def get_note_details(self):
        return {
            "note_id": self.note_id,
            "note_text": self.note_text,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "is_active": self.is_active,
            "is_deleted": self.is_deleted,
        }

class ConversationFeedback(models.Model):
    conversation = models.ForeignKey(
        Conversation,
        on_delete=models.CASCADE,
        related_name='feedback',
        help_text="The conversation this feedback is associated with."
    )
    feedback_id = models.AutoField(primary_key=True)
    feedback_text = models.TextField(
        blank=True,
        null=True,
        help_text="The content of the feedback."
    )
    rating = models.PositiveSmallIntegerField(
        blank=True,
        null=True,
        help_text="Optional rating value (e.g., 1-5)."
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="Timestamp when the feedback was created."
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        help_text="Timestamp when the feedback was last updated."
    )
    is_deleted = models.BooleanField(
        default=False,
        help_text="Indicates if the feedback is deleted."
    )

    class Meta:
        verbose_name = "Conversation Feedback"
        verbose_name_plural = "Conversation Feedback"
        ordering = ['-created_at', '-feedback_id']
        unique_together = ('conversation', 'feedback_text')

    def __str__(self):
        return self.feedback_text if self.feedback_text else "unknown"

    def get_feedback_details(self):
        return {
            "feedback_id": self.feedback_id,
            "feedback_text": self.feedback_text,
            "rating": self.rating,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "is_deleted": self.is_deleted,
        }