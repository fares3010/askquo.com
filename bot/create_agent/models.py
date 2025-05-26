from django.db import models
from django.conf import settings
from django.utils import timezone
from datetime import datetime, timedelta
from urllib.parse import urlparse
import numpy as np



class Agent(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='agents',
        help_text="The user who owns this agent."
    )
    agent_id = models.AutoField(
        primary_key=True,
        help_text="Primary key for the agent."
    )
    name = models.CharField(
        max_length=100,
        help_text="Name of the agent."
    )
    description = models.TextField(
        blank=True,
        null=True,
        help_text="Optional description of the agent."
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="Timestamp when the agent was created."
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        help_text="Timestamp when the agent was last updated."
    )
    is_deleted = models.BooleanField(
        default=False,
        help_text="Indicates if the agent is deleted."
    )
    is_archived = models.BooleanField(
        default=False,
        help_text="Indicates if the agent is archived."
    )
    is_favorite = models.BooleanField(
        default=False,
        help_text="Indicates if the agent is marked as favorite."
    )
    visibility = models.CharField(
        max_length=50,
        choices=[('public', 'Public'), ('private', 'Private')],
        default='private',
        help_text="Visibility of the agent: public or private."
    )
    avatar_url = models.URLField(
        max_length=200,
        blank=True,
        null=True,
        help_text="Optional URL for the agent's avatar."
    )
    configuration = models.JSONField(
        blank=True,
        null=True,
        help_text="Optional JSON configuration for the agent."
    )

    def __str__(self):
        return f"{self.name} (ID: {self.agent_id})" if self.name else f"Agent {self.agent_id}"

    def is_active(self):
        """
        Returns True if the agent has had a conversation with a message
        in the last 15 days, otherwise False.
        """
        last_conversation = self.agent_conversations.order_by('-updated_at').first()
        if not last_conversation:
            return False

        # Try to get the last message time from the conversation
        last_msg_time = None
        if hasattr(last_conversation, 'last_message_time') and callable(last_conversation.last_message_time):
            last_msg_time = last_conversation.last_message_time()
        elif hasattr(last_conversation, 'updated_at'):
            last_msg_time = last_conversation.updated_at

        if last_msg_time:
            return timezone.now() - last_msg_time < timezone.timedelta(days=15)
        return False

    def conversation_count(self):
        """
        Returns the number of conversations associated with this agent.
        """
        return self.agent_conversations.count()

    def get_documents_summary(self):
        """
        Returns a summary list of all documents associated with this agent.
        """
        summary = []
        for doc in self.documents.all():
            size_kb = None
            if doc.document_size is not None:
                try:
                    size_kb = round(float(doc.document_size) / 1024, 2)
                except (TypeError, ValueError):
                    size_kb = None
            summary.append({
                "document_id": doc.document_id,
                "name": doc.document_name,
                "format": doc.document_format,
                "size_kb": size_kb
            })
        return summary


# Create your models here.
class AgentDocuments(models.Model):
    agent = models.ForeignKey('Agent', on_delete=models.CASCADE, related_name='documents')
    document_id = models.AutoField(primary_key=True)
    document_name = models.CharField(max_length=255, blank=True, null=True)
    document_description = models.TextField(blank=True, null=True)
    document_url = models.URLField(max_length=500)  # Increased max_length for longer URLs
    document_size = models.BigIntegerField(blank=True, null=True)  # Use BigIntegerField for large files
    document_format = models.CharField(max_length=50, blank=True, null=True)  # e.g., PDF, DOCX
    document_language = models.CharField(max_length=50, blank=True, null=True)
    document_tags = models.JSONField(blank=True, null=True, default=list)  # Default to empty list for tags
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    is_deleted = models.BooleanField(default=False)
    is_archived = models.BooleanField(default=False)
    meta_data = models.JSONField(blank=True, null=True, default=dict)  # Default to empty dict

    def __str__(self):
        # Return a meaningful string representation, fallback to ID if name is missing
        return self.document_name or f"Document {self.document_id}"

    def get_document_details(self):
        # Return a dictionary with all relevant document details, using formatted_size property
        return {
            "document_id": self.document_id,
            "document_name": self.document_name,
            "document_description": self.document_description,
            "document_url": self.document_url,
            "document_size": self.formatted_size,
            "document_format": self.document_format,
            "document_language": self.document_language,
            "document_tags": self.document_tags if self.document_tags is not None else [],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "is_expired": self.is_expired(),
            "file_type_icon": self.get_file_type_icon(),
        }

    @property
    def formatted_size(self):
        # Return a human-readable file size string
        if self.document_size is None:
            return "Unknown"
        size = float(self.document_size)
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:.2f} {unit}"
            size /= 1024.0
        return f"{size:.2f} PB"

    def is_expired(self):
        # Use meta_data or created_at to decide expiration policy
        # If meta_data contains 'expiration_days', use it; otherwise default to 365 days
        expiration_days = 365
        if self.meta_data and isinstance(self.meta_data, dict):
            expiration_days = self.meta_data.get('expiration_days', 365)
        return self.created_at < timezone.now() - timezone.timedelta(days=expiration_days)

    def get_file_type_icon(self):
        # Return an emoji icon based on the document format
        ext = (self.document_format or '').lower()
        icon_map = {
            'pdf': '📄',
            'docx': '📝',
            'doc': '📝',
            'txt': '🗒️',
            'xlsx': '📊',
            'xls': '📊',
            'csv': '📑',
            'ppt': '📈',
            'pptx': '📈',
            'jpg': '🖼️',
            'jpeg': '🖼️',
            'png': '🖼️',
            'gif': '🖼️',
            'zip': '🗜️',
            'rar': '🗜️',
            'json': '🔢',
            'xml': '🔤',
        }
        return icon_map.get(ext, '📁')
        

    
class AgentIntegrations(models.Model):
    agent = models.ForeignKey(
        Agent,
        on_delete=models.CASCADE,
        related_name='agent_integrations',
        help_text="The agent this integration belongs to."
    )
    integration_id = models.AutoField(
        primary_key=True,
        help_text="Primary key for the integration."
    )
    integration_name = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        help_text="Name of the integration."
    )
    integration_category = models.CharField(
        max_length=100,
        blank=True,
        null=True,
        help_text="Category of the integration, e.g., CRM, ERP, etc."
    )
    integration_priority = models.IntegerField(
        default=0,
        help_text="Priority for UI or logic that chooses a default integration."
    )
    integration_logo_url = models.URLField(
        max_length=200,
        blank=True,
        null=True,
        help_text="URL for the integration's logo (for frontend display)."
    )
    integration_description = models.TextField(
        blank=True,
        null=True,
        help_text="Description of the integration."
    )
    integration_url = models.URLField(
        max_length=200,
        blank=True,
        null=True,
        help_text="URL to the integration's main page or API."
    )
    integration_api_key = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        help_text="API key for the integration, if applicable."
    )
    integration_api_secret = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        help_text="API secret for the integration, if applicable."
    )
    integration_auth_type = models.CharField(
        max_length=50,
        blank=True,
        null=True,
        help_text="Authentication type, e.g., OAuth, API Key."
    )
    integration_auth_url = models.URLField(
        max_length=200,
        blank=True,
        null=True,
        help_text="URL for authentication (OAuth, etc.)."
    )
    integration_auth_token = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        help_text="Authentication token for the integration."
    )
    integration_token_type = models.CharField(
        max_length=50,
        blank=True,
        null=True,
        help_text="Type of token, e.g., Bearer, JWT."
    )
    integration_auth_error_time = models.DateTimeField(
        blank=True,
        null=True,
        help_text="Timestamp of the last authentication error."
    )
    integration_auth_scope = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        help_text="OAuth scopes or permissions for the integration."
    )
    integration_auth_expiry = models.DateTimeField(
        blank=True,
        null=True,
        help_text="When the current auth token expires."
    )
    integration_auth_status = models.CharField(
        max_length=50,
        blank=True,
        null=True,
        help_text="Status of the authentication (e.g., Valid, Expired, Error)."
    )
    integration_auth_response = models.JSONField(
        blank=True,
        null=True,
        help_text="Raw response from the authentication endpoint."
    )
    integration_auth_error = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        help_text="Error message from the last authentication attempt."
    )
    integration_refresh_token = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        help_text="Refresh token for the integration, if applicable."
    )
    integration_auth_code = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        help_text="Auth code for OAuth flows, if required."
    )
    configuration = models.JSONField(
        blank=True,
        null=True,
        help_text="Additional configuration for the integration."
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="Timestamp when the integration was created."
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        help_text="Timestamp when the integration was last updated."
    )
    is_active = models.BooleanField(
        default=True,
        help_text="Whether the integration is currently active."
    )
    is_deleted = models.BooleanField(
        default=False,
        help_text="Whether the integration is deleted (soft delete)."
    )
    meta_data = models.JSONField(
        blank=True,
        null=True,
        help_text="Additional metadata for the integration."
    )

    class Meta:
        verbose_name = "Agent Integration"
        verbose_name_plural = "Agent Integrations"
        ordering = ['-created_at']

    def __str__(self):
        # Return a meaningful string representation for the integration
        if self.integration_name:
            return f"{self.integration_name} (ID: {self.integration_id})"
        return f"Integration {self.integration_id}"

    def is_token_expired(self):
        """
        Returns True if the integration's auth token is expired or missing expiry.
        """
        if not hasattr(self, 'integration_auth_expiry') or self.integration_auth_expiry is None:
            return True
        return self.integration_auth_expiry <= timezone.now()

    def get_status(self):
        """
        Returns a dictionary summarizing the integration's status.
        """
        return {
            "name": self.integration_name,
            "active": self.is_active,
            "auth_status": self.integration_auth_status,
            "token_expired": self.is_token_expired(),
            "last_updated": self.updated_at,
            "has_auth_error": self.has_auth_error(),
        }

    def update_auth_token(self, token, expiry=None):
        """
        Updates the integration's auth token and expiry, and marks status as valid.
        """
        self.integration_auth_token = token
        if expiry is not None:
            self.integration_auth_expiry = expiry
        self.integration_auth_status = "Valid"
        self.save(update_fields=["integration_auth_token", "integration_auth_expiry", "integration_auth_status", "updated_at"])

    def soft_delete(self):
        """
        Soft deletes the integration by marking it as inactive and deleted.
        """
        self.is_deleted = True
        self.is_active = False
        self.save(update_fields=["is_deleted", "is_active", "updated_at"])

    def get_public_details(self):
        """
        Returns a dictionary of public-facing integration details.
        """
        return {
            "integration_name": self.integration_name,
            "integration_description": self.integration_description,
            "integration_url": self.integration_url,
            "auth_type": self.integration_auth_type,
            "status": self.integration_auth_status,
            "is_active": self.is_active,
            "token_masked": bool(self.integration_auth_token),
        }

    def has_auth_error(self):
        """
        Returns True if there is an authentication error message.
        """
        return bool(self.integration_auth_error)

    
class AgentQaPairs(models.Model):
    agent = models.ForeignKey(
        'Agent',
        on_delete=models.CASCADE,
        related_name='qa_pairs',
        help_text="The agent this QA pair belongs to."
    )
    qa_pair_id = models.AutoField(
        primary_key=True,
        help_text="Primary key for the QA pair."
    )
    qa_pair_name = models.CharField(
        max_length=255,
        unique=True,
        blank=True,
        null=True,
        help_text="Optional unique name for the QA pair."
    )
    question_type = models.CharField(
        max_length=50,
        blank=True,
        null=True,
        help_text="Type of question, e.g., FAQ, General Knowledge."
    )
    question = models.TextField(
        help_text="The question text."
    )
    answer = models.TextField(
        help_text="The answer text."
    )
    tags = models.JSONField(
        blank=True,
        null=True,
        default=list,
        help_text="List of tags for categorization, e.g., ['billing', 'technical']."
    )
    question_language = models.CharField(
        max_length=50,
        blank=True,
        null=True,
        help_text="Language of the question, e.g., English, Spanish."
    )
    answer_language = models.CharField(
        max_length=50,
        blank=True,
        null=True,
        help_text="Language of the answer, e.g., English, Spanish."
    )
    question_format = models.CharField(
        max_length=50,
        blank=True,
        null=True,
        help_text="Format of the question, e.g., text, audio, video."
    )
    answer_format = models.CharField(
        max_length=50,
        blank=True,
        null=True,
        help_text="Format of the answer, e.g., text, audio, video."
    )
    question_size = models.IntegerField(
        blank=True,
        null=True,
        help_text="Size of the question in bytes."
    )
    answer_size = models.IntegerField(
        blank=True,
        null=True,
        help_text="Size of the answer in bytes."
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="Timestamp when the QA pair was created."
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        help_text="Timestamp when the QA pair was last updated."
    )
    is_active = models.BooleanField(
        default=True,
        help_text="Indicates if the QA pair is active."
    )
    is_deleted = models.BooleanField(
        default=False,
        help_text="Indicates if the QA pair is deleted."
    )
    meta_data = models.JSONField(
        blank=True,
        null=True,
        default=dict,
        help_text="Optional metadata for the QA pair."
    )

    def __str__(self):
        # Return a concise, informative string representation
        question_preview = (self.question[:47] + "...") if self.question and len(self.question) > 50 else (self.question or "No Question")
        return f"Q: {question_preview}"

    def is_faq(self):
        # Returns True if this QA pair is marked as an FAQ
        return getattr(self, "question_type", None) == "FAQ"

    def summary(self, q_len=50, a_len=50):
        # Returns a short summary of the question and answer
        q = (self.question[:q_len] + "...") if self.question and len(self.question) > q_len else (self.question or "")
        a = (self.answer[:a_len] + "...") if hasattr(self, "answer") and self.answer and len(self.answer) > a_len else (getattr(self, "answer", "") or "")
        return f"Q: {q} A: {a}"

    def mark_inactive(self, save=True):
        # Mark the QA pair as inactive
        self.is_active = False
        if save:
            self.save(update_fields=["is_active"])

    def mark_deleted(self, save=True):
        # Mark the QA pair as deleted
        self.is_deleted = True
        if save:
            self.save(update_fields=["is_deleted"])

    
class AgentTexts(models.Model):
    agent = models.ForeignKey(
        Agent,
        on_delete=models.CASCADE,
        related_name='texts',
        help_text="The agent this text belongs to."
    )
    text_id = models.AutoField(
        primary_key=True,
        help_text="Primary key for the text entry."
    )
    text_name = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        help_text="Optional name or label for the text."
    )
    text = models.TextField(
        help_text="The main text content."
    )
    text_language = models.CharField(
        max_length=50,
        blank=True,
        null=True,
        help_text="Language of the text, e.g., English, Spanish."
    )
    text_type = models.CharField(
        max_length=50,
        blank=True,
        null=True,
        help_text='Type of text, e.g., "system", "greeting", "error_response".'
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="Timestamp when the text was created."
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        help_text="Timestamp when the text was last updated."
    )
    is_active = models.BooleanField(
        default=True,
        help_text="Indicates if the text is active."
    )
    is_deleted = models.BooleanField(
        default=False,
        help_text="Indicates if the text is deleted."
    )
    is_archived = models.BooleanField(
        default=False,
        help_text="Indicates if the text is archived."
    )
    meta_data = models.JSONField(
        blank=True,
        null=True,
        default=dict,
        help_text="Optional metadata for the text."
    )

    def __str__(self):
        """
        Return a concise string representation of the text entry.
        Prefer the text_name if available, otherwise show the first 50 characters of the text.
        """
        if self.text_name:
            return f"{self.text_name} ({self.text[:30]}...)" if self.text and len(self.text) > 30 else self.text_name
        return (self.text[:50] + "...") if self.text and len(self.text) > 50 else (self.text or "")

    def short_text(self, length=30):
        """
        Return a shortened version of the text, appending '...' if truncated.
        """
        if not self.text:
            return ""
        return self.text[:length] + "..." if len(self.text) > length else self.text

    def mark_deleted(self, save=True):
        """
        Mark this text entry as deleted. Optionally save the change immediately.
        """
        self.is_deleted = True
        if save:
            self.save(update_fields=["is_deleted"])

    def is_system_text(self):
        """
        Return True if this text entry is of type 'system'.
        """
        return (self.text_type or "").lower() == "system"

    
class AgentVectorsDatabase(models.Model):
    agent = models.ForeignKey(
        Agent,
        on_delete=models.CASCADE,
        related_name='vectors',
        help_text="The agent this vector database belongs to."
    )
    vdb_id = models.AutoField(
        primary_key=True,
        help_text="Primary key for the vector database."
    )
    database_path = models.CharField(
        max_length=255,
        help_text="Filesystem or URI path to the vector database."
    )
    vector_count = models.PositiveIntegerField(
        default=0,
        help_text="Number of vectors currently stored in the database."
    )
    vector_dimension = models.PositiveIntegerField(
        blank=True,
        null=True,
        help_text="Dimension of the vectors (for validation and index building)."
    )
    embedding_model = models.CharField(
        max_length=100,
        blank=True,
        null=True,
        help_text="Model used to generate vectors (e.g., openai-ada, sentence-transformers)."
    )
    storage_backend = models.CharField(
        max_length=100,
        blank=True,
        null=True,
        help_text="Backend where vectors are stored (e.g., FAISS, Pinecone, Chroma, Weaviate)."
    )
    index_type = models.CharField(
        max_length=100,
        blank=True,
        null=True,
        help_text="Type of index used (e.g., flat, IVF, HNSW)."
    )
    last_indexed_at = models.DateTimeField(
        blank=True,
        null=True,
        help_text="Timestamp of the last indexing or refresh operation."
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="Timestamp when this vector database entry was created."
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        help_text="Timestamp when this vector database entry was last updated."
    )
    is_active = models.BooleanField(
        default=True,
        help_text="Indicates if the vector database is active."
    )
    is_deleted = models.BooleanField(
        default=False,
        help_text="Indicates if the vector database is deleted."
    )
    is_archived = models.BooleanField(
        default=False,
        help_text="Indicates if the vector database is archived."
    )
    meta_data = models.JSONField(
        blank=True,
        null=True,
        default=dict,
        help_text="Optional metadata for the vector database."
    )

    def __str__(self):
        agent_name = getattr(self.agent, "name", None) or "Unknown Agent"
        db_path = getattr(self, "database_path", None)
        db_path_display = db_path[:40] + "..." if db_path and len(db_path) > 40 else (db_path or "No Path")
        return f"{agent_name} VDB: {db_path_display}"

    def is_empty(self):
        return (self.vector_count or 0) == 0

    def get_embedding_model(self):
        return self.embedding_model or "default-model"

    def get_index_summary(self):
        return {
            "vector_count": self.vector_count,
            "dimension": self.vector_dimension,
            "model": self.embedding_model,
            "backend": self.storage_backend,
            "indexed_at": self.last_indexed_at,
        }

    def mark_indexed(self):
        from django.utils import timezone
        self.last_indexed_at = timezone.now()
        self.save(update_fields=["last_indexed_at"])

    def increment_vector_count(self, count=1):
        if self.vector_count is None:
            self.vector_count = 0
        self.vector_count += count
        self.save(update_fields=["vector_count"])

    def clear_vectors(self):
        self.vector_count = 0
        self.last_indexed_at = None
        self.save(update_fields=["vector_count", "last_indexed_at"])

    def is_backend_supported(self):
        supported = {'faiss', 'pinecone', 'chroma', 'weaviate'}
        backend = (self.storage_backend or "").strip().lower()
        return backend in supported

    def display_name(self):
        agent_name = getattr(self.agent, "name", None) or "Unknown Agent"
        model = self.embedding_model or "Unknown Model"
        return f"{agent_name} - {model}"

    def is_ready(self):
        return bool(self.is_active and not self.is_deleted and (self.vector_count or 0) > 0)

    def needs_reindexing(self, threshold_days=7):
        from django.utils import timezone
        if not self.last_indexed_at:
            return True
        delta = timezone.now() - self.last_indexed_at
        return delta.days > threshold_days


        
class AgentEmbeddings(models.Model):
    agent = models.ForeignKey(
        Agent,
        on_delete=models.CASCADE,
        related_name='embeddings',
        help_text="The agent this embedding belongs to."
    )
    embedding_id = models.AutoField(
        primary_key=True,
        help_text="Primary key for the embedding."
    )
    embedding_model = models.CharField(
        max_length=100,
        blank=True,
        null=True,
        help_text="Name of the embedding model (e.g., openai-ada, sentence-transformers, etc.)."
    )
    vector_dimension = models.IntegerField(
        blank=True,
        null=True,
        help_text="Dimension of the embedding vector."
    )
    similarity_score = models.FloatField(
        blank=True,
        null=True,
        help_text="Similarity score with respect to a query vector."
    )
    source_url = models.URLField(
        blank=True,
        null=True,
        help_text="Original source URL if the object was scraped or fetched online."
    )
    language = models.CharField(
        max_length=50,
        blank=True,
        null=True,
        help_text="Language of the text (e.g., English, Spanish)."
    )
    token_count = models.IntegerField(
        blank=True,
        null=True,
        help_text="Number of tokens in the text."
    )
    tags = models.JSONField(
        blank=True,
        null=True,
        default=list,
        help_text="Tags associated with the embedding (e.g., ['finance', 'health'])."
    )
    generated_by_user = models.BooleanField(
        default=True,
        help_text="Indicates if the embedding was generated by a user or an automated process."
    )
    object_id = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        help_text="ID of the object (e.g., document, text, etc.)."
    )
    object_type = models.CharField(
        max_length=50,
        blank=True,
        null=True,
        help_text="Type of the object (e.g., document, text, etc.)."
    )
    object_name = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        help_text="Name of the object (e.g., document name, text name, etc.)."
    )
    embedding_vector = models.JSONField(
        blank=True,
        null=True,
        help_text="The embedding vector as a list of floats."
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="Timestamp when the embedding was created."
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        help_text="Timestamp when the embedding was last updated."
    )
    is_active = models.BooleanField(
        default=True,
        help_text="Indicates if the embedding is active."
    )
    is_deleted = models.BooleanField(
        default=False,
        help_text="Indicates if the embedding is deleted."
    )
    is_archived = models.BooleanField(
        default=False,
        help_text="Indicates if the embedding is archived."
    )
    meta_data = models.JSONField(
        blank=True,
        null=True,
        default=dict,
        help_text="Optional metadata for the embedding."
    )

    def __str__(self):
        """
        Return a concise, human-readable string representation of the embedding.
        Shows object type, name, and a preview of the vector.
        """
        name = self.display_name()
        vector_preview = (
            f"{self.embedding_vector[:5]}..." if self.embedding_vector and isinstance(self.embedding_vector, list)
            else "No vector"
        )
        return f"{name} | Vector: {vector_preview}"

    def vector_length(self):
        """
        Returns the length of the embedding vector, or 0 if not present.
        """
        if isinstance(self.embedding_vector, list):
            return len(self.embedding_vector)
        return 0

    def is_valid_vector(self):
        """
        Checks if the embedding vector is a list of numbers (int or float).
        """
        return (
            isinstance(self.embedding_vector, list)
            and all(isinstance(x, (int, float)) for x in self.embedding_vector)
            and len(self.embedding_vector) > 0
        )

    def as_numpy_array(self):
        """
        Returns the embedding vector as a numpy array, or None if not present/invalid.
        """
        if not self.is_valid_vector():
            return None
        return np.array(self.embedding_vector, dtype=float)

    @staticmethod
    def cosine_similarity(vec1, vec2):
        """
        Compute cosine similarity between two vectors (lists or numpy arrays).
        Returns 0.0 if either vector is empty or invalid.
        """
        v1 = np.array(vec1, dtype=float)
        v2 = np.array(vec2, dtype=float)
        if v1.size == 0 or v2.size == 0:
            return 0.0
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    def is_stale(self, days=30):
        """
        Returns True if the embedding has not been updated in the given number of days.
        """
        if not self.updated_at:
            return True
        now = timezone.now()
        delta = now - self.updated_at
        return delta.days > days

    def is_for_object_type(self, type_str):
        """
        Returns True if the embedding is for the given object type (case-insensitive).
        """
        if self.object_type is None or type_str is None:
            return False
        return self.object_type.lower() == type_str.lower()

    def display_name(self):
        """
        Returns a display name combining object type and object name.
        """
        return f"{self.object_type or 'Unknown'} - {self.object_name or 'Unnamed'}"





    
class AgentWebsites(models.Model):
    agent = models.ForeignKey(
        Agent,
        on_delete=models.CASCADE,
        related_name='websites',
        help_text="The agent this website belongs to."
    )
    website_id = models.AutoField(
        primary_key=True,
        help_text="Primary key for the website."
    )
    website_url = models.URLField(
        max_length=500,
        help_text="URL of the website."
    )
    website_name = models.CharField(
        max_length=255,
        help_text="Name of the website."
    )
    website_type = models.CharField(
        max_length=100,
        blank=True,
        null=True,
        help_text='Type of website, e.g., "blog", "e-commerce", "portfolio".'
    )
    crawl_status = models.CharField(
        max_length=50,
        blank=True,
        null=True,
        help_text='Status of crawling: e.g., "pending", "success", "failed", "partial".'
    )
    last_crawled_at = models.DateTimeField(
        blank=True,
        null=True,
        help_text="Timestamp of the last crawl."
    )
    crawl_frequency = models.CharField(
        max_length=50,
        blank=True,
        null=True,
        help_text='Crawl frequency, e.g., "daily", "weekly", "monthly", or number of days.'
    )
    content_language = models.CharField(
        max_length=50,
        blank=True,
        null=True,
        help_text="Language of the website content, e.g., 'English', 'Spanish'."
    )
    page_limit = models.PositiveIntegerField(
        blank=True,
        null=True,
        help_text="Maximum number of pages to crawl."
    )
    is_verified = models.BooleanField(
        default=False,
        help_text="Indicates if the website has been verified."
    )
    source_type = models.CharField(
        max_length=50,
        blank=True,
        null=True,
        help_text='Source type, e.g., "manual", "automated".'
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="Timestamp when the website entry was created."
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        help_text="Timestamp when the website entry was last updated."
    )
    is_active = models.BooleanField(
        default=True,
        help_text="Indicates if the website is active."
    )
    is_deleted = models.BooleanField(
        default=False,
        help_text="Indicates if the website is deleted (soft delete)."
    )
    is_archived = models.BooleanField(
        default=False,
        help_text="Indicates if the website is archived."
    )
    meta_data = models.JSONField(
        blank=True,
        null=True,
        default=dict,
        help_text="Optional metadata for the website."
    )

    def __str__(self):
        # Return a meaningful string representation, fallback to ID if name is missing
        return self.website_name or f"Website {getattr(self, 'website_id', 'Unknown')}"

    def should_crawl(self):
        """
        Determines if the website should be crawled based on the last crawl time and frequency.
        Returns True if the website should be crawled now.
        """
        if not getattr(self, "last_crawled_at", None) or not getattr(self, "crawl_frequency", None):
            return True  # If never crawled or no frequency defined, crawl it

        frequency_map = {
            "daily": 1,
            "weekly": 7,
            "monthly": 30,
        }

        freq = str(self.crawl_frequency).strip().lower()
        days = frequency_map.get(freq)
        if days is None:
            try:
                days = int(self.crawl_frequency)
            except (ValueError, TypeError):
                return False  # Invalid frequency

        last_crawled = self.last_crawled_at
        if not isinstance(last_crawled, datetime):
            # Defensive: try to parse if it's a string
            try:
                last_crawled = datetime.fromisoformat(str(last_crawled))
            except Exception:
                return True  # If can't parse, err on the side of crawling

        next_crawl_due = last_crawled + timedelta(days=days)
        # Use timezone-aware comparison if possible
        now = timezone.now() if timezone.is_aware(last_crawled) else datetime.now()
        return now >= next_crawl_due

    def mark_crawled(self, status="success"):
        """
        Updates the last_crawled_at timestamp and crawl_status, then saves the model.
        """
        self.last_crawled_at = timezone.now()
        self.crawl_status = status
        self.save(update_fields=["last_crawled_at", "crawl_status"])

    def get_domain(self):
        """
        Returns the domain part of the website_url.
        """
        if not getattr(self, "website_url", None):
            return ""
        try:
            return urlparse(self.website_url).netloc
        except Exception:
            return ""

    def deactivate(self):
        """
        Sets the website as inactive and saves the model.
        """
        self.is_active = False
        self.save(update_fields=["is_active"])

    def soft_delete(self):
        """
        Marks the website as deleted (soft delete) and saves the model.
        """
        self.is_deleted = True
        self.save(update_fields=["is_deleted"])

    def to_dict(self, include_meta=False):
        """
        Returns a dictionary representation of the website.
        Optionally includes meta_data if include_meta is True.
        """
        data = {
            "id": getattr(self, "website_id", None),
            "name": getattr(self, "website_name", None),
            "url": getattr(self, "website_url", None),
            "status": getattr(self, "crawl_status", None),
            "last_crawled_at": getattr(self, "last_crawled_at", None),
            "is_active": getattr(self, "is_active", None),
        }
        if include_meta:
            data["meta_data"] = getattr(self, "meta_data", None)
        return data


