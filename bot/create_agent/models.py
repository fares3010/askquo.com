# Django imports
from django.db import models
from django.conf import settings
from django.utils import timezone
from django.core.files.storage import FileSystemStorage
from django.shortcuts import get_object_or_404

# Standard library imports
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Tuple, Any, Dict
from urllib.parse import urlparse
import hashlib
import json
import logging
import os
import jwt
import secrets
import uuid
import dotenv
from .content_parser.web_parsing import ContentParser
from .vdb.pinecone_vdb import PineconeVDB

# Langchain document loaders - grouped by file type
# Text-based formats
from langchain.document_loaders import (
    TextLoader,
    DirectoryLoader,
    JSONLoader,
)

# Office document formats
from langchain.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
)


# Load environment variables
dotenv.load_dotenv()

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('embedding_model.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)



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
    
    def get_agent_details(self):
        """
        Return a dictionary of agent details.
        """
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'description': self.description,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'is_archived': self.is_archived,
            'is_favorite': self.is_favorite,
            'visibility': self.visibility,
            'avatar_url': self.avatar_url,  
            'configuration': self.configuration
        }

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

    def create_vector_database_path(self):
        """
        Creates a vector database path for the agent.
        """
        vector_database_path = f"agents/{self.agent_id}/vector_database"

        AgentVectorsDatabase.objects.create(
            agent=self,
            database_path=vector_database_path,
            created_at=timezone.now(),
            updated_at=timezone.now()
        )
        return vector_database_path

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
    
    def no_of_documents(self):
        """
        Returns the number of documents associated with this agent.
        """
        return self.documents.count()
    
    def no_of_texts(self):
        """
        Returns the number of texts associated with this agent.
        """
        return self.texts.count()
    
    def no_of_integrations(self):   
        """
        Returns the number of integrations associated with this agent.
        """
        return self.integrations.count()
    
    def no_of_websites(self):   
        """
        Returns the number of websites associated with this agent.
        """
        return self.websites.count()
    
    def no_of_conversations(self):
        """
        Returns the number of conversations associated with this agent.
        """
        return self.agent_conversations.count()
    
    def no_of_team_members(self):
        """
        Returns the number of team members associated with this agent.
        """
        # Get teams that this agent belongs to through TeamAgent relationship
        team_count = 0
        active_team_agents = self.team_agents.filter(is_active=True, is_deleted=False)
        for team_agent in active_team_agents:
            team_count += team_agent.team.get_team_member_count()
        return team_count


# Create your models here.
class AgentDocuments(models.Model):
    """Model for storing and managing agent documents with various formats and metadata."""
    
    agent = models.ForeignKey(
        'Agent', 
        on_delete=models.CASCADE, 
        related_name='documents',
        help_text="The agent this document belongs to."
    )
    document_id = models.AutoField(
        primary_key=True,
        help_text="Unique identifier for the document."
    )
    document_name = models.CharField(
        max_length=255, 
        blank=True, 
        null=True,
        help_text="Name of the document."
    )
    document_description = models.TextField(
        blank=True, 
        null=True,
        help_text="Detailed description of the document's contents."
    )
    document_path = models.URLField(
        max_length=500, 
        blank=True, 
        null=True,
        help_text="URL path to the document if stored externally."
    )
    document_file = models.FileField(
        upload_to='documents/',
        storage=FileSystemStorage(location=settings.MEDIA_ROOT),
        blank=True, 
        null=True,
        help_text="The actual document file."
    )
    document_size = models.BigIntegerField(
        blank=True, 
        null=True,
        help_text="Size of the document in bytes."
    )
    document_format = models.CharField(
        max_length=50, 
        blank=True, 
        null=True,
        help_text="Format of the document (e.g., PDF, DOCX)."
    )
    document_language = models.CharField(
        max_length=50, 
        blank=True, 
        null=True,
        help_text="Language of the document content."
    )
    document_tags = models.JSONField(
        blank=True, 
        null=True, 
        default=list,
        help_text="List of tags associated with the document."
    )
    embedded = models.BooleanField(
        default=False,
        help_text="Whether the document has been embedded."
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="Timestamp when the document was created."
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        help_text="Timestamp when the document was last updated."
    )
    is_active = models.BooleanField(
        default=True,
        help_text="Whether the document is currently active."
    )
    is_deleted = models.BooleanField(
        default=False,
        help_text="Whether the document has been soft-deleted."
    )
    is_archived = models.BooleanField(
        default=False,
        help_text="Whether the document has been archived."
    )
    meta_data = models.JSONField(
        blank=True, 
        null=True, 
        default=dict,
        help_text="Additional metadata about the document."
    )

    class Meta:
        verbose_name = "Agent Document"
        verbose_name_plural = "Agent Documents"
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['agent', 'document_format']),
            models.Index(fields=['created_at']),
            models.Index(fields=['is_active', 'is_deleted']),
        ]

    def __str__(self) -> str:
        """Return a string representation of the document."""
        return self.document_name or f"Document {self.document_id}"

    def get_document_details(self) -> dict:
        """Get a dictionary containing all relevant document details.
        
        Returns:
            dict: Document details including ID, name, size, format, etc.
        """
        return {
            "document_id": self.document_id,
            "document_name": self.document_name,
            "document_description": self.document_description,
            "document_file": self.document_file,
            "document_size": self.formatted_size,
            "document_format": self.document_format,
            "document_language": self.document_language,
            "document_tags": self.document_tags if self.document_tags is not None else [],
            "embedded": self.embedded,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "is_expired": self.is_expired(),
            "is_active": self.is_active,
            "is_archived": self.is_archived,
        }

    @property
    def formatted_size(self) -> str:
        """Get a human-readable representation of the document size.
        
        Returns:
            str: Formatted size string (e.g., "1.5 MB")
        """
        if self.document_size is None:
            return "Unknown"
            
        size = float(self.document_size)
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:.2f} {unit}"
            size /= 1024.0
        return f"{size:.2f} PB"

    def is_expired(self) -> bool:
        """Check if the document has expired based on its creation date and metadata.
        
        Returns:
            bool: True if the document has expired, False otherwise.
        """
        expiration_days = 365
        if self.meta_data and isinstance(self.meta_data, dict):
            expiration_days = self.meta_data.get('expiration_days', 365)
        return self.created_at < timezone.now() - timezone.timedelta(days=expiration_days)

    def get_document_text(self) -> Optional[List[str]]:
        """Extract text content from the document using appropriate loaders.
        
        Returns:
            Optional[List[str]]: List of text strings extracted from the document if successful, 
            None if format not supported or if an error occurs.
        """
        format_loaders = {
            "application/pdf": PyPDFLoader,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": Docx2txtLoader,
            "text/plain": TextLoader,
            "text/csv": CSVLoader,
            "application/json": JSONLoader
        }
        
        loader_class = format_loaders.get(self.document_format)
        if not loader_class:
            logger.warning(f"Unsupported document format: {self.document_format}")
            return None
            
        try:
            loader = loader_class(self.document_file.path)
            documents = loader.load()
            # Extract just the text content from each Document object
            return [doc.page_content for doc in documents]
        except Exception as e:
            logger.error(f"Error loading document {self.document_id}: {str(e)}")
            return None

    
class AgentIntegrations(models.Model):
    """
    Model for managing agent integrations with external services.
    
    This model handles various types of integrations including OAuth, API key,
    and other authentication methods. It provides comprehensive tracking of
    authentication status, tokens, and integration metadata.
    """
    
    # Core relationships
    agent = models.ForeignKey(
        Agent,
        on_delete=models.CASCADE,
        related_name='agent_integrations',
        help_text="The agent this integration belongs to."
    )
    
    # Primary identifier
    integration_id = models.AutoField(
        primary_key=True,
        help_text="Primary key for the integration."
    )
    
    # Basic integration information
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
    integration_description = models.TextField(
        blank=True,
        null=True,
        help_text="Description of the integration."
    )
    
    # UI/Display fields
    integration_logo_url = models.URLField(
        max_length=200,
        blank=True,
        null=True,
        help_text="URL for the integration's logo (for frontend display)."
    )
    integration_url = models.URLField(
        max_length=200,
        blank=True,
        null=True,
        help_text="URL to the integration's main page or API."
    )
    
    # API credentials (consider encryption in production)
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
    
    # Content and configuration
    integration_content = models.TextField(
        blank=True,
        null=True,
        help_text="Content of the integration."
    )
    configuration = models.JSONField(
        blank=True,
        null=True,
        help_text="Additional configuration for the integration."
    )
    meta_data = models.JSONField(
        blank=True,
        null=True,
        help_text="Additional metadata for the integration."
    )
    
    # Authentication fields
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
    integration_auth_scope = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        help_text="OAuth scopes or permissions for the integration."
    )
    
    # Authentication status and tracking
    integration_auth_status = models.CharField(
        max_length=50,
        blank=True,
        null=True,
        help_text="Status of the authentication (e.g., Valid, Expired, Error)."
    )
    integration_auth_expiry = models.DateTimeField(
        blank=True,
        null=True,
        help_text="When the current auth token expires."
    )
    integration_auth_error = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        help_text="Error message from the last authentication attempt."
    )
    integration_auth_error_time = models.DateTimeField(
        blank=True,
        null=True,
        help_text="Timestamp of the last authentication error."
    )
    integration_auth_response = models.JSONField(
        blank=True,
        null=True,
        help_text="Raw response from the authentication endpoint."
    )
    
    # Status and lifecycle fields
    embedded = models.BooleanField(
        default=False,
        help_text="Whether the integration has been embedded."
    )
    is_active = models.BooleanField(
        default=True,
        help_text="Whether the integration is currently active."
    )
    is_deleted = models.BooleanField(
        default=False,
        help_text="Whether the integration is deleted (soft delete)."
    )
    
    # Timestamps
    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="Timestamp when the integration was created."
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        help_text="Timestamp when the integration was last updated."
    )

    class Meta:
        verbose_name = "Agent Integration"
        verbose_name_plural = "Agent Integrations"
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['agent', 'is_active']),
            models.Index(fields=['integration_auth_status']),
            models.Index(fields=['integration_category']),
        ]

    def __str__(self):
        """Return a meaningful string representation for the integration."""
        if self.integration_name:
            return f"{self.integration_name} (ID: {self.integration_id})"
        return f"Integration {self.integration_id}"

    def is_token_expired(self) -> bool:
        """
        Check if the integration's auth token is expired or missing expiry.
        
        Returns:
            bool: True if token is expired or missing expiry, False otherwise.
        """
        if not self.integration_auth_expiry:
            return True
        return self.integration_auth_expiry <= timezone.now()

    def get_status(self) -> dict:
        """
        Get a dictionary summarizing the integration's status.
        
        Returns:
            dict: Status information including name, active state, auth status,
                  token expiry, last updated, and auth error status.
        """
        return {
            "name": self.integration_name,
            "active": self.is_active,
            "auth_status": self.integration_auth_status,
            "token_expired": self.is_token_expired(),
            "last_updated": self.updated_at,
            "has_auth_error": self.has_auth_error(),
            "category": self.integration_category,
            "priority": self.integration_priority,
        }

    def update_auth_token(self, token: str, expiry: Optional[datetime] = None) -> None:
        """
        Update the integration's auth token and expiry, and mark status as valid.
        
        Args:
            token: The new authentication token
            expiry: Optional expiry datetime for the token
        """
        self.integration_auth_token = token
        if expiry is not None:
            self.integration_auth_expiry = expiry
        self.integration_auth_status = "Valid"
        self.integration_auth_error = None  # Clear any previous errors
        self.integration_auth_error_time = None
        self.save(update_fields=[
            "integration_auth_token", 
            "integration_auth_expiry", 
            "integration_auth_status", 
            "integration_auth_error",
            "integration_auth_error_time",
            "updated_at"
        ])

    def soft_delete(self) -> None:
        """
        Soft delete the integration by marking it as inactive and deleted.
        """
        self.is_deleted = True
        self.is_active = False
        self.save(update_fields=["is_deleted", "is_active", "updated_at"])

    def get_public_details(self) -> dict:
        """
        Get a dictionary of public-facing integration details.
        
        Returns:
            dict: Public integration information without sensitive data.
        """
        return {
            "integration_name": self.integration_name,
            "integration_description": self.integration_description,
            "integration_url": self.integration_url,
            "integration_category": self.integration_category,
            "integration_priority": self.integration_priority,
            "auth_type": self.integration_auth_type,
            "status": self.integration_auth_status,
            "is_active": self.is_active,
            "token_masked": bool(self.integration_auth_token),
            "embedded": self.embedded,
            "created_at": self.created_at,
        }

    def has_auth_error(self) -> bool:
        """
        Check if there is an authentication error message.
        
        Returns:
            bool: True if there is an authentication error, False otherwise.
        """
        return bool(self.integration_auth_error)

    def clear_auth_error(self) -> None:
        """
        Clear any authentication error and reset error timestamp.
        """
        self.integration_auth_error = None
        self.integration_auth_error_time = None

class AgentIntegrationsFrontend(models.Model):
    """
    Model for managing agent integrations with frontend websites.
    """

    agent = models.ForeignKey(
        Agent,
        on_delete=models.CASCADE,
        related_name='integrations_frontend',
        help_text="The agent this integration belongs to."
    )
    integration_id = models.AutoField(
        primary_key=True,
        help_text="Primary key for the integration."
    )
    api_key = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        unique=True,
        help_text="API key for the integration."
    )
    integration_name = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        help_text="Name of the integration."
    )
    integration_description = models.TextField(
        blank=True,
        null=True,
        help_text="Description of the integration."
    )
    website_name = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        help_text="Name of the website."
    )
    website_domain = models.CharField(
        max_length=255,
        help_text="Domain of the website."
    )
    website_logo = models.ImageField(
        upload_to='integrations_frontend/logos/',
        blank=True,
        null=True,
        help_text="Logo of the website."
    )
    integration_auth_type = models.CharField(
        max_length=255,
        default='JWT',
        blank=True,
        null=True,
        help_text="Type of authentication for the integration."
    )
    start_message = models.TextField(
        blank=True,
        null=True,
        help_text="Start message for the integration."
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
        help_text="Indicates if the integration is active."
    )
    is_deleted = models.BooleanField(
        default=False,
        help_text="Indicates if the integration is deleted."
    )
    meta_data = models.JSONField(
        blank=True,
        null=True,
        default=dict,
        help_text="Optional metadata for the integration."
    )

    
    def __str__(self) -> str:
        return f"{self.integration_name} - {self.website_name}"
    
    def get_status(self) -> dict:
        """
        Get a dictionary summarizing the integration's status.
        """
        return {
            "integration_name": self.integration_name,
            "integration_description": self.integration_description,
            "website_name": self.website_name,
            "website_domain": self.website_domain,
            "website_logo": self.website_logo,
            "integration_auth_type": self.integration_auth_type,
        }
    def get_number_of_clients(self) -> int:
        """
        Get the number of clients for the integration.
        """
        return self.clients.count()

class FrontendIntegrationClient(models.Model):
    
    integration = models.ForeignKey(
        AgentIntegrationsFrontend,
        on_delete=models.CASCADE,
        related_name='clients',
        help_text="The integration this client belongs to."
    )
    frontend_client_id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        help_text="Primary key for the client."
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="Timestamp when the client was created."
    )
    is_deleted = models.BooleanField(
        default=False,
        help_text="Indicates if the client is deleted."
    )
    is_active = models.BooleanField(
        default=True,
        help_text="Indicates if the client is active."
    )

    def __str__(self) -> str:
        return f"{self.frontend_client_id} - {self.integration.integration_name}"

    def generate_access_token(self) -> str:
        """
        Create a JWT token for the client.
        """
        payload = {
            "frontend_client_id": self.frontend_client_id,
            "type": "access",
            "exp": datetime.now() + timedelta(hours=1),
            "iat": datetime.now()
        }
        return jwt.encode(payload,os.getenv("CLIENT_SECRET_KEY"), algorithm="HS256")
    
    def generate_refresh_token(self) -> str:
    
        """
        Create a JWT token for the client.
        """
        payload = {
            "frontend_client_id": self.frontend_client_id,
            "type": "refresh",
            "exp": datetime.now() + timedelta(days=1),
            "iat": datetime.now()
        }
        return jwt.encode(payload,os.getenv("CLIENT_SECRET_KEY"), algorithm="HS256")
    
    def generate_jwt_tokens(self) -> dict:
        """
        Generate a pair of JWT tokens for the client.
        """
        return {
            "access": self.generate_access_token(),
            "refresh": self.generate_refresh_token()
        }
    
    def decode_jwt_token(self, token: str) -> dict:
        """
        Decode a JWT token for the client.
        """
        try:
            return jwt.decode(token, os.getenv("CLIENT_SECRET_KEY"), algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            return {"error": "Token has expired."}
        except jwt.InvalidTokenError:
            return {"error": "Invalid token."}
        except Exception as e:
            return {"error": str(e)}

    

    
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
    qa_content = models.TextField(
        blank=True,
        null=True,
        default=dict,
        help_text="QA pair content."
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
    text_title = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        help_text="Optional name or label for the text."
    )
    text_content = models.TextField(
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
    embedded = models.BooleanField(
        default=False,
        help_text="Whether the text has been embedded."
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
        if self.text_title:
            return f"{self.text_title} ({self.text_content[:30]}...)" if self.text_content and len(self.text_content) > 30 else self.text_title
        return (self.text_content[:50] + "...") if self.text_content and len(self.text_content) > 50 else (self.text_content or "")
    
    def get_text_details(self):
        """
        Return a dictionary of text details.
        """
        return {
            'text_id': self.text_id,
            'text_title': self.text_title,    
            'text_content': self.text_content,
            'text_language': self.text_language,
            'text_type': self.text_type,
            'embedded': self.embedded,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'is_active': self.is_active,    
            'is_deleted': self.is_deleted,
            'is_archived': self.is_archived,
            'meta_data': self.meta_data
        }

    def short_text(self, length=30):
        """
        Return a shortened version of the text, appending '...' if truncated.
        """
        if not self.text_content:
            return ""
        return self.text_content[:length] + "..." if len(self.text_content) > length else self.text_content

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
    vdb_name = models.CharField(
        default="pinecone",
        max_length=255,
        help_text="Name of the vector database where vectors are stored (e.g., FAISS, Pinecone, Chroma, Weaviate)."
    )
    database_path = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        help_text="Path to the vector database."
    )
    database_index = models.CharField(
        default=os.getenv("DATABASE_INDEX"),
        max_length=255,
        blank=True,
        null=True,
        help_text="Index name for the vector database."
    )
    namespace = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        help_text="Namespace for the index in the vector database."
    )
    vector_count = models.PositiveIntegerField(
        default=0,
        help_text="Number of vectors currently stored in the database."
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

    meta_data = models.JSONField(
        blank=True,
        null=True,
        default=dict,
        help_text="Optional metadata for the vector database."
    )

    def __str__(self):
        agent_name = getattr(self.agent, "name", None) or "Unknown Agent"
        db_name = getattr(self, "vdb_name", None)
        db_name_display = db_name[:40] + "..." if db_name and len(db_name) > 40 else (db_name or "No Name")
        return f"{agent_name} VDB: {db_name_display}"
    

    def is_empty(self):
        return (self.vector_count or 0) == 0
            

    def get_index_summary(self):
        return {
            "vector_count": self.vector_count,
            "vdb_name": self.vdb_name,
            "indexed_at": self.last_indexed_at,
        }

    def mark_indexed(self):
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
        backend = (self.vdb_name or "").strip().lower()
        return backend in supported

    def display_name(self):
        agent_name = getattr(self.agent, "name", None) or "Unknown Agent"
        model = self.vdb_name or "Unknown Model"
        return f"{agent_name} - {model}"

    def needs_reindexing(self, threshold_days=7):
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
        default="openai-ada",
        max_length=100,
        blank=True,
        null=True,
        help_text="Name of the embedding model (e.g., openai-ada, sentence-transformers, etc.)."
    )
    embedding_model_version = models.CharField(
        default="text-embedding-3-small",
        max_length=100,
        blank=True,
        null=True,
        help_text="Version of the embedding model (e.g., text-embedding-3-small, all-MiniLM-L6-v2, etc.)."
    )
    vector_dimension = models.IntegerField(
        default=1536,
        blank=True,
        null=True,
        help_text="Dimension of the embedding vector."
    )
    similarity_score = models.FloatField(
        default=0.0,
        blank=True,
        null=True,
        help_text="Similarity score with respect to a query vector."
    )
    num_chunks = models.IntegerField(
        default=0,
        blank=True,
        null=True,
        help_text="Number of chunks in the embedding."
    )
    source_url = models.URLField(
        default="",
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
    object_id = models.IntegerField(
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
        help_text="Optional metadata for the embedding."
    )

    def __str__(self):
        return f"{self.object_type} - {self.object_name}"


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
    def delete_embedding(self):
        """
        Delete the embedding from the vector database.
        """
        vdb = get_object_or_404(AgentVectorsDatabase, agent=self.agent)
        pinecone_vdb = PineconeVDB(
            index_name=vdb.database_index,
            namespace=vdb.namespace,
        )
        for i in range(self.num_chunks):
            pinecone_vdb.delete_vectors(ids=[f"{self.embedding_id}-{i}"], namespace=vdb.namespace)
        self.delete()
        

class AgentWebsites(models.Model):
    """
    Model for storing website information associated with agents.
    Handles website crawling, content management, and metadata storage.
    """
    
    # Choices for better data consistency
    CRAWL_STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('success', 'Success'),
        ('failed', 'Failed'),
        ('partial', 'Partial'),
        ('crawling', 'Crawling'),
    ]
    
    WEBSITE_TYPE_CHOICES = [
        ('blog', 'Blog'),
        ('e-commerce', 'E-commerce'),
        ('portfolio', 'Portfolio'),
        ('news', 'News'),
        ('documentation', 'Documentation'),
        ('social', 'Social Media'),
        ('other', 'Other'),
    ]
    
    SOURCE_TYPE_CHOICES = [
        ('manual', 'Manual'),
        ('automated', 'Automated'),
        ('imported', 'Imported'),
    ]
    
    # Core fields
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
        choices=WEBSITE_TYPE_CHOICES,
        blank=True,
        null=True,
        help_text='Type of website, e.g., "blog", "e-commerce", "portfolio".'
    )
    website_content = models.TextField(
        blank=True,
        null=True,
        help_text="Content of the website."
    )
    
    # Crawling related fields
    crawl_status = models.CharField(
        max_length=50,
        choices=CRAWL_STATUS_CHOICES,
        default='pending',
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
    page_limit = models.PositiveIntegerField(
        blank=True,
        null=True,
        help_text="Maximum number of pages to crawl."
    )
    
    # Content and language fields
    content_language = models.CharField(
        max_length=50,
        blank=True,
        null=True,
        help_text="Language of the website content, e.g., 'English', 'Spanish'."
    )
    
    # Status fields
    is_verified = models.BooleanField(
        default=False,
        help_text="Indicates if the website has been verified."
    )
    embedded = models.BooleanField(
        default=False,
        help_text="Whether the website has been embedded."
    )
    source_type = models.CharField(
        max_length=50,
        choices=SOURCE_TYPE_CHOICES,
        default='manual',
        help_text='Source type, e.g., "manual", "automated".'
    )
    
    # Timestamp fields
    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="Timestamp when the website entry was created."
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        help_text="Timestamp when the website entry was last updated."
    )
    
    # State management fields
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
    
    # Metadata field
    meta_data = models.JSONField(
        blank=True,
        null=True,
        default=dict,
        help_text="Optional metadata for the website."
    )

    class Meta:
        db_table = 'agent_websites'
        verbose_name = 'Agent Website'
        verbose_name_plural = 'Agent Websites'
        indexes = [
            models.Index(fields=['agent', 'is_deleted']),
            models.Index(fields=['crawl_status', 'is_active']),
            models.Index(fields=['last_crawled_at']),
            models.Index(fields=['website_url']),
        ]
        ordering = ['-created_at']

    def __str__(self):
        """Return a meaningful string representation."""
        return self.website_name or f"Website {self.website_id}"
    
    def get_website_content(self):
        """
        Returns the content of the website by parsing the URL.
        Returns a structured dictionary with parsed content from the home page and internal pages.
        """
        try:
            # Initialize parser with better error handling
            parser = ContentParser()
            result = {
                "info": "here is the scraped and parsed content of the website",
                'website_url': self.website_url,
                'website_name': self.website_name,
                'home_page': None,
                'internal_pages': [],
                'total_pages_crawled': 0,
                'crawl_status': 'success',
                'errors': []
            }
            
            # Parse home page
            html = parser.fetch_content(self.website_url)
            if not html:
                error_msg = f"Failed to fetch content from {self.website_url}"
                result['errors'].append(error_msg)
                result['crawl_status'] = 'failed'
                logger.error(error_msg)
                return self._create_error_response(result, error_msg)
            
            home_page_content = parser.parse_content(html, self.website_url)
            if not home_page_content:
                error_msg = f"Failed to parse content from home page {self.website_url}"
                result['errors'].append(error_msg)
                result['crawl_status'] = 'failed'
                logger.error(error_msg)
                return self._create_error_response(result, error_msg)
            
            # Add home page to result
            result['home_page'] = {
                'url': self.website_url,
                "parsed_content": home_page_content.to_dict()
            }
            result['total_pages_crawled'] += 1
            
            # Crawl internal pages with improved logic
            self._crawl_internal_pages(parser, home_page_content, result)
            
            # Set crawl status based on results
            if result['total_pages_crawled'] == 0:
                result['crawl_status'] = 'failed'
            elif result['errors']:
                result['crawl_status'] = 'partial'

            content = json.dumps(result, indent=2, ensure_ascii=False)
            logger.info(f"Website content saved for {self.website_url} with total pages crawled: {result['total_pages_crawled']}, status: {result['crawl_status']}")
            return content
            
        except Exception as e:
            error_msg = f"Failed to get content for website {self.website_url}: {str(e)}"
            logger.error(error_msg)
            return self._create_error_response({
                'website_url': self.website_url,
                'website_name': self.website_name,
                'home_page': None,
                'internal_pages': [],
                'total_pages_crawled': 0,
                'crawl_status': 'failed',
                'errors': []
            }, error_msg)
    
    def _crawl_internal_pages(self, parser, home_page_content, result):
        """
        Helper method to crawl internal pages with improved logic and error handling.
        """
        page_limit = self.page_limit if self.page_limit else 10
        crawled_urls = set()  # Track crawled URLs to avoid duplicates
        
        # Get internal links from home page
        internal_links = [link for link in home_page_content.links if not link.get('is_external')]
        
        # Crawl first level internal pages
        for link in internal_links[:page_limit]:
            if link['url'] in crawled_urls:
                continue
                
            try:
                self._crawl_single_page(parser, link['url'], result, crawled_urls)
            except Exception as e:
                error_msg = f"Failed to crawl internal page {link['url']}: {str(e)}"
                result['errors'].append(error_msg)
                logger.warning(error_msg)
        
        # Crawl second level pages if we haven't reached the limit
        if result['total_pages_crawled'] < page_limit:
            for page_data in result['internal_pages']:
                if result['total_pages_crawled'] >= page_limit:
                    break
                    
                page_content = page_data['parsed_content']
                second_level_links = [link for link in page_content['links'] if not link.get('is_external')]
                
                for link in second_level_links:
                    if result['total_pages_crawled'] >= page_limit or link['url'] in crawled_urls:
                        continue
                        
                    try:
                        self._crawl_single_page(parser, link['url'], result, crawled_urls)
                    except Exception as e:
                        error_msg = f"Failed to crawl second level page {link['url']}: {str(e)}"
                        result['errors'].append(error_msg)
                        logger.warning(error_msg)
    
    def _crawl_single_page(self, parser, url, result, crawled_urls):
        """
        Helper method to crawl a single page and add it to results.
        """
        page_html = parser.fetch_content(url)
        if not page_html:
            error_msg = f"Failed to fetch content from internal page {url}"
            result['errors'].append(error_msg)
            logger.warning(error_msg)
            return
        
        page_content = parser.parse_content(page_html, url)
        if not page_content:
            error_msg = f"Failed to parse content from internal page {url}"
            result['errors'].append(error_msg)
            logger.warning(error_msg)
            return
        
        page_data = {
            'url': url,
            "parsed_content": page_content.to_dict()
        }
        result['internal_pages'].append(page_data)
        result['total_pages_crawled'] += 1
        crawled_urls.add(url)
    
    def _create_error_response(self, result, error_msg):
        """
        Helper method to create standardized error response.
        """
        result['errors'].append(error_msg)
        return json.dumps(result, indent=2, ensure_ascii=False)
    def get_website_details(self):
        """
        Return a dictionary of website details.
        """
        return {
            'website_id': self.website_id,
            'website_url': self.website_url,
            'website_name': self.website_name,
            'website_type': self.website_type,
            'website_content': self.website_content,
            'crawl_status': self.crawl_status,
            'last_crawled_at': self.last_crawled_at,
            'crawl_frequency': self.crawl_frequency,
            'content_language': self.content_language,
            'page_limit': self.page_limit,
            'is_verified': self.is_verified,
            'embedded': self.embedded,
            'source_type': self.source_type,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'is_active': self.is_active,
            'is_deleted': self.is_deleted,
            'is_archived': self.is_archived,
            'meta_data': self.meta_data
        }

    def should_crawl(self):
        """
        Determines if the website should be crawled based on the last crawl time and frequency.
        Returns True if the website should be crawled now.
        """
        # If never crawled or no frequency defined, crawl it
        if not self.last_crawled_at or not self.crawl_frequency:
            return True

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

        next_crawl_due = self.last_crawled_at + timedelta(days=days)
        now = timezone.now()
        return now >= next_crawl_due

    def mark_crawled(self, status="success"):
        """
        Updates the last_crawled_at timestamp and crawl_status, then saves the model.
        """
        self.last_crawled_at = timezone.now()
        self.crawl_status = status
        self.save(update_fields=["last_crawled_at", "crawl_status", "updated_at"])

    def get_domain(self):
        """
        Returns the domain part of the website_url.
        """
        if not self.website_url:
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
        self.save(update_fields=["is_active", "updated_at"])

    def soft_delete(self):
        """
        Marks the website as deleted (soft delete) and saves the model.
        """
        self.is_deleted = True
        self.save(update_fields=["is_deleted", "updated_at"])

    def archive(self):
        """
        Archives the website and saves the model.
        """
        self.is_archived = True
        self.save(update_fields=["is_archived", "updated_at"])

    def to_dict(self, include_meta=False):
        """
        Returns a dictionary representation of the website.
        Optionally includes meta_data if include_meta is True.
        """
        data = {
            "id": self.website_id,
            "name": self.website_name,
            "url": self.website_url,
            "type": self.website_type,
            "status": self.crawl_status,
            "last_crawled_at": self.last_crawled_at,
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "embedded": self.embedded,
        }
        if include_meta:
            data["meta_data"] = self.meta_data
        return data

    @property
    def is_crawlable(self):
        """
        Returns True if the website can be crawled (active, not deleted, not archived).
        """
        return self.is_active and not self.is_deleted and not self.is_archived

    @property
    def needs_crawling(self):
        """
        Returns True if the website needs to be crawled.
        """
        return self.is_crawlable and self.should_crawl()

    def clean(self):
        """
        Custom validation for the model.
        """
        from django.core.exceptions import ValidationError
        
        if self.website_url:
            try:
                # Validate URL format
                parsed_url = urlparse(self.website_url)
                if not all([parsed_url.scheme, parsed_url.netloc]):
                    raise ValidationError({
                        'website_url': "Invalid URL format. Must include scheme (e.g., http:// or https://)."
                    })
                # Validate domain
                domain = parsed_url.netloc
                if not domain:
                    raise ValidationError({
                        'website_url': "Invalid URL. Must include a domain."
                    })
            except ValidationError as e:
                raise ValidationError({
                    'website_url': e.message
                })
            except Exception as e:
                raise ValidationError({
                    'website_url': "An unexpected error occurred while validating the URL."
                })
        if self.crawl_frequency:
            try:
                # Validate frequency
                frequency_map = {
                    "daily": 1,
                    "weekly": 7,
                    "monthly": 30,
                    "yearly": 365
                }
                if self.crawl_frequency not in frequency_map:
                    raise ValidationError({
                        'crawl_frequency': "Invalid frequency. Must be one of: daily, weekly, monthly, yearly."
                    })
            except ValidationError as e:
                raise ValidationError({
                    'crawl_frequency': e.message
                })
            except Exception as e:
                raise ValidationError({
                    'crawl_frequency': "An unexpected error occurred while validating the frequency."
                })
        if self.page_limit:
            try:
                # Validate page limit
                if self.page_limit <= 0:    
                    raise ValidationError({
                        'page_limit': "Page limit must be greater than 0."
                    })
            except ValidationError as e:
                raise ValidationError({
                    'page_limit': e.message
                })
            except Exception as e:
                raise ValidationError({
                    'page_limit': "An unexpected error occurred while validating the page limit."
                })
