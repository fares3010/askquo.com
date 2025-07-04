from datetime import date
from django.shortcuts import render, get_object_or_404
from rest_framework.response import Response # type: ignore
from rest_framework.decorators import api_view, permission_classes
from rest_framework import status # type: ignore
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.tokens import RefreshToken, AccessToken # type: ignore
from django.core.cache import cache
from django.core.exceptions import ValidationError
from django.db.models import Q, Max, Subquery, OuterRef
from django.db import transaction, IntegrityError
from django.utils import timezone
from .models import Conversation, ConversationMessages
from django.http import JsonResponse, Http404
import math
import jwt
import os
import dotenv
import logging
from create_agent.models import Agent, AgentIntegrationsFrontend, FrontendIntegrationClient
from conversations.gpt.openai_chat import initialize_chat_agent
from plans.models import UserSubscription
from accounts.models import TeamMember, Team

dotenv.load_dotenv()

logger = logging.getLogger(__name__)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def conversations(request, agent_id):
    """Retrieve paginated and filtered conversations for authenticated user."""
    try:
        agent = get_object_or_404(Agent,user=request.user, agent_id=agent_id)
        # Get and validate pagination parameters
        page = max(1, int(request.GET.get("page", 1)))
        limit = min(100, max(1, int(request.GET.get("limit", 10))))

        # Get filtered queryset with annotation for last message time
        qs = Conversation.objects.filter(
            agent=agent, 
            is_archived=False, 
            is_deleted=False
        ).annotate(
            last_msg_time=Max('messages__message_time')
        )
        
        # Calculate pagination
        total = qs.count()
        if total == 0:
            return Response(
                {"error": "No conversations found."},
                status=status.HTTP_404_NOT_FOUND
            )
            
        no_of_pages = math.ceil(total / limit)
        if page > no_of_pages:
            return Response(
                {"error": "Page number exceeds total pages."},
                status=status.HTTP_400_BAD_REQUEST
            )
            
        # Get paginated conversations ordered by last message time
        offset = (page - 1) * limit
        conversations = qs.order_by('-last_msg_time')[offset:offset + limit]
        
        # Prepare response data
        data = [{
            "id": conv.conversation_id,
            "name": conv.conversation_name,
            "last_message": conv.last_message_text(),
            "timestamp": conv.last_message_time(),
            "unread": conv.check_last_message_is_read(),
            "status": conv.check_is_active(),
            "agent_name": agent.name,
        } for conv in conversations]
        response_data = {
            "data": data,
            "total": total,
            "page": page,
            "limit": limit,
            "total_pages": no_of_pages,
            "is_loading": False,
            "error": None,
            "refetch": True,
        }
        
        return Response(response_data, status=status.HTTP_200_OK)
        
    except ValueError as e:
        logger.error(f"Invalid pagination parameters: {str(e)}")
        return Response(
            {"error": str(e)},
            status=status.HTTP_400_BAD_REQUEST
        )
    except Exception as e:
        logger.error(f"Error retrieving conversations: {str(e)}")
        return Response(
            {"error": "An unexpected error occurred."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def conversation_messages(request, agent_id, conversation_id):
    """Retrieve messages for a specific conversation."""
    try:
        # Validate required parameters
        if not agent_id or not conversation_id:
            return Response(
                {"error": "Agent ID and conversation ID are required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Validate pagination parameters
        try:
            page = max(1, int(request.GET.get("page", 1)))
            limit = min(100, max(1, int(request.GET.get("limit", 30))))
        except ValueError:
            return Response(
                {"error": "Invalid pagination parameters."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Validate agent ownership
        agent = get_object_or_404(Agent, agent_id=agent_id, is_deleted=False)

        # Try to get conversation from cache first
        cache_key = f"conversation_{conversation_id}_{agent_id}"
        conversation = cache.get(cache_key)
        
        if not conversation:
            conversation = get_object_or_404(
                Conversation, 
                conversation_id=conversation_id,
                agent=agent,
                is_deleted=False
            )
            cache.set(cache_key, conversation, timeout=300)  # Cache for 5 minutes

        # Get messages with pagination
        messages_qs = conversation.messages.filter(is_deleted=False).order_by('-message_time')
        total = messages_qs.count()
        
        if total == 0:
            return Response({
                "data": [],
                "is_loading": False,
                "error": None,
                "refetch": True,
                "total": 0,
                "page": page,
                "limit": limit,
                "total_pages": 0,
            }, status=status.HTTP_200_OK)

        # Calculate pagination
        offset = (page - 1) * limit
        total_pages = math.ceil(total / limit)
        
        if page > total_pages:
            return Response(
                {"error": "Page number exceeds total pages."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Get paginated messages
        messages = messages_qs[offset:offset + limit]

        # Prepare response data
        data = [{
            "message_id": msg.message_id,
            "message_text": msg.message_text,
            "sender_type": msg.sender_type,
            "timestamp": msg.message_time,
            "is_read": msg.is_read,
            "is_deleted": msg.is_deleted,
            "is_archived": msg.is_archived,
            "attachments": [
                {
                    "attachment_id": att.attachment_id,
                    "attachment_name": att.attachment_name,
                    "attachment_path": att.attachment_path,
                    "attachment_type": att.attachment_type,
                    "attachment_size": att.attachment_size,
                }
                for att in msg.attachments.all()
            ] if hasattr(msg, "attachments") else [],
        } for msg in messages]

        return Response({
            "data": data,
            "is_loading": False,
            "error": None,
            "refetch": True,
            "total": total,
            "page": page,
            "limit": limit,
            "total_pages": total_pages,
        }, status=status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"Error retrieving messages: {str(e)}")
        return Response(
            {"error": "An unexpected error occurred."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(["GET"])
def index(request):
    """Health check endpoint for the Conversations API."""
    return JsonResponse({
        'message': 'Conversations API',
        'status': 'healthy',
        'version': '1.0.0'
    })

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def create_conversation(request):
    """Create a new conversation with improved error handling and validation."""
    try:
        data = request.data
        
        # Validate required fields
        if not data:
            return Response(
                {"error": "Request data is required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        conversation_name = data.get("name", "").strip()
        if not conversation_name:
            return Response(
                {"error": "Conversation name is required and cannot be empty."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        agent_id = data.get("agent_id")
        if not agent_id:
            return Response(
                {"error": "Agent ID is required."},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Validate user subscription with better error handling
        try:
            user_subscription = UserSubscription.objects.get(
                user=request.user, 
                is_active=True, 
                is_deleted=False
            )
        except UserSubscription.DoesNotExist:
            return Response(
                {"error": "No active subscription found. Please subscribe to continue."},
                status=status.HTTP_403_FORBIDDEN
            )
        
        if not user_subscription.is_valid_subscription():
            return Response(
                {"error": "You have reached your subscription limit. Please upgrade your subscription to continue."},
                status=status.HTTP_403_FORBIDDEN
            )
        
        # Check conversation usage limit
        try:
            conversation_limit = user_subscription.price.plan.feature_limit("conversations")
            if user_subscription.conversations_usage >= conversation_limit:
                return Response(
                    {"error": f"You have reached your conversation limit ({conversation_limit}). Please upgrade your subscription to continue."},
                    status=status.HTTP_403_FORBIDDEN
                )
        except AttributeError:
            logger.warning(f"Could not determine conversation limit for user {request.user.id}")
            # Continue without limit check if plan structure is unexpected

        
        agent = get_object_or_404(
            Agent,
            agent_id=agent_id,
            is_deleted=False
        )

        # Create conversation with transaction to ensure data consistency
        try:
            with transaction.atomic():
                conversation = Conversation.objects.create(
                    conversation_name=conversation_name,
                    agent=agent,
                    is_active=True,
                    is_archived=False,
                    is_deleted=False,
                    is_favorite=False,
                )
                
                # Create welcome message
                start_message = ConversationMessages.objects.create(
                    conversation=conversation,
                    message_text="Hello, how can I help you?",
                    sender_type="ai",
                    is_read=True,
                    message_time=timezone.now()
                )
                
                # Update conversation usage count
                user_subscription.conversations_usage += 1
                user_subscription.save(update_fields=['conversations_usage'])
        
        except Exception as e:
            logger.error(f"Error creating conversation in transaction: {str(e)}")
            return Response(
                {"error": "Failed to create conversation. Please try again."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        logger.info(f"Conversation created successfully: {conversation.conversation_id} by user {request.user.id}")
        
        return Response({
            "message": "Conversation created successfully.",
            "conversation_id": conversation.conversation_id,
            "conversation_name": conversation.conversation_name,
            "is_active": conversation.is_active,
            "is_archived": conversation.is_archived,
            "is_deleted": conversation.is_deleted,
            "is_favorite": conversation.is_favorite,
            "ai_message": {
                "message_id": start_message.message_id,
                "message_text": start_message.message_text,
                "sender_type": start_message.sender_type,
                "timestamp": start_message.message_time,
                "is_read": start_message.is_read
            }
        }, status=status.HTTP_201_CREATED)
        
    except ValidationError as e:
        logger.warning(f"Validation error creating conversation: {str(e)}")
        return Response(
            {"error": "Invalid data provided.", "details": str(e)},
            status=status.HTTP_400_BAD_REQUEST
        )
    except IntegrityError as e:
        logger.error(f"Database integrity error creating conversation: {str(e)}")
        return Response(
            {"error": "Database constraint violation."},
            status=status.HTTP_400_BAD_REQUEST
        )
    except Exception as e:
        logger.error(f"Unexpected error creating conversation: {str(e)}", exc_info=True)
        return Response(
            {"error": "An unexpected error occurred while creating the conversation."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def update_conversation(request, agent_id, conversation_id):
    """Update a conversation."""
    try:
        # Validate agent ownership
        agent = get_object_or_404(Agent, user=request.user, agent_id=agent_id, is_deleted=False)
        
        # Get conversation and validate ownership
        conversation = get_object_or_404(Conversation, conversation_id=conversation_id, agent=agent, is_deleted=False)
        
        data = request.data
        
        # Update conversation fields with validation
        if "name" in data:
            conversation.conversation_name = data["name"]
        if "is_archived" in data:
            conversation.is_archived = data["is_archived"]
        if "is_deleted" in data:
            conversation.is_deleted = data["is_deleted"]
        if "is_active" in data:
            conversation.is_active = data["is_active"]
        if "is_favorite" in data:
            conversation.is_favorite = data["is_favorite"]
            
        conversation.save()
        
        logger.info(f"Conversation {conversation_id} updated successfully by user {request.user.id}")
        
        return Response({
            "message": "Conversation updated successfully.",
            "conversation_id": conversation.conversation_id,
            "conversation_name": conversation.conversation_name
        }, status=status.HTTP_200_OK)
        
    except ValidationError as e:
        logger.warning(f"Validation error updating conversation: {str(e)}")
        return Response({
            "error": "Invalid data provided.",
            "details": str(e)
        }, status=status.HTTP_400_BAD_REQUEST)
        
    except Exception as e:
        logger.error(f"Error updating conversation {conversation_id}: {str(e)}", exc_info=True)
        return Response({
            "error": "An unexpected error occurred while updating the conversation."
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def update_message(request, agent_id, conversation_id, message_id):
    """Update a message."""
    try:
        # Validate agent ownership
        agent = get_object_or_404(Agent, user=request.user, agent_id=agent_id, is_deleted=False)
        
        # Get conversation and validate ownership
        conversation = get_object_or_404(Conversation, conversation_id=conversation_id, agent=agent, is_deleted=False)
        
        # Get message and validate it belongs to the conversation
        message = get_object_or_404(ConversationMessages, message_id=message_id, conversation=conversation, is_deleted=False)
        
        data = request.data
        
        # Update message fields with validation
        if "message_text" in data:
            message.message_text = data["message_text"]
        if "is_deleted" in data:
            message.is_deleted = data["is_deleted"]
        if "is_archived" in data:
            message.is_archived = data["is_archived"]
        if "is_read" in data:
            message.is_read = data["is_read"]
            
        message.save()
        
        logger.info(f"Message {message_id} updated successfully by user {request.user.id}")
        
        return Response({
            "message": "Message updated successfully.",
            "message_id": message.message_id,
            "message_text": message.message_text
        }, status=status.HTTP_200_OK)
        
    except ValidationError as e:
        logger.warning(f"Validation error updating message: {str(e)}")
        return Response({
            "error": "Invalid data provided.",
            "details": str(e)
        }, status=status.HTTP_400_BAD_REQUEST)
        
    except Exception as e:
        logger.error(f"Error updating message {message_id}: {str(e)}", exc_info=True)
        return Response({
            "error": "An unexpected error occurred while updating the message."
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def chat_with_ai(request, agent_id, conversation_id):
    """Get AI chat response for a conversation."""
    try:
        # Validate required parameters
        message_text = request.GET.get("message_text")
        if not all([agent_id, conversation_id, message_text]):
            return Response(
                {"error": "Agent ID, conversation ID, and message_text are required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Validate agent ownership
        agent = get_object_or_404(Agent, agent_id=agent_id, is_deleted=False)
        
        # Validate conversation ownership
        conversation = get_object_or_404(
            Conversation,
            conversation_id=conversation_id,
            agent=agent,
            is_deleted=False
        )
        
        # Get conversation history
        conversation_history = conversation.get_ordered_messages
        conversation_history_text = ""
        for message in conversation_history:
            conversation_history_text += f"{message.sender_type}: {message.message_text}\n"
        
        # Prepare chat context
        chat_history = (
            f"You are having a conversation with a user. "
            f"Here is the conversation history: {conversation_history_text}. "
            f"The user asked: {message_text}. "
            f"Please respond to the user's question."
        )
        
        # Generate AI response
        ai_response = initialize_chat_agent(
            user_id=request.user.id,
            agent_id=agent_id,
            query=message_text,
            chat_history=chat_history
        )
        
        # Create messages in transaction
        with transaction.atomic():
            # Create user message
            user_message = ConversationMessages.objects.create(
                conversation=conversation,
                message_text=message_text,
                sender_type="user",
                is_read=True
            )
            
            # Create AI message
            ai_message = ConversationMessages.objects.create(
                conversation=conversation,
                message_text=ai_response,
                sender_type="ai",
                is_read=False
            )
        
        logger.info(f"Chat messages created successfully for conversation {conversation_id}")
        
        return Response({
            "message": "Chat messages created successfully",
            "user_message": {
                "message_id": user_message.message_id,
                "message_text": user_message.message_text,
                "sender_type": "user",
                "timestamp": user_message.message_time,
                "is_read": user_message.is_read,
                "is_deleted": user_message.is_deleted,
                "is_archived": user_message.is_archived
            },
            "ai_message": {
                "message_id": ai_message.message_id,
                "message_text": ai_message.message_text,
                "sender_type": "ai",
                "timestamp": ai_message.message_time,
                "is_read": ai_message.is_read,
                "is_deleted": ai_message.is_deleted,
                "is_archived": ai_message.is_archived
            }
        }, status=status.HTTP_200_OK)

    except Http404:
        logger.warning(f"Agent or conversation not found for user {request.user.id}")
        return Response(
            {"error": "Agent or conversation not found."},
            status=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        logger.error(f"Error in chat_with_ai: {str(e)}", exc_info=True)
        return Response(
            {"error": "An unexpected error occurred while processing the chat."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def check_conversation_active(request, agent_id, conversation_id):
    """Check if a conversation is active within the last minute."""
    try:
        # Validate agent ownership and existence
        agent = get_object_or_404(
            Agent, 
            agent_id=agent_id, 
            is_deleted=False
        )
        
        # Get conversation and validate ownership
        conversation = get_object_or_404(
            Conversation, 
            conversation_id=conversation_id, 
            agent=agent, 
            is_deleted=False
        )
        
        # Check if conversation is active (last message within 1 minute)
        is_active = conversation.check_is_active_with_time(timezone.timedelta(minutes=1))
        
        logger.info(f"Conversation {conversation_id} active status checked: {is_active}")
        
        return Response({
            "is_active": is_active,
            "conversation_id": conversation_id,
            "agent_id": agent_id
        }, status=status.HTTP_200_OK)
        
    except Http404:
        logger.warning(f"Agent {agent_id} or conversation {conversation_id} not found for user {request.user.id}")
        return Response(
            {"error": "Agent or conversation not found."},
            status=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        logger.error(f"Error checking conversation {conversation_id} active status: {str(e)}", exc_info=True)
        return Response(
            {"error": "An unexpected error occurred while checking the conversation active status."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(["GET"])
def get_api_tokens(request):
    """Get a new API token."""
    try:
        api_key = request.GET.get("api_key")
        if not api_key:
            return Response(
                {"error": "API key is required and cannot be empty."},
                status=status.HTTP_400_BAD_REQUEST
            )
        agent_integration = get_object_or_404(AgentIntegrationsFrontend, api_key=api_key, is_deleted=False)
        if not agent_integration:
            return Response(
                {"error": "Invalid API key."},
                status=status.HTTP_400_BAD_REQUEST
            )
        user_subscription = UserSubscription.objects.get(user=agent_integration.agent.user, is_active=True, is_deleted=False)
        if not user_subscription.is_valid_subscription():
            return Response(
                {"error": "You have reached your subscription limit. Please upgrade your subscription to continue."},
                status=status.HTTP_403_FORBIDDEN
            )
        if user_subscription.conversations_usage >= user_subscription.price.plan.feature_limit("conversations"):
            return Response(
                {"error": "You have reached your subscription limit in creating conversations. Please upgrade your subscription to continue."},
                status=status.HTTP_403_FORBIDDEN
            )
        
        frontend_client = FrontendIntegrationClient.objects.create(integration=agent_integration)
        tokens = frontend_client.generate_jwt_tokens()
        logger.info(f"API token created successfully for integration {agent_integration.integration_id}")                
        return Response(tokens, status=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"Error getting API token: {str(e)}", exc_info=True)
        return Response(
            {"error": "An unexpected error occurred while getting the API token."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(["GET"])
def refresh_api_token(request):
    """Refresh an API token."""
    try:
        api_key = request.GET.get("api_key")
        if not api_key:
            return Response(
                {"error": "API key is required and cannot be empty."},
                status=status.HTTP_400_BAD_REQUEST
            )
        agent_integration = get_object_or_404(AgentIntegrationsFrontend, api_key=api_key, is_deleted=False)
        if not agent_integration:
            return Response(
                {"error": "Invalid API key."},
                status=status.HTTP_400_BAD_REQUEST
            )
        user_subscription = UserSubscription.objects.get(user=agent_integration.agent.user, is_active=True, is_deleted=False)
        if not user_subscription.is_valid_subscription():
            return Response(
                {"error": "You have reached your subscription limit. Please upgrade your subscription to continue."},
                status=status.HTTP_403_FORBIDDEN
            )
        if user_subscription.conversations_usage >= user_subscription.price.plan.feature_limit("conversations"):
            return Response(
                {"error": "You have reached your subscription limit in creating conversations. Please upgrade your subscription to continue."},
                status=status.HTTP_403_FORBIDDEN
            )
        frontend_client = get_object_or_404(FrontendIntegrationClient, integration=agent_integration, is_deleted=False)
        refresh_token = request.GET.get("refresh_token")
        if not refresh_token:
            return Response(
                {"error": "Refresh token is required and cannot be empty."},
                status=status.HTTP_400_BAD_REQUEST
            )
        try:
            decoded_token = jwt.decode(refresh_token, os.getenv("CLIENT_SECRET_KEY"), algorithms=["HS256"])
            if decoded_token["type"] != "refresh":
                return Response(
                    {"error": "Invalid token type."},
                    status=status.HTTP_400_BAD_REQUEST
                )
            frontend_client = get_object_or_404(FrontendIntegrationClient, frontend_client_id=decoded_token["frontend_client_id"])
            new_access_token = frontend_client.generate_access_token()
            return Response(new_access_token, status=status.HTTP_200_OK)
        except jwt.ExpiredSignatureError:
            return Response(
                {"error": "Token has expired."},
                status=status.HTTP_400_BAD_REQUEST
            )
        except jwt.InvalidTokenError:
            return Response(
                {"error": "Invalid token."},
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
    except Exception as e:
        logger.error(f"Error refreshing API token: {str(e)}", exc_info=True)
        return Response(
            {"error": "An unexpected error occurred while refreshing the API token."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(["POST"])
def create_api_conversation(request):
    """Create a new API conversation."""
    try:
        data = request.data
        
        # Validate required fields
        if not data:
            return Response(
                {"error": "Request data is required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        token = data.get("token")
        if not token:
            return Response(
                {"error": "Token is required and cannot be empty."},
                status=status.HTTP_400_BAD_REQUEST
            )
        try:
            decoded_token = jwt.decode(token, os.getenv("CLIENT_SECRET_KEY"), algorithms=["HS256"])
            if decoded_token["type"] != "access":
                return Response(
                    {"error": "Invalid token."},
                    status=status.HTTP_400_BAD_REQUEST
                )
            frontend_client = get_object_or_404(FrontendIntegrationClient, frontend_client_id=decoded_token["frontend_client_id"], is_deleted=False)
        except jwt.ExpiredSignatureError:
            return Response(
                {"error": "Token has expired."},
                status=status.HTTP_400_BAD_REQUEST
            )
        except jwt.InvalidTokenError:
            return Response(
                {"error": "Invalid token."},
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
        agent_integrations = frontend_client.integration
        if not agent_integrations:
            return Response(
                {"error": "Invalid API key."},
                status=status.HTTP_400_BAD_REQUEST
            )
        user = agent_integrations.agent.user
        if user != request.user:
            return Response(
                {"error": "You are not authorized to access this API key."},
                status=status.HTTP_403_FORBIDDEN
            )
        
        # Validate user subscription
        user_subscription = UserSubscription.objects.get(user=user, is_active=True, is_deleted=False)
        if not user_subscription.is_valid_subscription():
            return Response(
                {"error": "You have reached your subscription limit. Please upgrade your subscription to continue."},
                status=status.HTTP_403_FORBIDDEN
            )
        if user_subscription.conversations_usage >= user_subscription.price_id.plan.feature_limit("conversations"):
            return Response(
                {"error": "You have reached your subscription limit in creating conversations. Please upgrade your subscription to continue."},
                status=status.HTTP_403_FORBIDDEN
            )

        # Validate agent exists (assuming Agent model exists)
        try:
            agent = agent_integrations.agent
        except Agent.DoesNotExist:
            return Response(
                {"error": "Specified agent does not exist."},
                status=status.HTTP_404_NOT_FOUND
            )        
        # Create conversation with transaction to ensure data consistency
        with transaction.atomic():
            conversation = Conversation.objects.create(
                conversation_name=agent_integrations.website_name,
                agent=agent,
                frontend_client=frontend_client,
                is_active=True,
                is_archived=False,
                is_deleted=False,
                is_favorite=False,
            )
            
            # Create welcome message
            start_message = ConversationMessages.objects.create(
                conversation=conversation,
                message_text=agent_integrations.start_message,
                sender_type="ai",
                is_read=True,
                message_time=timezone.now()  # Ensure timestamp is set
            )
        
        logger.info(f"Conversation created successfully: {conversation.conversation_id} for integration {agent_integrations.integration_id}")
        
        return Response({
            "message": "Conversation created successfully.",
            "conversation_id": conversation.conversation_id,
            "conversation_name": conversation.conversation_name,
            "is_active": conversation.is_active,
            "is_archived": conversation.is_archived,
            "is_deleted": conversation.is_deleted,
            "is_favorite": conversation.is_favorite,
            "ai_message": {
                "message_id": start_message.message_id,
                "message_text": start_message.message_text,
                "sender_type": start_message.sender_type,
                "timestamp": start_message.message_time,
                "is_read": start_message.is_read
            }
        }, status=status.HTTP_201_CREATED)
        
    except ValidationError as e:
        logger.warning(f"Validation error creating conversation: {str(e)}")
        return Response(
            {"error": "Invalid data provided.", "details": str(e)},
            status=status.HTTP_400_BAD_REQUEST
        )
    except IntegrityError as e:
        logger.error(f"Database integrity error creating conversation: {str(e)}")
        return Response(
            {"error": "Database constraint violation."},
            status=status.HTTP_400_BAD_REQUEST
        )
    except Exception as e:
        logger.error(f"Unexpected error creating conversation: {str(e)}", exc_info=True)
        return Response(
            {"error": "An unexpected error occurred while creating the conversation."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(["GET"])
def chat_with_api(request, conversation_id):
    """Get AI chat response for a conversation."""
    try:
        # Validate required parameters
        message_text = request.GET.get("message_text")
        token = request.GET.get("token")
        if not all([conversation_id, message_text, token]):
            return Response(
                {"error": "Conversation ID and message_text are required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        try:
            decoded_token = jwt.decode(token, os.getenv("CLIENT_SECRET_KEY"), algorithms=["HS256"])
            if decoded_token["type"] != "access":
                return Response(
                    {"error": "Invalid token."},
                    status=status.HTTP_400_BAD_REQUEST
                )
            frontend_client = get_object_or_404(FrontendIntegrationClient, frontend_client_id=decoded_token["frontend_client_id"], is_deleted=False)
        except jwt.ExpiredSignatureError:
            return Response(
                {"error": "Token has expired."},
                status=status.HTTP_400_BAD_REQUEST
            )
        except jwt.InvalidTokenError:
            return Response(
                {"error": "Invalid token."},
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Validate agent ownership
        agent_integrations = frontend_client.integration
        if not agent_integrations:
            return Response(
                {"error": "Invalid API key."},
                status=status.HTTP_400_BAD_REQUEST
            )
        user = agent_integrations.agent.user
        if user != request.user:
            return Response(
                {"error": "You are not authorized to access this API key."},
                status=status.HTTP_403_FORBIDDEN
            )
        
        # Validate conversation ownership
        conversation = get_object_or_404(
            Conversation,
            conversation_id=conversation_id,
            agent=agent_integrations.agent,
            frontend_client=frontend_client,
            is_deleted=False
        )
        
        # Get conversation history
        conversation_history = conversation.get_ordered_messages
        conversation_history_text = ""
        for message in conversation_history:
            conversation_history_text += f"{message.sender_type}: {message.message_text}\n"
        
        # Prepare chat context
        chat_history = (
            f"You are having a conversation with a user. "
            f"Here is the conversation history: {conversation_history_text}. "
            f"The user asked: {message_text}. "
            f"Please respond to the user's question."
        )
        
        # Generate AI response
        ai_response = initialize_chat_agent(
            user_id=user.id,
            agent_id=agent_integrations.agent.agent_id,
            query=message_text,
            chat_history=chat_history
        )
        
        # Create messages in transaction
        with transaction.atomic():
            # Create user message
            user_message = ConversationMessages.objects.create(
                conversation=conversation,
                message_text=message_text,
                sender_type="user",
                is_read=True
            )
            
            # Create AI message
            ai_message = ConversationMessages.objects.create(
                conversation=conversation,
                message_text=ai_response,
                sender_type="ai",
                is_read=False
            )
        
        logger.info(f"Chat messages created successfully for conversation {conversation_id}")
        
        return Response({
            "message": "Chat messages created successfully",
            "user_message": {
                "message_id": user_message.message_id,
                "message_text": user_message.message_text,
                "sender_type": "user",
                "timestamp": user_message.message_time,
                "is_read": user_message.is_read,
                "is_deleted": user_message.is_deleted,
                "is_archived": user_message.is_archived
            },
            "ai_message": {
                "message_id": ai_message.message_id,
                "message_text": ai_message.message_text,
                "sender_type": "ai",
                "timestamp": ai_message.message_time,
                "is_read": ai_message.is_read,
                "is_deleted": ai_message.is_deleted,
                "is_archived": ai_message.is_archived
            }
        }, status=status.HTTP_200_OK)

    except Http404:
        logger.warning(f"Agent or conversation not found for user {user.id}")
        return Response(
            {"error": "Agent or conversation not found."},
            status=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        logger.error(f"Error in chat_with_api: {str(e)}", exc_info=True)
        return Response(
            {"error": "An unexpected error occurred while processing the chat."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )    
