from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework import status
from django.shortcuts import get_object_or_404
from django.db import transaction
from .models import Agent, AgentDocuments, AgentTexts, AgentIntegrations, AgentEmbeddings, AgentVectorsDatabase, AgentWebsites, AgentIntegrationsFrontend
from .embedding.openai import create_embedding_model
from .vdb.pinecone_vdb import PineconeVDB
from bs4 import BeautifulSoup
from plans.models import UserSubscription
import logging
import os   
import tiktoken
from django.utils import timezone
from dotenv import load_dotenv
import secrets
from django.core.exceptions import ValidationError
from django.db import IntegrityError
from django.http import Http404
from accounts.models import Team, TeamMember, TeamAgent
from django.db.models import Q

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)




@api_view(['GET'])
@permission_classes([IsAuthenticated])
def agent_list(request):
    """
    List all agents for the current user.
    Returns a list of active (non-deleted) agents with pagination support.
    """
    try:
        # Validate and set pagination parameters with defaults
        try:
            page = max(1, int(request.query_params.get('page', 1)))
            page_size = min(max(1, int(request.query_params.get('page_size', 10))), 100)
        except (ValueError, TypeError):
            return Response({
                'error': 'Invalid pagination parameters',
                'detail': 'Page and page_size must be positive integers'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Optimize team membership query with proper field filtering
        team_memberships = TeamMember.objects.filter(
            team_member_email=request.user.email,  # Use user instead of team_member_email for consistency
            is_active=True,
            is_deleted=False
        ).select_related('team')
        
        # Get user's own agents
        user_agents = Agent.objects.filter(user=request.user, is_deleted=False)
        agents = []
        
        # Build agent query based on team membership status
        if team_memberships.exists():
            # Get active teams where user is a member
            user_teams = [tm.team for tm in team_memberships if tm.team.is_active and not tm.team.is_deleted]
            team_agents = TeamAgent.objects.filter(team__in=user_teams, is_active=True, is_deleted=False).select_related('agent').order_by('-created_at')
            real_team_agents = [team_agent.agent for team_agent in team_agents]
            agents.extend(real_team_agents)

            if user_agents.exists():    
                # User has both team memberships and own agents
                agents.extend(user_agents)
                logger.info(f"Retrieved {len(agents)} agents from user's account and teams")
        else:
            # User has no team memberships, only get their own agents
            agents = user_agents.order_by('-created_at')
            logger.info(f"Retrieved {len(agents)} agents from user's own account")
        
        # Optimize pagination with select_related and prefetch_related
        total_count = len(agents)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        # Get paginated agents with optimized queries
        paginated_agents = agents[start_idx:end_idx]
        
        # Build response data with error handling for method calls
        agent_data = []
        for agent in paginated_agents:
            try:
                agent_info = {
                    'agent_id': agent.agent_id,
                    'name': agent.name,
                    'description': agent.description,
                    'created_at': agent.created_at.isoformat() if agent.created_at else None,
                    'updated_at': agent.updated_at.isoformat() if agent.updated_at else None,
                    'is_archived': agent.is_archived,
                    'is_favorite': agent.is_favorite,
                    'visibility': agent.visibility,
                    'avatar_url': agent.avatar_url,
                    'configuration': agent.configuration,
                    'teams': [tm.team.team_name for tm in team_memberships if tm.team.is_active and not tm.team.is_deleted],
                }
                
                # Safely call methods that might not exist
                try:
                    agent_info['is_active'] = agent.is_active()
                except AttributeError:
                    agent_info['is_active'] = False
                
                try:
                    agent_info['conversation_count'] = agent.conversation_count()
                except AttributeError:
                    agent_info['conversation_count'] = 0
                
                try:
                    agent_info['documents_summary'] = agent.get_documents_summary()
                except AttributeError:
                    agent_info['documents_summary'] = {}
                
                agent_data.append(agent_info)
                
            except Exception as agent_error:
                logger.warning(f"Error processing agent {agent.agent_id}: {str(agent_error)}")
                continue
        
        return Response({
            'agents': agent_data,
            'pagination': {
                'total_count': total_count,
                'page': page,
                'page_size': page_size,
                'total_pages': (total_count + page_size - 1) // page_size,
                'has_next': page * page_size < total_count,
                'has_previous': page > 1
            }
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Failed to retrieve agents for user {request.user.id}: {str(e)}")
        return Response({
            'error': 'Failed to retrieve agents',
            'detail': 'An unexpected error occurred while fetching agents'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def agent_details(request, agent_id):
    """
    Get a single agent's details with optimized database queries.
    Returns 404 if agent not found or doesn't belong to user.
    """
    try:
        agent = get_object_or_404(
            Agent,
            agent_id=agent_id,
            is_deleted=False
        )
        
        team_agents = TeamAgent.objects.filter(agent=agent, is_active=True, is_deleted=False)
        
        if team_agents.exists():
            team_agents = [team_agent.team.team_name for team_agent in team_agents]
        else:
            team_agents = []

        # Cache expensive computations
        is_active = agent.is_active()
        conversation_count = agent.conversation_count() if hasattr(agent, 'conversation_count') else 0
        documents_summary = agent.get_documents_summary() if hasattr(agent, 'get_documents_summary') else {}
        
        agent_data = {
            'agent_id': agent.agent_id,
            'name': agent.name,
            'description': agent.description,
            'created_at': agent.created_at.isoformat() if agent.created_at else None,
            'updated_at': agent.updated_at.isoformat() if agent.updated_at else None,
            'is_archived': agent.is_archived,
            'is_favorite': agent.is_favorite,
            'visibility': agent.visibility,
            'avatar_url': agent.avatar_url,
            'configuration': agent.configuration,
            'is_active': is_active,
            'conversation_count': conversation_count,
            'documents_summary': documents_summary,
            'teams': team_agents,
            'no_of_documents': agent.no_of_documents(),
            'no_of_texts': agent.no_of_texts(),
            'no_of_websites': agent.no_of_websites(),
            'no_of_conversations': agent.no_of_conversations(),
            'no_of_team_members': agent.no_of_team_members(),
        }
        
        return Response(agent_data, status=status.HTTP_200_OK)
        
    except Agent.DoesNotExist:
        logger.warning(f"Agent {agent_id} not found for user {request.user.id}")
        return Response({
            'error': 'Agent not found',
            'detail': 'The requested agent does not exist or has been deleted'
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        logger.error(f"Failed to retrieve agent {agent_id}: {str(e)}")
        return Response({
            'error': 'Failed to retrieve agent',
            'detail': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def create_agent(request):
    """
    Create a new agent with validation and atomic transactions.
    Returns 201 on success, 400 on validation error.
    """
    try:
        # Extract and validate required fields
        name = request.data.get('name', '').strip()
        if not name:
            return Response({
                'error': 'Agent name is required',
                'detail': 'Name cannot be empty'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Validate name length
        if len(name) > 100:  # Assuming max length is 100
            return Response({
                'error': 'Invalid agent name',
                'detail': 'Name must be less than 100 characters'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        user_subscription = UserSubscription.objects.get(user=request.user, is_active=True, is_deleted=False)
        if not user_subscription.is_valid_subscription():
            return Response(
                {"error": "You have reached your subscription limit. Please upgrade your subscription to continue."},
                status=status.HTTP_403_FORBIDDEN
            )
        if user_subscription.agents_usage >= user_subscription.price.plan.feature_limit("agents"):
            return Response(
                {"error": "You have reached your subscription limit in creating agents. Please upgrade your subscription to continue."},
                status=status.HTTP_403_FORBIDDEN
            )
        
        # Check for duplicate agent names for this user
        if Agent.objects.filter(user=request.user, name=name, is_deleted=False).exists():
            return Response({
                'error': 'Duplicate agent name',
                'detail': 'An agent with this name already exists'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Extract and validate optional fields
        description = request.data.get('description', '').strip()
        visibility = request.data.get('visibility', 'private')
        avatar_url = request.data.get('avatar_url', '').strip()
        configuration = request.data.get('configuration', {})
        
        # Validate visibility choice
        if visibility not in ['public', 'private']:
            visibility = 'private'
            logger.warning(f"Invalid visibility value provided, defaulting to private")
        
        # Validate configuration is a dictionary
        if not isinstance(configuration, dict):
            configuration = {}
            logger.warning("Invalid configuration provided, defaulting to empty dict")
        
        # Create both agent and VDB in a single transaction
        with transaction.atomic():
            agent = Agent.objects.create(
                user=request.user,
                name=name,
                description=description,
                visibility=visibility,
                avatar_url=avatar_url if avatar_url else None,
                configuration=configuration
            )
            
            vdb = AgentVectorsDatabase.objects.create(
                agent=agent,
                database_path=agent.name,
                namespace=agent.name
            )
            user_subscription.agents_usage += 1
            user_subscription.save(update_fields=['agents_usage'])
            logger.info(f"Created new agent {agent.agent_id} with VDB for user {request.user.id}")
        
        return Response({
            'message': 'Agent created successfully',
            'agent': agent.get_agent_details() # Use model method for consistent response
        }, status=status.HTTP_201_CREATED)
        
    except Exception as e:
        logger.error(f"Failed to create agent: {str(e)}")
        return Response({
            'error': 'Failed to create agent',
            'detail': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['PUT'])
@permission_classes([IsAuthenticated])
def update_agent(request, agent_id):
    """
    Update an existing agent with validation and atomic transactions.
    Returns 200 on success, 400 on validation error.
    """
    try:
        # Validate required fields first
        name = request.data.get('name', '').strip()
        if not name:
            return Response({
                'error': 'Agent name is required',
                'detail': 'Name cannot be empty'
            }, status=status.HTTP_400_BAD_REQUEST)
            
        # Validate name length
        if len(name) > 100:  # Assuming max length is 100
            return Response({
                'error': 'Invalid agent name',
                'detail': 'Name must be less than 100 characters'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Get agent and VDB in a single query using select_related
        agent = get_object_or_404(
            Agent.objects.select_related('agentvectorsdatabase'),
            agent_id=agent_id,
            user=request.user,
            is_deleted=False
        )
        vdb = agent.agentvectorsdatabase
        
        # Check for name conflicts if name is being updated
        if name != agent.name:
            if Agent.objects.filter(user=request.user, name=name, is_deleted=False).exclude(agent_id=agent_id).exists():
                return Response({
                    'error': 'Duplicate agent name',
                    'detail': 'Another agent with this name already exists'
                }, status=status.HTTP_400_BAD_REQUEST)
        
        # Track which fields to update
        # Validate agent exists and belongs to user
        agent = get_object_or_404(Agent, agent_id=agent_id, user=request.user, is_deleted=False)

        # Get the agent vector database
        vdb = get_object_or_404(AgentVectorsDatabase, agent=agent, is_deleted=False)
        
        # Track which fields to update
        update_fields = ['updated_at']
        vdb_update_fields = ['updated_at']
        
        # Update fields if provided
        if 'name' in request.data:
            agent.name = request.data.get('name')
            update_fields.append('name')
            vdb.database_path = agent.name
            vdb.namespace = agent.name
            vdb_update_fields.append('database_path')
            vdb_update_fields.append('namespace')
            
        if 'description' in request.data:
            agent.description = request.data.get('description')
            update_fields.append('description')
            
        if 'visibility' in request.data:
            agent.visibility = request.data.get('visibility')
            update_fields.append('visibility')
            
        if 'avatar_url' in request.data:
            agent.avatar_url = request.data.get('avatar_url')
            update_fields.append('avatar_url')
            
        if 'configuration' in request.data:
            agent.configuration = request.data.get('configuration')
            update_fields.append('configuration')
            
        with transaction.atomic():
            agent.save(update_fields=update_fields)
            vdb.save(update_fields=vdb_update_fields)
            
        return Response({
            'message': 'Agent updated successfully',
            'agent': agent.to_dict()
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({
            'error': 'Failed to update agent',
            'detail': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_agent(request, agent_id):
    """
    Soft delete an existing agent and all associated resources.
    Returns 204 on success, 404 if agent not found.
    """
    try:
        # Get agent with select_related to optimize query
        agent = get_object_or_404(
            Agent.objects.select_related('user'),
            agent_id=agent_id,
            user=request.user,
            is_deleted=False
        )

        # Get associated resources with select_related
        vdb = get_object_or_404(
            AgentVectorsDatabase.objects.select_related('agent'),
            agent=agent,
            is_deleted=False
        )
        
        # Get embeddings for cleanup
        embeddings = AgentEmbeddings.objects.filter(
            agent=agent,
            is_deleted=False
        ).select_related('agent')

        # Get documents for cleanup
        documents = AgentDocuments.objects.filter(
            agent=agent,
            is_deleted=False
        ).select_related('agent')

        # Get texts for cleanup
        texts = AgentTexts.objects.filter(
            agent=agent,
            is_deleted=False
        ).select_related('agent')

        # Get integrations for cleanup
        integrations = AgentIntegrations.objects.filter(
            agent=agent,
            is_deleted=False
        ).select_related('agent')

        with transaction.atomic():
            # Update timestamps
            now = timezone.now()
            
            # Soft delete agent
            agent.is_deleted = True
            agent.updated_at = now
            agent.save(update_fields=['is_deleted', 'updated_at'])
            
            # Soft delete vector database
            vdb.is_deleted = True
            vdb.updated_at = now
            vdb.save(update_fields=['is_deleted', 'updated_at'])
            
            # Soft delete embeddings
            if embeddings.exists():
                embeddings.update(is_deleted=True, updated_at=now)
            
            # Soft delete documents
            if documents.exists():
                documents.update(is_deleted=True, updated_at=now)
            
            # Soft delete texts
            if texts.exists():
                texts.update(is_deleted=True, updated_at=now)
            
            # Soft delete integrations
            if integrations.exists():
                integrations.update(is_deleted=True, updated_at=now)

            logger.info(f"Agent {agent_id} and associated resources deleted by user {request.user.id}")
        
        return Response(status=status.HTTP_204_NO_CONTENT)
    
    except Agent.DoesNotExist:
        logger.warning(f"Agent {agent_id} not found for deletion")
        return Response({
            'error': 'Agent not found',
            'detail': 'The requested agent does not exist or has already been deleted'
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        logger.error(f"Error deleting agent {agent_id}: {str(e)}")
        return Response({
            'error': 'Failed to delete agent',
            'detail': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def create_agent_document(request, agent_id):
    """
    Create a new agent document.
    Returns 201 on success, 400 on validation error.
    """
    try:
        # Extract and validate required fields
        document_name = request.data.get('document_name')
        document_format = request.data.get('document_format')
        document_path = request.data.get('document_path')
        document_file = request.FILES.get('document_file')
        document_size = request.data.get('document_size')
        document_description = request.data.get('document_description', '')
        document_language = request.data.get('document_language', 'en')
        document_tags = request.data.get('document_tags', [])
        meta_data = request.data.get('meta_data', {})
        
        # Validate required fields
        required_fields = {
            'document_name': document_name,
            'document_format': document_format,
            'document_file': document_file,
        }
        
        missing_fields = [field for field, value in required_fields.items() if not value]
        if missing_fields:
            return Response({
                'error': 'Missing required fields',
                'missing_fields': [field.replace('_', ' ').title() for field in missing_fields]
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Validate file format
        allowed_formats = [
            'application/pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'text/plain',
            'text/csv',
            'application/json'
        ]
        if document_format not in allowed_formats:
            return Response({
                'error': 'Invalid document format',
                'allowed_formats': allowed_formats
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Validate file size (max 10MB)
        max_size = 50 * 1024 * 1024  # 50MB in bytes
        if document_file.size > max_size:
            return Response({
                'error': 'File too large',
                'max_size_mb': 50
            }, status=status.HTTP_400_BAD_REQUEST)
        
        user_subscription = UserSubscription.objects.get(user=request.user, is_active=True, is_deleted=False)
        if not user_subscription.is_valid_subscription():
            return Response(
                {"error": "You have reached your subscription limit. Please upgrade your subscription to continue."},
                status=status.HTTP_403_FORBIDDEN
            )
        if user_subscription.tokens_usage >= user_subscription.price.plan.feature_limit("tokens"):
            return Response(
                {"error": "You have reached your subscription limit in training your agent. Please upgrade your subscription to continue."},
                status=status.HTTP_403_FORBIDDEN
            )
        
        # Validate agent exists and belongs to user
        agent = get_object_or_404(Agent, agent_id=agent_id, user=request.user, is_deleted=False)
        
        # Get the agent vector database
        vdb = get_object_or_404(AgentVectorsDatabase, agent=agent)
        
        # Calculate actual file size if not provided
        if not document_size:
            document_size = document_file.size
        
        # Create the document with transaction
        with transaction.atomic():
            document = AgentDocuments.objects.create(
                agent=agent,
                document_name=document_name,
                document_format=document_format,
                document_path=document_path,
                document_file=document_file,
                document_size=document_size,
                document_description=document_description,
                document_language=document_language,
                document_tags=document_tags,
                meta_data=meta_data
            )
            
            # Log document creation
            logger.info(f"Document created: {document.document_id} - {document_name} for agent {agent_id}")
        
        # Return document details using the model's get_document_details method
        return Response({
            'message': 'Document created successfully',
            'document': document.get_document_details()
        }, status=status.HTTP_201_CREATED)
    
    except Agent.DoesNotExist:
        return Response({
            'error': 'Agent not found'
        }, status=status.HTTP_404_NOT_FOUND)
    except AgentVectorsDatabase.DoesNotExist:
        return Response({
            'error': 'Agent vector database not found'
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        logger.error(f"Error creating document: {str(e)}")
        return Response({
            'error': 'Failed to create document',
            'detail': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_agent_documents(request, agent_id):
    """
    Get all documents for a specific agent.
    Returns 200 on success, 404 if agent not found.
    Includes document metadata and formatted size.
    """
    try:
        # Get agent and verify ownership
        agent = get_object_or_404(Agent, agent_id=agent_id, user=request.user, is_deleted=False)
        
        # Get documents with select_related to optimize query
        documents = (AgentDocuments.objects
                    .filter(agent=agent, is_deleted=False)
                    .select_related('agent')
                    .order_by('-created_at'))
        
        # Use list comprehension for better performance
        document_data = [{
            'document_id': doc.document_id,
            'document_name': doc.document_name,
            'document_format': doc.document_format,
            'document_url': doc.document_path,  # Changed from document_url to match model
            'document_size': doc.formatted_size,  # Use the formatted_size property
            'document_description': doc.document_description,
            'document_language': doc.document_language,
            'document_tags': doc.document_tags or [],
            "embedded": doc.embedded,
            'created_at': doc.created_at.isoformat(),
            'updated_at': doc.updated_at.isoformat(),
            'is_expired': doc.is_expired(),
        } for doc in documents]
            
        return Response({
            'documents': document_data,
            'count': len(document_data),
            'agent_id': agent_id,
            'agent_name': agent.name
        }, status=status.HTTP_200_OK)
    
    except Agent.DoesNotExist:
        logger.warning(f"Agent {agent_id} not found for user {request.user.id}")
        return Response({
            'error': 'Agent not found',
            'detail': 'The requested agent does not exist or you do not have permission to access it'
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        logger.error(f"Error retrieving documents for agent {agent_id}: {str(e)}")
        return Response({
            'error': 'Failed to retrieve documents',
            'detail': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_agent_document(request, agent_id, document_id):
    """
    Soft delete a specific agent document.
    Returns 204 on success, 404 if document not found.
    """
    try:
        # Get agent and verify ownership
        agent = get_object_or_404(Agent, agent_id=agent_id, user=request.user, is_deleted=False)
        if agent.user != request.user:
            logger.warning(f"Permission denied: User {request.user.id} attempted to delete document {document_id} without permission")
            return Response({
                'error': 'Permission denied',
                'detail': 'You do not have permission to delete this document'
            }, status=status.HTTP_403_FORBIDDEN)
        
        # Get document with select_related to optimize query
        document = get_object_or_404(AgentDocuments, agent=agent, document_id=document_id, is_deleted=False)
        if document.embedded:
            embedding = get_object_or_404(AgentEmbeddings, agent=document.agent, object_id=document_id, object_type='document', is_deleted=False)
        
        with transaction.atomic():
            document.is_deleted = True
            document.updated_at = timezone.now()
            document.save(update_fields=['is_deleted', 'updated_at'])
            
            if document.embedded:
                embedding.delete_embedding()

            logger.info(f"Document {document_id} soft deleted by user {request.user.id}")
        
        return Response(status=status.HTTP_204_NO_CONTENT)
    
    except Agent.DoesNotExist:
        logger.warning(f"Agent {agent_id} not found for user {request.user.id}")
        return Response({
            'error': 'Agent not found',
            'detail': 'The requested agent does not exist or you do not have permission to access it'
        }, status=status.HTTP_404_NOT_FOUND)
    except AgentDocuments.DoesNotExist:
        logger.warning(f"Document {document_id} not found for deletion")
        return Response({
            'error': 'Document not found',
            'detail': 'The requested document does not exist or has already been deleted'
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {str(e)}")
        return Response({
            'error': 'Failed to delete document',
            'detail': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['PUT'])  
@permission_classes([IsAuthenticated])
def update_agent_document(request, document_id):
    """
    Update a specific agent document's metadata.
    Returns 200 on success, 404 if document not found.
    """
    try:
        # Get document with select_related to optimize query
        document = get_object_or_404(AgentDocuments, document_id=document_id, is_deleted=False)
        
        # Verify user owns the agent that owns this document
        if document.agent.user != request.user:
            logger.warning(f"Permission denied: User {request.user.id} attempted to update document {document_id}")
            return Response({
                'error': 'Permission denied',
                'detail': 'You do not have permission to update this document'
            }, status=status.HTTP_403_FORBIDDEN)

        # Define allowed fields and their validation
        allowed_fields = {
            'document_name': str,
            'document_description': str,
            'document_format': str,
            'document_path': str,
            'document_language': str,
            'document_tags': list,
            'meta_data': dict
        }
        
        # Track which fields to update
        update_fields = ['updated_at']
        document.updated_at = timezone.now()
        
        # Update fields if provided and valid
        for field, field_type in allowed_fields.items():
            if field in request.data:
                value = request.data.get(field)
                if not isinstance(value, field_type):
                    return Response({
                        'error': f'Invalid type for {field}',
                        'detail': f'Expected {field_type.__name__}, got {type(value).__name__}'
                    }, status=status.HTTP_400_BAD_REQUEST)
                setattr(document, field, value)
                update_fields.append(field)
        
        with transaction.atomic():
            document.save(update_fields=update_fields)
            logger.info(f"Document {document_id} updated by user {request.user.id}")
        
        return Response({
            'message': 'Document updated successfully',
            'document': document.get_document_details()  # Use model's method for consistent response
        }, status=status.HTTP_200_OK)       
    
    except AgentDocuments.DoesNotExist:
        logger.warning(f"Document {document_id} not found for update")
        return Response({
            'error': 'Document not found',
            'detail': 'The requested document does not exist or has been deleted'
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        logger.error(f"Error updating document {document_id}: {str(e)}")
        return Response({
            'error': 'Failed to update document',
            'detail': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def create_agent_text(request, agent_id):
    """
    Create a new agent text.
    Returns 201 on success, 400 on validation error.
    """
    try:
        # Extract and validate required fields
        text_content = request.data.get('text_content')
        text_title = request.data.get('text_title', '')
        meta_data = request.data.get('meta_data', {})
        
        # Validate required fields
        if not text_content or not isinstance(text_content, str):
            return Response({
                'error': 'Valid text content is required',
                'detail': 'Text content must be a non-empty string'
            }, status=status.HTTP_400_BAD_REQUEST)
            
        # Validate text length (max 100,000 characters)
        if len(text_content) > 100000:
            return Response({
                'error': 'Text content too long',
                'detail': 'Maximum text length is 100,000 characters'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        user_subscription = UserSubscription.objects.get(user=request.user, is_active=True, is_deleted=False)
        if not user_subscription.is_valid_subscription():
            return Response(
                {"error": "You have reached your subscription limit. Please upgrade your subscription to continue."},
                status=status.HTTP_403_FORBIDDEN
            )
        if user_subscription.tokens_usage >= user_subscription.price.plan.feature_limit("tokens"):
            return Response(
                {"error": "You have reached your subscription limit in training your agent. Please upgrade your subscription to continue."},
                status=status.HTTP_403_FORBIDDEN
            )
        
        # Validate agent exists and belongs to user
        agent = get_object_or_404(Agent, agent_id=agent_id, user=request.user, is_deleted=False)
        
        
        # Create the text with transaction
        with transaction.atomic():
            text = AgentTexts.objects.create(
                agent=agent,
                text_content=text_content,
                text_title=text_title,
                meta_data=meta_data
            )
            logger.info(f"Text created: {text.text_id} for agent {agent_id}")

        
        return Response({
            'message': 'Text created successfully',
            'text': text.get_text_details()  # Use model method for consistent response
        }, status=status.HTTP_201_CREATED)
    
    except Agent.DoesNotExist:
        logger.warning(f"Agent {agent_id} not found for text creation")
        return Response({
            'error': 'Agent not found',
            'detail': 'The requested agent does not exist or you do not have permission to access it'
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        logger.error(f"Error creating text for agent {agent_id}: {str(e)}")
        return Response({
            'error': 'Failed to create text',
            'detail': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_agent_texts(request, agent_id):
    """
    Get all texts for a specific agent.
    Returns 200 on success, 404 if agent not found.
    Includes pagination and filtering options.
    """
    try:
        # Get agent and verify ownership
        agent = get_object_or_404(Agent, agent_id=agent_id, user=request.user, is_deleted=False)
        
        # Get query parameters for filtering and pagination
        page = int(request.query_params.get('page', 1))
        page_size = min(int(request.query_params.get('page_size', 10)), 100)  # Max 100 items per page
        search_query = request.query_params.get('search', '')
        sort_by = request.query_params.get('sort_by', '-created_at')
        
        # Base queryset with select_related for optimization
        texts = (AgentTexts.objects
                .filter(agent=agent, is_deleted=False)
                .select_related('agent'))
        
        # Apply search filter if provided
        if search_query:
            texts = texts.filter(text_content__icontains=search_query)
        
        # Apply sorting
        allowed_sort_fields = ['created_at', '-created_at', 'updated_at', '-updated_at']
        if sort_by in allowed_sort_fields:
            texts = texts.order_by(sort_by)
        
        # Calculate pagination
        total_count = texts.count()
        start_index = (page - 1) * page_size
        end_index = start_index + page_size
        
        # Get paginated texts
        texts = texts[start_index:end_index]
        
        # Use list comprehension for better performance
        text_data = [text.get_text_details() for text in texts]
            
        return Response({
            'texts': text_data,
            'count': total_count,
            'page': page,
            'page_size': page_size,
            'total_pages': (total_count + page_size - 1) // page_size,
            'agent_id': agent_id,
            'agent_name': agent.name
        }, status=status.HTTP_200_OK)
    
    except Agent.DoesNotExist:
        logger.warning(f"Agent {agent_id} not found for text retrieval")
        return Response({
            'error': 'Agent not found',
            'detail': 'The requested agent does not exist or you do not have permission to access it'
        }, status=status.HTTP_404_NOT_FOUND)
    except ValueError as e:
        logger.warning(f"Invalid pagination parameters: {str(e)}")
        return Response({
            'error': 'Invalid pagination parameters',
            'detail': str(e)
        }, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        logger.error(f"Error retrieving texts for agent {agent_id}: {str(e)}")
        return Response({
            'error': 'Failed to retrieve texts',
            'detail': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
@api_view(['DELETE'])   
@permission_classes([IsAuthenticated])
def delete_agent_text(request, text_id):
    """
    Delete a specific agent text.
    Returns 204 on success, 404 if text not found.
    """ 
    try:
        text = get_object_or_404(AgentTexts, text_id=text_id, is_deleted=False)
        if text.embedded:
            embedding = get_object_or_404(AgentEmbeddings, agent=text.agent, object_id=text_id, object_type='text', is_deleted=False)
        
        # Verify user owns the agent that owns this text
        if text.agent.user != request.user:
            logger.warning(f"User {request.user.id} attempted to delete text {text_id} without permission")
            return Response({
                'error': 'Permission denied',
                'detail': 'You do not have permission to delete this text'
            }, status=status.HTTP_403_FORBIDDEN)
        
        with transaction.atomic():
            text.is_deleted = True
            text.updated_at = timezone.now()
            text.save(update_fields=['is_deleted', 'updated_at'])
            if text.embedded:
                embedding.delete_embedding()
            logger.info(f"Text {text_id} deleted by user {request.user.id}")
        
        return Response({
            'message': 'Text deleted successfully'
        }, status=status.HTTP_204_NO_CONTENT)
    
    except AgentTexts.DoesNotExist:
        logger.warning(f"Text {text_id} not found for deletion")
        return Response({
            'error': 'Text not found',
            'detail': 'The requested text does not exist or has been deleted'
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        logger.error(f"Error deleting text {text_id}: {str(e)}")
        return Response({
            'error': 'Failed to delete text',
            'detail': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
@api_view(['PUT'])
@permission_classes([IsAuthenticated])
def update_agent_text(request, text_id):
    """
    Update a specific agent text.
    Returns 200 on success, 404 if text not found.
    """
    try:
        text = get_object_or_404(AgentTexts, text_id=text_id, is_deleted=False)
        
        # Verify user owns the agent that owns this text
        if text.agent.user != request.user:
            logger.warning(f"User {request.user.id} attempted to update text {text_id} without permission")
            return Response({
                'error': 'Permission denied',
                'detail': 'You do not have permission to update this text'
            }, status=status.HTTP_403_FORBIDDEN)
        
        # Define allowed fields and their validation
        allowed_fields = {
            'text_content': str,
            'text_title': str,
            'text_description': str,
            'text_tags': list,
            'meta_data': dict
        }
        
        # Track which fields to update
        update_fields = ['updated_at']
        text.updated_at = timezone.now()
        
        # Update fields if provided and valid
        for field, field_type in allowed_fields.items():
            if field in request.data:
                value = request.data.get(field)
                if not isinstance(value, field_type):
                    return Response({
                        'error': f'Invalid type for {field}',
                        'detail': f'Expected {field_type.__name__}, got {type(value).__name__}'
                    }, status=status.HTTP_400_BAD_REQUEST)
                setattr(text, field, value)
                update_fields.append(field)
        
        # Validate text content length if being updated
        if 'text_content' in update_fields and len(text.text_content) > 100000:
            return Response({
                'error': 'Text content too long',
                'detail': 'Maximum text length is 100,000 characters' 
            }, status=status.HTTP_400_BAD_REQUEST)
        
        with transaction.atomic():
            text.save(update_fields=update_fields)
            logger.info(f"Text {text_id} updated by user {request.user.id}")
        with transaction.atomic():
            text.save(update_fields=['text_content', 'updated_at'])
        
        return Response({
            'message': 'Text updated successfully',
            'text': {
                'text_id': text.text_id,
                'text_content': text.text_content,
                'created_at': text.created_at.isoformat(),
                'updated_at': text.updated_at.isoformat()
            }
        }, status=status.HTTP_200_OK)   
    
    except AgentTexts.DoesNotExist:
        return Response({
            'error': 'Text not found'
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({
            'error': 'Failed to update text',
            'detail': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def create_agent_integration(request, agent_id):
    """
    Create a new agent integration.
    Returns 201 on success, 400 on validation error.
    """
    try:
        # Extract and validate required fields
        integration_type = request.data.get('integration_type')
        
        if not integration_type:
            return Response({
                'error': 'Integration type is required'
            }, status=status.HTTP_400_BAD_REQUEST)  
        
        # Validate agent exists and belongs to user
        agent = get_object_or_404(Agent, agent_id=agent_id, user=request.user, is_deleted=False)
        
        # Create the integration with transaction
        with transaction.atomic():
            integration = AgentIntegrations.objects.create(
                agent=agent,
                integration_type=integration_type   
            )
        
        integration_data = {
            'integration_id': integration.integration_id,
            'integration_type': integration.integration_type,
            'created_at': integration.created_at.isoformat(),   
            'updated_at': integration.updated_at.isoformat()
        }
        
        return Response({
            'message': 'Integration created successfully',
            'integration': integration_data
        }, status=status.HTTP_201_CREATED)
    
    except Agent.DoesNotExist:
        return Response({
            'error': 'Agent not found'
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({
            'error': 'Failed to create integration',
            'detail': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)    
    
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_agent_integrations(request, agent_id):
    """
    Get all integrations for a specific agent.
    Returns 200 on success, 404 if agent not found.
    """
    try:
        agent = get_object_or_404(Agent, agent_id=agent_id, user=request.user, is_deleted=False)
        
        integrations = AgentIntegrations.objects.filter(agent=agent, is_deleted=False).order_by('-created_at')
        
        integration_data = []
        for integration in integrations:
            integration_data.append({
                'integration_id': integration.integration_id,
                'integration_name': integration.integration_name,
                'created_at': integration.created_at.isoformat(),
                'updated_at': integration.updated_at.isoformat()
            })
            
        return Response({
            'integrations': integration_data,
            'count': len(integration_data)
        }, status=status.HTTP_200_OK)
    
    except Agent.DoesNotExist:
        return Response({
            'error': 'Agent not found'
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({
            'error': 'Failed to retrieve integrations',
            'detail': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)    

@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_agent_integration(request, integration_id):
    """
    Delete a specific agent integration.
    Returns 204 on success, 404 if integration not found.
    """
    try:
        integration = get_object_or_404(AgentIntegrations, integration_id=integration_id, is_deleted=False)
        if integration.embedded:
            embedding = get_object_or_404(AgentEmbeddings, agent=integration.agent, object_id=integration_id, object_type='integration', is_deleted=False)
        
        # Verify user owns the agent that owns this integration
        if integration.agent.user != request.user:
            logger.warning(f"Permission denied: User {request.user.id} attempted to delete integration {integration_id} without permission")
            return Response({
                'error': 'Permission denied'
            }, status=status.HTTP_403_FORBIDDEN)
        
        with transaction.atomic():
            integration.is_deleted = True
            integration.save(update_fields=['is_deleted', 'updated_at'])
            if integration.embedded:
                embedding.delete_embedding()

        return Response({
            'message': 'Integration deleted successfully'
        }, status=status.HTTP_204_NO_CONTENT)
    
    except AgentIntegrations.DoesNotExist:
        return Response({
            'error': 'Integration not found'
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({
            'error': 'Failed to delete integration',
            'detail': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['PUT'])
@permission_classes([IsAuthenticated])
def update_agent_integration(request, integration_id):
    """
    Update a specific agent integration.
    Returns 200 on success, 404 if integration not found.
    """
    try:
        integration = get_object_or_404(AgentIntegrations, integration_id=integration_id, is_deleted=False)
        
        # Verify user owns the agent that owns this integration
        if integration.agent.user != request.user:
            logger.warning(f"Permission denied: User {request.user.id} attempted to update integration {integration_id} without permission")
            return Response({
                'error': 'Permission denied',
                'detail': 'You do not have permission to update this integration'
            }, status=status.HTTP_403_FORBIDDEN)
        
        # Update fields if provided
        if 'integration_type' in request.data:
            integration.integration_type = request.data.get('integration_type')
        
        with transaction.atomic():
            integration.save(update_fields=['integration_type', 'updated_at'])
        
        return Response({
            'message': 'Integration updated successfully',
            'integration': {
                'integration_id': integration.integration_id,
                'integration_type': integration.integration_type,
                'created_at': integration.created_at.isoformat(),
                'updated_at': integration.updated_at.isoformat()
            }   
        }, status=status.HTTP_200_OK)
    
    except AgentIntegrations.DoesNotExist:
        return Response({
            'error': 'Integration not found'
        }, status=status.HTTP_404_NOT_FOUND)    
    
    except Exception as e:
        return Response({
            'error': 'Failed to update integration',
            'detail': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_agent_websites(request, agent_id):
    """
    Get all websites for a specific agent.
    Returns 200 on success, 404 if agent not found.
    Includes website metadata and content status.
    """
    try:
        # Get agent and verify ownership with select_related for optimization
        agent = get_object_or_404(Agent.objects.select_related('user'), 
                                 agent_id=agent_id, 
                                 user=request.user, 
                                 is_deleted=False)
        
        # Get websites with select_related to optimize query
        websites = (AgentWebsites.objects
                   .filter(agent=agent, is_deleted=False)
                   .select_related('agent')
                   .order_by('-created_at'))
        
        # Use list comprehension for better performance
        website_data = [{
            'website_id': website.website_id,
            'website_url': website.website_url,
            'content_status': 'available' if website.website_content else 'pending',
            'embedded': website.embedded,
            'created_at': website.created_at.isoformat(),
            'updated_at': website.updated_at.isoformat(),
            'agent_id': agent.agent_id,
            'agent_name': agent.name
        } for website in websites]
            
        return Response({
            'websites': website_data,
            'count': len(website_data),
            'agent_id': agent_id,
            'agent_name': agent.name
        }, status=status.HTTP_200_OK)
    
    except Agent.DoesNotExist:
        logger.warning(f"Agent {agent_id} not found for user {request.user.id}")
        return Response({
            'error': 'Agent not found',
            'detail': 'The requested agent does not exist or you do not have permission to access it'
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        logger.error(f"Error retrieving websites for agent {agent_id}: {str(e)}")
        return Response({
            'error': 'Failed to retrieve websites',
            'detail': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def create_agent_website(request, agent_id):
    """
    Create a new agent website.
    Returns 201 on success, 400 on validation error.
    Validates URL format and attempts to fetch content.
    """
    try:
        # Extract and validate required fields
        website_url = request.data.get('website_url', '').strip()
        website_name = request.data.get('website_name', '').strip()
        
        # Validate URL format
        if not website_url:
            return Response({
                'error': 'Website URL is required',
                'detail': 'Please provide a valid website URL'
            }, status=status.HTTP_400_BAD_REQUEST)
            
        # Validate URL format
        if not website_url.startswith(('http://', 'https://')):
            website_url = f'https://{website_url}'

        user_subscription = UserSubscription.objects.get(user=request.user, is_active=True, is_deleted=False)
        if not user_subscription.is_valid_subscription():
            return Response(
                {"error": "You have reached your subscription limit. Please upgrade your subscription to continue."},
                status=status.HTTP_403_FORBIDDEN
            )
        if user_subscription.tokens_usage >= user_subscription.price.plan.feature_limit("tokens"):
            return Response(
                {"error": "You have reached your subscription limit in training your agent. Please upgrade your subscription to continue."},
                status=status.HTTP_403_FORBIDDEN
            )
        
        try:
            from urllib.parse import urlparse
            parsed_url = urlparse(website_url)
            if not all([parsed_url.scheme, parsed_url.netloc]):
                raise ValueError("Invalid URL format")
        except ValueError as e:
            return Response({
                'error': 'Invalid URL format',
                'detail': str(e)
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Validate agent exists and belongs to user
        agent = get_object_or_404(Agent.objects.select_related('user'), 
                                 agent_id=agent_id, 
                                 user=request.user, 
                                 is_deleted=False)

        # Create the website with transaction
        with transaction.atomic():
            website = AgentWebsites.objects.create(
                agent=agent,
                website_url=website_url,
                website_name=website_name
            )
            logger.info(f"Website created: {website.website_id} - {website_url} for agent {agent_id}")
            website.website_content = website.get_website_content()
            website.save()

        
        website_data = {
            'website_id': website.website_id,
            'website_url': website.website_url,
            'website_name': website.website_name,
            'created_at': website.created_at.isoformat(),
            'updated_at': website.updated_at.isoformat(),
            'agent_id': agent.agent_id,
            'agent_name': agent.name
        }
        
        return Response({
            'message': 'Website created successfully',
            'website': website_data
        }, status=status.HTTP_201_CREATED)
    
    except Agent.DoesNotExist:
        logger.warning(f"Agent {agent_id} not found for user {request.user.id}")
        return Response({
            'error': 'Agent not found',
            'detail': 'The requested agent does not exist or you do not have permission to access it'
        }, status=status.HTTP_404_NOT_FOUND)    
    except Exception as e:
        logger.error(f"Error creating website for agent {agent_id}: {str(e)}")
        return Response({
            'error': 'Failed to create website',
            'detail': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['PUT'])
@permission_classes([IsAuthenticated])
def update_agent_website(request, website_id):
    """
    Update a specific agent website.
    Returns 200 on success, 404 if website not found.
    Validates URL format and updates content if URL changes.
    """
    try:
        # Get website with select_related for optimization
        website = get_object_or_404(AgentWebsites.objects.select_related('agent__user'), 
                                   website_id=website_id, 
                                   is_deleted=False)
        
        # Verify user owns the agent that owns this website
        if website.agent.user != request.user:
            logger.warning(f"Permission denied for user {request.user.id} on website {website_id}")
            return Response({
                'error': 'Permission denied',
                'detail': 'You do not have permission to update this website'
            }, status=status.HTTP_403_FORBIDDEN)
        
        # Update fields if provided
        website_url = request.data.get('website_url', '').strip()
        if website_url:
            # Validate URL format
            if not website_url.startswith(('http://', 'https://')):
                website_url = f'https://{website_url}'
                
            try:
                from urllib.parse import urlparse
                parsed_url = urlparse(website_url)
                if not all([parsed_url.scheme, parsed_url.netloc]):
                    raise ValueError("Invalid URL format")
            except ValueError as e:
                return Response({
                    'error': 'Invalid URL format',
                    'detail': str(e)
                }, status=status.HTTP_400_BAD_REQUEST)
                
            # If URL changed, attempt to fetch new content
            if website_url != website.website_url:
                try:
                    import requests
                    from requests.exceptions import RequestException
                    
                    response = requests.get(website_url, timeout=10)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.text, 'html.parser')
                    website_content = soup.get_text(separator=' ', strip=True)
                    
                    if not website_content:
                        logger.warning(f"No content extracted from website: {website_url}")
                        website_content = None
                        
                except RequestException as e:
                    logger.error(f"Failed to fetch website content: {str(e)}")
                    website_content = None
                    
                website.website_url = website_url
                website.website_content = website_content
                update_fields = ['website_url', 'website_content', 'updated_at']
            else:
                update_fields = ['updated_at']
        
        with transaction.atomic():
            website.save(update_fields=update_fields)
            logger.info(f"Website updated: {website.website_id} - {website.website_url}")
        
        return Response({
            'message': 'Website updated successfully',
            'website': {
                'website_id': website.website_id,
                'website_url': website.website_url,
                'content_status': 'available' if website.website_content else 'pending',
                'created_at': website.created_at.isoformat(),
                'updated_at': website.updated_at.isoformat(),
                'agent_id': website.agent.agent_id,
                'agent_name': website.agent.name
            }
        }, status=status.HTTP_200_OK)
    
    except AgentWebsites.DoesNotExist:
        logger.warning(f"Website {website_id} not found")
        return Response({
            'error': 'Website not found',
            'detail': 'The requested website does not exist'
        }, status=status.HTTP_404_NOT_FOUND)
    
    except Exception as e:
        logger.error(f"Error updating website {website_id}: {str(e)}")
        return Response({
            'error': 'Failed to update website',
            'detail': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)    
    
@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_agent_website(request,agent_id, website_id):
    """
    Delete a specific agent website.
    Returns 204 on success, 404 if website not found.
    """
    try:
        website = get_object_or_404(AgentWebsites, website_id=website_id, agent_id=agent_id, is_deleted=False)
        if website.embedded:
            embedding = get_object_or_404(AgentEmbeddings, agent=website.agent, object_id=website_id, object_type='website', is_deleted=False)
        
        # Verify user owns the agent that owns this website
        if website.agent.user != request.user:
            logger.warning(f"Permission denied: User {request.user.id} attempted to delete website {website_id} without permission")
            return Response({
                'error': 'Permission denied',
                'detail': 'You do not have permission to delete this website'
            }, status=status.HTTP_403_FORBIDDEN)
        
        with transaction.atomic():
            website.is_deleted = True
            website.save(update_fields=['is_deleted', 'updated_at'])
            if website.embedded:
                embedding.delete_embedding()
        
        return Response({
            'message': 'Website deleted successfully'
        }, status=status.HTTP_204_NO_CONTENT)
    
    except AgentWebsites.DoesNotExist:
        return Response({
            'error': 'Website not found'
        }, status=status.HTTP_404_NOT_FOUND)
    
    except Exception as e:
        return Response({
            'error': 'Failed to delete website',
            'detail': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def embed_agent_data(request, agent_id, object_type, object_id):
    """
    Embed agent data into vector database.
    
    Args:
        request: HTTP request object
        agent_id: ID of the agent
        object_type: Type of object to embed ('document', 'text', 'integration', 'website')
        object_id: ID of the object to embed
        
    Returns:
        Response with status:
        - 200: Success
        - 400: Invalid request or object already embedded
        - 404: Agent or object not found
        - 500: Server error
    """
    try:
        logger.info(f"Starting embed_agent_data for agent_id={agent_id}, object_type={object_type}, object_id={object_id}")
        logger.debug(f"Request data: {request.data}")
        
        # Validate request data
        if not request.data.get('object_name'):
            logger.warning(f"Missing object_name in request data for {object_type} {object_id}")
            return Response({
                'error': 'Missing required field: object_name'
            }, status=status.HTTP_400_BAD_REQUEST)

        # Get agent and verify ownership
        try:
            agent = Agent.objects.get(agent_id=agent_id, user=request.user, is_deleted=False)
            vdb = AgentVectorsDatabase.objects.get(agent=agent)
            user_subscription = UserSubscription.objects.get(user=request.user, is_active=True, is_deleted=False)
            if not user_subscription.is_valid_subscription():
                return Response(
                    {"error": "You have reached your subscription limit. Please upgrade your subscription to continue."},
                    status=status.HTTP_403_FORBIDDEN
                )
            if user_subscription.tokens_usage >= user_subscription.price.plan.feature_limit("tokens"):
                return Response(
                    {"error": "You have reached your subscription limit in training your agent. Please upgrade your subscription to continue."},
                    status=status.HTTP_403_FORBIDDEN
                )
            logger.info(f"Found agent {agent_id} and vector database")
        except (Agent.DoesNotExist, AgentVectorsDatabase.DoesNotExist):
            logger.warning(f"Agent {agent_id} or vector database not found")
            return Response({
                'error': 'Agent or vector database not found'
            }, status=status.HTTP_404_NOT_FOUND)

        # Check for existing embedding
        if AgentEmbeddings.objects.filter(
            agent=agent,
            object_id=object_id,
            object_type=object_type,
            is_deleted=False
        ).exists():
            logger.warning(f"Embedding already exists for {object_type} {object_id}")
            return Response({
                'error': f'{object_type.title()} already embedded for this agent'
            }, status=status.HTTP_400_BAD_REQUEST)

        # Define supported object types and their configurations
        object_models = {
            'document': {
                'model': AgentDocuments,
                'id_field': 'document_id',
                'url_field': 'document_path',
                'content_field': 'get_document_text',
                'chunk_size': 1000,  # Larger chunks for documents
                'chunk_overlap': 100
            },
            'text': {
                'model': AgentTexts,
                'id_field': 'text_id',
                'url_field': None,
                'content_field': 'text_content',
                'chunk_size': 500,
                'chunk_overlap': 50
            },
            'integration': {
                'model': AgentIntegrations,
                'id_field': 'integration_id',
                'url_field': None,
                'content_field': 'integration_content',
                'chunk_size': 200,
                'chunk_overlap': 20
            },
            'website': {
                'model': AgentWebsites,
                'id_field': 'website_id',
                'url_field': None,
                'content_field': 'website_content',
                'chunk_size': 300,
                'chunk_overlap': 30
            }
        }

        if object_type not in object_models:
            logger.warning(f"Invalid object type: {object_type}")
            return Response({
                'error': 'Invalid object type',
                'valid_types': list(object_models.keys())
            }, status=status.HTTP_400_BAD_REQUEST)
            
        # Check if object is already embedded
        if object_models[object_type]['model'].objects.filter(
            **{object_models[object_type]['id_field']: object_id},
            is_deleted=False,
            embedded=True
        ).exists():
            logger.warning(f"Object {object_type} {object_id} already has embedded=True")
            return Response({
                'error': f'{object_type.title()} already embedded for this agent'
            }, status=status.HTTP_400_BAD_REQUEST)

        # Get object configuration
        config = object_models[object_type]
        
        # Get object and verify it exists
        try:
            obj = config['model'].objects.get(
                **{config['id_field']: object_id},
                is_deleted=False
            )
            logger.info(f"Found {object_type} object {object_id}")
        except config['model'].DoesNotExist:
            logger.warning(f"{object_type.title()} {object_id} not found")
            return Response({
                'error': f'{object_type.title()} not found'
            }, status=status.HTTP_404_NOT_FOUND)

        # Initialize embedding model with type-specific settings
        embedding_model = create_embedding_model(
            api_key=os.environ.get("OPENAI_API_KEY"),
            chunk_size=config['chunk_size'],
            chunk_overlap=config['chunk_overlap']
        )

        # Initialize vector database
        pinecone_vdb = PineconeVDB(
            api_key=os.environ.get("PINECONE_API_KEY"),
            index_name=vdb.database_index,
            namespace=vdb.namespace,
            metric="cosine",
            dimension=1536
        )

        # Get content and create embedding
        content = getattr(obj, config['content_field'])
        if callable(content):
            content = content()
            
        # Check if content is None or empty after getting it
        if content is None or (isinstance(content, str) and not content.strip()):
            logger.error(f"No content available for {object_type} {object_id}")
            return Response({
                'error': f'No content available for {object_type}. Please ensure the content has been properly loaded.'
            }, status=status.HTTP_400_BAD_REQUEST)
            
        logger.info(f"Extracted content of length: {len(content) if content else 0}")
            
        # Split content into chunks
        content_list = []
        if isinstance(content, list):
            for text in content:
                if text and text.strip():  # Only process non-empty text
                    content_list.extend(embedding_model.split_texts(text=text))
        else:
            content_list.extend(embedding_model.split_texts(text=content))
            
        logger.info(f"Split content into {len(content_list)} chunks")
        
        if len(content_list) == 0:
            logger.error(f"No content to embed for {object_type} {object_id}")
            return Response({
                'error': 'No content to embed'
            }, status=status.HTTP_400_BAD_REQUEST)

        encoding = tiktoken.encoding_for_model("text-embedding-3-small")
        num_tokens = len(encoding.encode(content))
        logger.info(f"Number of tokens: {num_tokens}")
        if num_tokens > 8192:
            logger.error(f"Number of tokens exceeds the limit of 8192 for {object_type} {object_id}")
            return Response({
                'error': 'Number of tokens exceeds the limit of 8192'
            }, status=status.HTTP_400_BAD_REQUEST)
        total_tokens = sum(len(encoding.encode(chunk)) for chunk in content_list)
        logger.info(f"Total number of tokens: {total_tokens}")
        if total_tokens > 8192:
            logger.error(f"Total number of tokens exceeds the limit of 8192 for {object_type} {object_id}")
            return Response({
                'error': 'Total number of tokens exceeds the limit of 8192'
            }, status=status.HTTP_400_BAD_REQUEST)
        cost_per_1k_tokens = 0.02
        cost = (total_tokens / 1000) * cost_per_1k_tokens
        logger.info(f"Cost: {cost}")
        user_subscription.tokens_usage += total_tokens
        user_subscription.save(update_fields=['tokens_usage'])
        num_chunks = len(content_list)

        # Create embedding record data
        embedding_data = {
            'agent': agent,
            'embedding_model': "openai-ada",
            'embedding_model_version': "text-embedding-3-small",
            'vector_dimension': 1536,
            'object_id': object_id,
            'object_type': object_type,
            'object_name': request.data['object_name'],
            'num_chunks': num_chunks
        }
        
        # Add source URL if applicable
        if config['url_field']:
            embedding_data['source_url'] = getattr(obj, config['url_field'])
        # Create and store embedding record
        with transaction.atomic():
            embedding = AgentEmbeddings.objects.create(**embedding_data)

        # Generate embeddings for all content chunks
        embeddings, _ = embedding_model.get_embeddings(content_list)

        # Store each chunk with its embedding in vector database
        vectors_data = []
        for i, text in enumerate(content_list):
            vectors_data.append({
                "id": f"{embedding.embedding_id}-{i}",  # Simplified ID format
                "values": embeddings[i],
                "metadata": {
                    "text": text,
                    "object_id": embedding.object_id,
                    "object_type": embedding.object_type,
                    "object_name": embedding.object_name,
                    "agent_id": embedding.agent.agent_id,
                    "created_at": embedding.created_at.isoformat()
                }
            })  
            # Store in vector database
        pinecone_vdb.upsert_vectors(embedding_data=vectors_data, namespace=vdb.namespace)

        obj.embedded = True
        obj.save(update_fields=['embedded'])      

        logger.info(f"Successfully embedded {object_type} {object_id} with {num_chunks} chunks")
        return Response({
            'message': f'{object_type.title()} embedded successfully',
            'embedding_id': embedding.embedding_id,
            'object_name': request.data['object_name'],
            'created_at': embedding.created_at
        }, status=status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"Error embedding {object_type} {object_id}: {str(e)}")
        return Response({
            'error': 'Failed to embed object',
            'detail': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
  

@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_agent_embedding(request, agent_id, object_type, object_id):
    """
    Delete a specific agent embedding.
    Returns 204 on success, 404 if embedding not found.
    """
    try:
        agent = get_object_or_404(Agent, agent_id=agent_id, user=request.user, is_deleted=False)
        embedding = get_object_or_404(AgentEmbeddings, agent=agent, object_id=object_id, object_type=object_type, is_deleted=False)
        vdb = get_object_or_404(AgentVectorsDatabase, agent=agent, is_deleted=False)
        
        with transaction.atomic():
            embedding.is_deleted = True
            embedding.save(update_fields=['is_deleted', 'updated_at'])

        pinecone_vdb = PineconeVDB(
            index_name=vdb.database_index,
            namespace=vdb.namespace,
            metric="cosine",
            dimension=1536
        )
        for i in range(embedding.num_chunks):
            pinecone_vdb.delete_vectors(ids=[f"{embedding.embedding_id}-{i}"], namespace=vdb.namespace)
        embedding.delete()
        
        return Response({
            'message': 'Embedding deleted successfully'
        }, status=status.HTTP_204_NO_CONTENT)
    except Exception as e:
        return Response({
            'error': 'Failed to delete embedding',
            'detail': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def create_agent_integration_frontend(request, agent_id):
    """
    Create a new website integration.
    Returns 201 on success, 400 on validation error.
    """
    try:
        # Validate agent ownership
        agent = get_object_or_404(Agent, agent_id=agent_id, user=request.user, is_deleted=False)
        user_subscription = UserSubscription.objects.get(user=request.user, is_active=True, is_deleted=False)
        if not user_subscription.is_valid_subscription():
            return Response(
                {"error": "You have reached your subscription limit. Please upgrade your subscription to continue."},
                status=status.HTTP_403_FORBIDDEN
            )
        if user_subscription.widgets_usage >= user_subscription.price.plan.feature_limit("website_integration"):
            return Response(
                {"error": "You have reached your subscription limit in creating website integrations. Please upgrade your subscription to continue."},
                status=status.HTTP_403_FORBIDDEN
            )
        
        # Extract and validate required data
        data = request.data
        website_name = data.get('website_name', '').strip()
        website_domain = data.get('website_domain', '').strip()
        website_logo = request.FILES.get('website_logo')
        start_message = data.get('start_message', '').strip()
        
        # Validate required fields
        if not website_name:
            return Response({
                'error': 'Website name is required and cannot be empty'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        if not website_domain:
            return Response({
                'error': 'Website domain is required and cannot be empty'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Validate domain format (basic validation)
        if not website_domain.replace('.', '').replace('-', '').isalnum():
            return Response({
                'error': 'Invalid website domain format'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Check for duplicate integrations
        existing_integration = AgentIntegrationsFrontend.objects.filter(
            agent=agent,
            website_domain=website_domain,
            is_deleted=False
        ).first()
        
        if existing_integration:
            return Response({
                'error': f'Integration for domain "{website_domain}" already exists'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Generate secure API key
        api_key = secrets.token_hex(32)
        
        # Create integration with transaction for data consistency
        with transaction.atomic():
            integration = AgentIntegrationsFrontend.objects.create(
                agent=agent,
                website_name=website_name,
                website_domain=website_domain,
                website_logo=website_logo,
                api_key=api_key,
                start_message=start_message
            )
            user_subscription.widgets_usage += 1
            user_subscription.save(update_fields=['widgets_usage'])
        
        # Prepare response data
        response_data = {
            'message': 'Website integration created successfully',
            'integration_id': integration.integration_id,
            'api_key': api_key,
            'website_name': website_name,
            'website_domain': website_domain,
            'start_message': start_message,
            'created_at': integration.created_at
        }
        
        # Add logo URL if provided
        if website_logo:
            response_data['website_logo'] = website_logo.url if hasattr(website_logo, 'url') else str(website_logo)
        
        logger.info(f"Website integration created successfully for agent {agent_id} by user {request.user.id}")
        
        return Response(response_data, status=status.HTTP_201_CREATED)
        
    except Http404:
        logger.warning(f"Agent {agent_id} not found for user {request.user.id}")
        return Response({
            'error': 'Agent not found'
        }, status=status.HTTP_404_NOT_FOUND)
        
    except ValidationError as e:
        logger.warning(f"Validation error creating integration: {str(e)}")
        return Response({
            'error': 'Invalid data provided',
            'details': str(e)
        }, status=status.HTTP_400_BAD_REQUEST)
        
    except IntegrityError as e:
        logger.error(f"Database integrity error creating integration: {str(e)}")
        return Response({
            'error': 'Database constraint violation'
        }, status=status.HTTP_400_BAD_REQUEST)
        
    except Exception as e:
        logger.error(f"Unexpected error creating integration: {str(e)}", exc_info=True)
        return Response({
            'error': 'An unexpected error occurred while creating the integration'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_agent_integrations_frontend(request, agent_id):
    """
    Get all integrations for an agent.
    Returns 200 on success, 404 if agent not found.
    """
    try:
        # Validate agent ownership
        agent = get_object_or_404(Agent, agent_id=agent_id, user=request.user, is_deleted=False)
        
        # Get integrations with optimized query
        integrations = AgentIntegrationsFrontend.objects.filter(
            agent=agent, 
            is_deleted=False
        )
        # Build response data efficiently
        data = [
            {
                'integration_id': integration.integration_id,
                'integration_name': integration.integration_name,
                'integration_description': integration.integration_description,
                'api_key': integration.api_key,
                'website_name': integration.website_name,
                'website_domain': integration.website_domain,
                'website_logo': integration.website_logo.url if integration.website_logo else None,
                'start_message': integration.start_message,
                'created_at': integration.created_at.isoformat() if integration.created_at else None,
            }
            for integration in integrations
        ]
        
        logger.info(f"Integrations fetched successfully for agent {agent_id} by user {request.user.id}")
        
        return Response({
            'message': 'Integrations fetched successfully',
            'integrations': data,
            'count': len(data)
        }, status=status.HTTP_200_OK)
        
    except Http404:
        logger.warning(f"Agent {agent_id} not found for user {request.user.id}")
        return Response({
            'error': 'Agent not found'
        }, status=status.HTTP_404_NOT_FOUND)
        
    except Exception as e:
        logger.error(f"Unexpected error fetching integrations for agent {agent_id}: {str(e)}", exc_info=True)
        return Response({
            'error': 'An unexpected error occurred while fetching integrations'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['PUT', 'PATCH'])
@permission_classes([IsAuthenticated])
def update_agent_integration_frontend(request, agent_id, integration_id):
    """
    Update a specific agent integration.
    Returns 200 on success, 404 if integration not found, 400 for validation errors.
    """
    try:
        # Validate agent ownership
        agent = get_object_or_404(Agent, agent_id=agent_id, user=request.user, is_deleted=False)
        
        # Get integration and validate ownership
        integration = get_object_or_404(
            AgentIntegrationsFrontend, 
            agent=agent, 
            integration_id=integration_id, 
            is_deleted=False
        )
        
        # Validate request data
        if not request.data:
            return Response({
                'error': 'Request data is required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Extract and validate fields with improved validation
        data = request.data
        website_name = data.get('website_name', '').strip() if data.get('website_name') else None
        website_domain = data.get('website_domain', '').strip() if data.get('website_domain') else None
        website_logo = request.FILES.get('website_logo')
        start_message = data.get('start_message', '').strip() if data.get('start_message') else None
        deactivate = data.get('deactivate', False)
        
        # Validate domain format if provided
        if website_domain and not website_domain.startswith(('http://', 'https://')):
            website_domain = f'https://{website_domain}'
        
        # Build update fields list efficiently
        update_fields = []
        field_updates = {}
        
        if website_name is not None:
            field_updates['website_name'] = website_name
            update_fields.append('website_name')
            
        if website_domain is not None:
            field_updates['website_domain'] = website_domain
            update_fields.append('website_domain')
            
        if website_logo is not None:
            field_updates['website_logo'] = website_logo
            update_fields.append('website_logo')
            
        if start_message is not None:
            field_updates['start_message'] = start_message
            update_fields.append('start_message')
            
        if deactivate:
            field_updates['is_active'] = False
            update_fields.append('is_active')
        
        if not update_fields:
            return Response({
                'error': 'No valid fields to update'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Update integration with transaction
        with transaction.atomic():
            for field, value in field_updates.items():
                setattr(integration, field, value)
            
            integration.updated_at = timezone.now()
            update_fields.append('updated_at')
            integration.save(update_fields=update_fields)
        
        logger.info(f"Integration {integration_id} updated successfully for agent {agent_id} by user {request.user.id}")
        
        # Return updated data with consistent formatting
        return Response({
            'message': 'Integration updated successfully',
            'integration_id': integration.integration_id,
            'website_name': integration.website_name,
            'website_domain': integration.website_domain,
            'website_logo': integration.website_logo.url if integration.website_logo else None,
            'start_message': integration.start_message,
            'is_active': integration.is_active,
            'updated_at': integration.updated_at.isoformat() if integration.updated_at else None
        }, status=status.HTTP_200_OK)
        
    except Http404:
        logger.warning(f"Integration {integration_id} not found for agent {agent_id} by user {request.user.id}")
        return Response({
            'error': 'Integration not found'
        }, status=status.HTTP_404_NOT_FOUND)
        
    except ValidationError as e:
        logger.warning(f"Validation error updating integration {integration_id}: {str(e)}")
        return Response({
            'error': 'Invalid data provided',
            'details': str(e)
        }, status=status.HTTP_400_BAD_REQUEST)
        
    except IntegrityError as e:
        logger.error(f"Database integrity error updating integration {integration_id}: {str(e)}")
        return Response({
            'error': 'Database constraint violation'
        }, status=status.HTTP_400_BAD_REQUEST)
        
    except Exception as e:
        logger.error(f"Unexpected error updating integration {integration_id}: {str(e)}", exc_info=True)
        return Response({
            'error': 'An unexpected error occurred while updating the integration'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_agent_integration_frontend(request, agent_id, integration_id):
    """
    Delete a specific agent integration.
    Returns 204 on success, 404 if integration not found.
    """
    try:
        # Validate agent ownership
        agent = get_object_or_404(Agent, agent_id=agent_id, user=request.user, is_deleted=False)
        
        # Get integration and validate ownership
        integration = get_object_or_404(
            AgentIntegrationsFrontend, 
            agent=agent, 
            integration_id=integration_id, 
            is_deleted=False
        )
        
        # Soft delete integration with transaction
        with transaction.atomic():
            integration.is_deleted = True
            integration.is_active = False  # Also deactivate for consistency
            integration.updated_at = timezone.now()
            integration.save(update_fields=['is_deleted', 'is_active', 'updated_at'])
            
        logger.info(f"Integration {integration_id} deleted successfully for agent {agent_id} by user {request.user.id}")
        
        return Response({
            'message': 'Integration deleted successfully'   
        }, status=status.HTTP_204_NO_CONTENT)
        
    except Http404:
        logger.warning(f"Integration {integration_id} not found for agent {agent_id} by user {request.user.id}")
        return Response({
            'error': 'Integration not found'
        }, status=status.HTTP_404_NOT_FOUND)
        
    except Exception as e:
        logger.error(f"Unexpected error deleting integration {integration_id}: {str(e)}", exc_info=True)    
        return Response({ 
            'error': 'An unexpected error occurred while deleting the integration'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)