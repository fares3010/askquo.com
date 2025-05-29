from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework import status
from django.shortcuts import get_object_or_404
from django.db import transaction
from .models import Agent, AgentDocuments, AgentTexts, AgentQaPairs, AgentIntegrations, AgentEmbeddings, AgentVectorsDatabase, AgentWebsites

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def agent_list(request):
    """
    List all agents for the current user.
    Returns a list of active (non-deleted) agents.
    """
    try:
        agents = Agent.objects.filter(user=request.user, is_deleted=False).order_by('-created_at')
        
        agent_data = []
        for agent in agents:
            agent_data.append({
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
                'is_active': agent.is_active(),
                'conversation_count': agent.conversation_count(),
                'documents_summary': agent.get_documents_summary()
            })
        
        return Response({
            'agents': agent_data,
            'count': len(agent_data)
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({
            'error': 'Failed to retrieve agents',
            'detail': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def agent_detail(request, agent_id):
    """
    Get a single agent's details.
    Returns 404 if agent not found or doesn't belong to user.
    """
    try:
        agent = get_object_or_404(Agent, agent_id=agent_id, user=request.user, is_deleted=False)
        
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
            'is_active': agent.is_active(),
            'conversation_count': agent.conversation_count() if hasattr(agent, 'conversation_count') else 0,
            'documents_summary': agent.get_documents_summary() if hasattr(agent, 'get_documents_summary') else {}
        }
        
        return Response(agent_data, status=status.HTTP_200_OK)
        
    except Agent.DoesNotExist:
        return Response({
            'error': 'Agent not found'
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({
            'error': 'Failed to retrieve agent',
            'detail': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def create_agent(request):
    """
    Create a new agent.
    Returns 201 on success, 400 on validation error.
    """
    try:
        # Extract and validate required fields
        name = request.data.get('name', '').strip()
        if not name:
            return Response({
                'error': 'Agent name is required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Check for duplicate agent names for this user
        if Agent.objects.filter(user=request.user, name=name, is_deleted=False).exists():
            return Response({
                'error': 'An agent with this name already exists'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Extract optional fields with defaults
        description = request.data.get('description', '')
        visibility = request.data.get('visibility', 'private')
        avatar_url = request.data.get('avatar_url', '')
        configuration = request.data.get('configuration', {})
        
        # Validate visibility choice
        if visibility not in ['public', 'private']:
            visibility = 'private'
        
        # Create the agent with transaction for data integrity
        with transaction.atomic():
            agent = Agent.objects.create(
                user=request.user,
                name=name,
                description=description,
                visibility=visibility,
                avatar_url=avatar_url if avatar_url else None,
                configuration=configuration if configuration else None
            )
        
        agent_data = {
            'agent_id': agent.agent_id,
            'name': agent.name,
            'description': agent.description,
            'created_at': agent.created_at.isoformat(),
            'updated_at': agent.updated_at.isoformat(),
            'is_archived': agent.is_archived,
            'is_favorite': agent.is_favorite,
            'visibility': agent.visibility,
            'avatar_url': agent.avatar_url,
            'configuration': agent.configuration
        }
        
        return Response({
            'message': 'Agent created successfully',
            'agent': agent_data
        }, status=status.HTTP_201_CREATED)
        
    except Exception as e:
        return Response({
            'error': 'Failed to create agent',
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
        document_url = request.data.get('document_url')
        
        # Validate required fields
        required_fields = {
            'document_name': document_name,
            'document_format': document_format,
            'document_url': document_url
        }
        
        for field_name, field_value in required_fields.items():
            if not field_value:
                return Response({
                    'error': f'{field_name.replace("_", " ").title()} is required'
                }, status=status.HTTP_400_BAD_REQUEST)
        
        # Extract optional fields
        document_size = request.data.get('document_size')
        
        # Validate agent exists and belongs to user
        agent = get_object_or_404(Agent, agent_id=agent_id, user=request.user, is_deleted=False)
        
        # Create the document with transaction
        with transaction.atomic():
            document = AgentDocuments.objects.create(
                agent=agent,
                document_name=document_name,
                document_format=document_format,
                document_url=document_url,
                document_size=document_size
            )
        
        document_data = {
            'document_id': document.document_id,
            'document_name': document.document_name,
            'document_format': document.document_format,
            'document_url': document.document_url,
            'document_size': document.document_size,
            'created_at': document.created_at.isoformat(),
            'updated_at': document.updated_at.isoformat()
        }
        
        return Response({
            'message': 'Document created successfully',
            'document': document_data
        }, status=status.HTTP_201_CREATED)
    
    except Agent.DoesNotExist:
        return Response({
            'error': 'Agent not found'
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
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
    """
    try:
        agent = get_object_or_404(Agent, agent_id=agent_id, user=request.user, is_deleted=False)
        
        documents = AgentDocuments.objects.filter(agent=agent, is_deleted=False).order_by('-created_at')
        
        document_data = []
        for document in documents:
            document_data.append({
                'document_id': document.document_id,
                'document_name': document.document_name,
                'document_format': document.document_format,
                'document_url': document.document_url,
                'document_size': document.document_size,
                'created_at': document.created_at.isoformat(),
                'updated_at': document.updated_at.isoformat()
            })
            
        return Response({
            'documents': document_data,
            'count': len(document_data)
        }, status=status.HTTP_200_OK)
    
    except Agent.DoesNotExist:
        return Response({
            'error': 'Agent not found'
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({
            'error': 'Failed to retrieve documents',
            'detail': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_agent_document(request, document_id):
    """
    Delete a specific agent document.
    Returns 204 on success, 404 if document not found.  
    """
    try:
        document = get_object_or_404(AgentDocuments, document_id=document_id, is_deleted=False)
        
        # Verify user owns the agent that owns this document
        if document.agent.user != request.user:
            return Response({
                'error': 'Permission denied'
            }, status=status.HTTP_403_FORBIDDEN)
        
        with transaction.atomic():
            document.is_deleted = True
            document.save(update_fields=['is_deleted', 'updated_at'])
        
        return Response({
            'message': 'Document deleted successfully'
        }, status=status.HTTP_204_NO_CONTENT)
    
    except AgentDocuments.DoesNotExist: 
        return Response({
            'error': 'Document not found'
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({
            'error': 'Failed to delete document',
            'detail': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['PUT'])  
@permission_classes([IsAuthenticated])
def update_agent_document(request, document_id):
    """
    Update a specific agent document.
    Returns 200 on success, 404 if document not found.
    """
    try:
        document = get_object_or_404(AgentDocuments, document_id=document_id, is_deleted=False)
        
        # Verify user owns the agent that owns this document
        if document.agent.user != request.user:
            return Response({
                'error': 'Permission denied'
            }, status=status.HTTP_403_FORBIDDEN)

        # Track which fields to update
        update_fields = ['updated_at']
        
        # Update fields if provided
        if 'document_name' in request.data:
            document.document_name = request.data.get('document_name')
            update_fields.append('document_name')
        
        if 'document_format' in request.data:
            document.document_format = request.data.get('document_format')
            update_fields.append('document_format')
        
        if 'document_url' in request.data:
            document.document_url = request.data.get('document_url')
            update_fields.append('document_url')
        
        if 'document_size' in request.data:
            document.document_size = request.data.get('document_size')
            update_fields.append('document_size')
        
        with transaction.atomic():
            document.save(update_fields=update_fields)
        
        return Response({
            'message': 'Document updated successfully',
            'document': {
                'document_id': document.document_id,
                'document_name': document.document_name,
                'document_format': document.document_format,    
                'document_url': document.document_url,
                'document_size': document.document_size,
                'created_at': document.created_at.isoformat(),
                'updated_at': document.updated_at.isoformat()
            }
        }, status=status.HTTP_200_OK)       
    
    except AgentDocuments.DoesNotExist:
        return Response({
            'error': 'Document not found'
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
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
        
        if not text_content:
            return Response({
                'error': 'Text content is required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Validate agent exists and belongs to user
        agent = get_object_or_404(Agent, agent_id=agent_id, user=request.user, is_deleted=False)
        
        # Create the text with transaction
        with transaction.atomic():
            text = AgentTexts.objects.create(
                agent=agent,
                text_content=text_content
            )
        
        text_data = {
            'text_id': text.text_id,
            'text_content': text.text_content,
            'created_at': text.created_at.isoformat(),
            'updated_at': text.updated_at.isoformat()
        }
        
        return Response({
            'message': 'Text created successfully',
            'text': text_data
        }, status=status.HTTP_201_CREATED)
    
    except Agent.DoesNotExist:
        return Response({
            'error': 'Agent not found'
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
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
    """
    try:
        agent = get_object_or_404(Agent, agent_id=agent_id, user=request.user, is_deleted=False)
        
        texts = AgentTexts.objects.filter(agent=agent, is_deleted=False).order_by('-created_at')
        
        text_data = []
        for text in texts:
            text_data.append({
                'text_id': text.text_id,
                'text_content': text.text_content,
                'created_at': text.created_at.isoformat(),
                'updated_at': text.updated_at.isoformat()
            })
            
        return Response({
            'texts': text_data,
            'count': len(text_data)
        }, status=status.HTTP_200_OK)
    
    except Agent.DoesNotExist:
        return Response({
            'error': 'Agent not found'
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
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
        
        # Verify user owns the agent that owns this text
        if text.agent.user != request.user:
            return Response({
                'error': 'Permission denied'
            }, status=status.HTTP_403_FORBIDDEN)
        
        with transaction.atomic():
            text.is_deleted = True
            text.save(update_fields=['is_deleted', 'updated_at'])
        
        return Response({
            'message': 'Text deleted successfully'
        }, status=status.HTTP_204_NO_CONTENT)
    
    except AgentTexts.DoesNotExist:
        return Response({
            'error': 'Text not found'
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
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
            return Response({
                'error': 'Permission denied'
            }, status=status.HTTP_403_FORBIDDEN)
        
        # Update fields if provided
        if 'text_content' in request.data:
            text.text_content = request.data.get('text_content')    
        
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
def create_agent_qa_pair(request, agent_id):
    """
    Create a new agent QA pair.
    Returns 201 on success, 400 on validation error.
    """ 
    try:
        # Extract and validate required fields
        question = request.data.get('question')
        answer = request.data.get('answer')
        
        required_fields = {
            'question': question,
            'answer': answer
        }
        
        for field_name, field_value in required_fields.items():
            if not field_value:
                return Response({
                    'error': f'{field_name.replace("_", " ").title()} is required'
                }, status=status.HTTP_400_BAD_REQUEST)
        
        # Validate agent exists and belongs to user
        agent = get_object_or_404(Agent, agent_id=agent_id, user=request.user, is_deleted=False)
        
        # Create the QA pair with transaction
        with transaction.atomic():
            qa_pair = AgentQaPairs.objects.create(
                agent=agent,
                question=question,
                answer=answer
            )
        
        qa_pair_data = {
            'qa_pair_id': qa_pair.qa_pair_id,
            'question': qa_pair.question,
            'answer': qa_pair.answer,
            'created_at': qa_pair.created_at.isoformat(),
            'updated_at': qa_pair.updated_at.isoformat()
        }
        
        return Response({
            'message': 'QA pair created successfully',
            'qa_pair': qa_pair_data
        }, status=status.HTTP_201_CREATED)
    
    except Agent.DoesNotExist:
        return Response({
            'error': 'Agent not found'
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({
            'error': 'Failed to create QA pair',
            'detail': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_agent_qa_pairs(request, agent_id):
    """
    Get all QA pairs for a specific agent.
    Returns 200 on success, 404 if agent not found.
    """
    try:
        agent = get_object_or_404(Agent, agent_id=agent_id, user=request.user, is_deleted=False)
        
        qa_pairs = AgentQaPairs.objects.filter(agent=agent, is_deleted=False).order_by('-created_at')
        
        qa_pair_data = []
        for qa_pair in qa_pairs:
            qa_pair_data.append({
                'qa_pair_id': qa_pair.qa_pair_id,
                'question': qa_pair.question,
                'answer': qa_pair.answer,
                'created_at': qa_pair.created_at.isoformat(),
                'updated_at': qa_pair.updated_at.isoformat()
            })
            
        return Response({
            'qa_pairs': qa_pair_data,
            'count': len(qa_pair_data)
        }, status=status.HTTP_200_OK)

    except Agent.DoesNotExist:
        return Response({
            'error': 'Agent not found'
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({
            'error': 'Failed to retrieve QA pairs',
            'detail': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
@api_view(['DELETE'])   
@permission_classes([IsAuthenticated])
def delete_agent_qa_pair(request, qa_pair_id):
    """
    Delete a specific agent QA pair.
    Returns 204 on success, 404 if QA pair not found.
    """
    try:
        qa_pair = get_object_or_404(AgentQaPairs, qa_pair_id=qa_pair_id, is_deleted=False)
        
        # Verify user owns the agent that owns this QA pair
        if qa_pair.agent.user != request.user:
            return Response({
                'error': 'Permission denied'
            }, status=status.HTTP_403_FORBIDDEN)
        
        with transaction.atomic():
            qa_pair.is_deleted = True
            qa_pair.save(update_fields=['is_deleted', 'updated_at'])
        
        return Response({
            'message': 'QA pair deleted successfully'
        }, status=status.HTTP_204_NO_CONTENT)
    
    except AgentQaPairs.DoesNotExist:
        return Response({
            'error': 'QA pair not found'
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({
            'error': 'Failed to delete QA pair',
            'detail': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
@api_view(['PUT'])
@permission_classes([IsAuthenticated])
def update_agent_qa_pair(request, qa_pair_id):
    """
    Update a specific agent QA pair.
    Returns 200 on success, 404 if QA pair not found.
    """
    try:
        qa_pair = get_object_or_404(AgentQaPairs, qa_pair_id=qa_pair_id, is_deleted=False)
        
        # Verify user owns the agent that owns this QA pair
        if qa_pair.agent.user != request.user:
            return Response({
                'error': 'Permission denied'
            }, status=status.HTTP_403_FORBIDDEN)
        
        # Track which fields to update
        update_fields = ['updated_at']
        
        # Update fields if provided
        if 'question' in request.data:
            qa_pair.question = request.data.get('question')
            update_fields.append('question')
        
        if 'answer' in request.data:
            qa_pair.answer = request.data.get('answer')
            update_fields.append('answer')
            
        with transaction.atomic():
            qa_pair.save(update_fields=update_fields)
        
        return Response({
            'message': 'QA pair updated successfully',
            'qa_pair': {
                'qa_pair_id': qa_pair.qa_pair_id,
                'question': qa_pair.question,
                'answer': qa_pair.answer,
                'created_at': qa_pair.created_at.isoformat(),
                'updated_at': qa_pair.updated_at.isoformat()
            }
        }, status=status.HTTP_200_OK)
    
    except AgentQaPairs.DoesNotExist:
        return Response({
            'error': 'QA pair not found'
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({
            'error': 'Failed to update QA pair',
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
                'integration_type': integration.integration_type,
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
        
        # Verify user owns the agent that owns this integration
        if integration.agent.user != request.user:
            return Response({
                'error': 'Permission denied'
            }, status=status.HTTP_403_FORBIDDEN)
        
        with transaction.atomic():
            integration.is_deleted = True
            integration.save(update_fields=['is_deleted', 'updated_at'])
        
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
            return Response({
                'error': 'Permission denied'
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
    """
    try:
        agent = get_object_or_404(Agent, agent_id=agent_id, user=request.user, is_deleted=False)
        
        websites = AgentWebsites.objects.filter(agent=agent, is_deleted=False).order_by('-created_at')
        
        website_data = []
        for website in websites:
            website_data.append({
                'website_id': website.website_id,
                'website_url': website.website_url,
                'created_at': website.created_at.isoformat(),
                'updated_at': website.updated_at.isoformat()
            })
            
        return Response({
            'websites': website_data,
            'count': len(website_data)
        }, status=status.HTTP_200_OK)
    
    except Agent.DoesNotExist:
        return Response({
            'error': 'Agent not found'
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
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
    """
    try:
        # Extract and validate required fields
        website_url = request.data.get('website_url')
        
        if not website_url:
            return Response({
                'error': 'Website URL is required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Validate agent exists and belongs to user
        agent = get_object_or_404(Agent, agent_id=agent_id, user=request.user, is_deleted=False)
        
        # Create the website with transaction
        with transaction.atomic():
            website = AgentWebsites.objects.create(
                agent=agent,
                website_url=website_url
            )
        
        website_data = {
            'website_id': website.website_id,
            'website_url': website.website_url,
            'created_at': website.created_at.isoformat(),
            'updated_at': website.updated_at.isoformat()
        }
        
        return Response({
            'message': 'Website created successfully',
            'website': website_data
        }, status=status.HTTP_201_CREATED)
    
    except Agent.DoesNotExist:
        return Response({
            'error': 'Agent not found'
        }, status=status.HTTP_404_NOT_FOUND)    
    except Exception as e:
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
    """
    try:
        website = get_object_or_404(AgentWebsites, website_id=website_id, is_deleted=False)
        
        # Verify user owns the agent that owns this website
        if website.agent.user != request.user:
            return Response({
                'error': 'Permission denied'
            }, status=status.HTTP_403_FORBIDDEN)
        
        # Update fields if provided
        if 'website_url' in request.data:
            website.website_url = request.data.get('website_url')   
        
        with transaction.atomic():
            website.save(update_fields=['website_url', 'updated_at'])
        
        return Response({
            'message': 'Website updated successfully',
            'website': {
                'website_id': website.website_id,
                'website_url': website.website_url,
                'created_at': website.created_at.isoformat(),
                'updated_at': website.updated_at.isoformat()
            }
        }, status=status.HTTP_200_OK)
    
    except AgentWebsites.DoesNotExist:
        return Response({
            'error': 'Website not found'
        }, status=status.HTTP_404_NOT_FOUND)
    
    except Exception as e:
        return Response({
            'error': 'Failed to update website',
            'detail': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)    
    
@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_agent_website(request, website_id):
    """
    Delete a specific agent website.
    Returns 204 on success, 404 if website not found.
    """
    try:
        website = get_object_or_404(AgentWebsites, website_id=website_id, is_deleted=False)
        
        # Verify user owns the agent that owns this website
        if website.agent.user != request.user:
            return Response({
                'error': 'Permission denied'
            }, status=status.HTTP_403_FORBIDDEN)
        
        with transaction.atomic():
            website.is_deleted = True
            website.save(update_fields=['is_deleted', 'updated_at'])
        
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
    

@api_view(['PUT'])
@permission_classes([IsAuthenticated])
def update_agent(request, agent_id):
    """
    Update an existing agent.
    Returns 200 on success, 404 if not found, 400 on validation error.
    """
    try:
        agent = get_object_or_404(Agent, agent_id=agent_id, user=request.user, is_deleted=False)
        
        # Track which fields to update
        update_fields = ['updated_at']
        
        # Update fields if provided
        name = request.data.get('name', '').strip()
        if name:
            # Check for duplicate names (excluding current agent)
            if Agent.objects.filter(
                user=request.user, 
                name=name, 
                is_deleted=False
            ).exclude(agent_id=agent_id).exists():
                return Response({
                    'error': 'An agent with this name already exists'
                }, status=status.HTTP_400_BAD_REQUEST)
            agent.name = name
            update_fields.append('name')
        
        if 'description' in request.data:
            agent.description = request.data.get('description', '')
            update_fields.append('description')
        
        if 'visibility' in request.data:
            visibility = request.data.get('visibility')
            if visibility in ['public', 'private']:
                agent.visibility = visibility
                update_fields.append('visibility')
        
        if 'avatar_url' in request.data:
            agent.avatar_url = request.data.get('avatar_url') or None
            update_fields.append('avatar_url')
        
        if 'configuration' in request.data:
            agent.configuration = request.data.get('configuration') or None
            update_fields.append('configuration')
        
        if 'is_archived' in request.data:
            agent.is_archived = bool(request.data.get('is_archived'))
            update_fields.append('is_archived')
        
        if 'is_favorite' in request.data:
            agent.is_favorite = bool(request.data.get('is_favorite'))
            update_fields.append('is_favorite')
        
        with transaction.atomic():
            agent.save(update_fields=update_fields)
        
        agent_data = {
            'agent_id': agent.agent_id,
            'name': agent.name,
            'description': agent.description,
            'created_at': agent.created_at.isoformat(),
            'updated_at': agent.updated_at.isoformat(),
            'is_archived': agent.is_archived,
            'is_favorite': agent.is_favorite,
            'visibility': agent.visibility,
            'avatar_url': agent.avatar_url,
            'configuration': agent.configuration
        }
        
        return Response({
            'message': 'Agent updated successfully',
            'agent': agent_data
        }, status=status.HTTP_200_OK)
        
    except Agent.DoesNotExist:
        return Response({
            'error': 'Agent not found'
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({
            'error': 'Failed to update agent',
            'detail': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_agent(request, agent_id):
    """
    Soft delete an existing agent.
    Returns 204 on success, 404 if not found.
    """
    try:
        agent = get_object_or_404(Agent, agent_id=agent_id, user=request.user, is_deleted=False)
        
        with transaction.atomic():
            agent.is_deleted = True
            agent.save(update_fields=['is_deleted', 'updated_at'])
        
        return Response({
            'message': 'Agent deleted successfully'
        }, status=status.HTTP_204_NO_CONTENT)
        
    except Agent.DoesNotExist:
        return Response({
            'error': 'Agent not found'
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({
            'error': 'Failed to delete agent',
            'detail': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
