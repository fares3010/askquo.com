
from rest_framework.response import Response # type: ignore
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from .models import DashboardStats
from conversations.models import Conversation, ConversationMessages # type: ignore
from rest_framework import status # type: ignore
from create_agent.models import Agent # type: ignore
from django.utils import timezone

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def dashboard_stats(request):
    """
    Get dashboard statistics for the authenticated user.
    
    Returns:
        - Total agents, conversations, active conversations, and messages
        - Trend data for conversations and messages over the last 2 days
    """
    try:
        # Get user's agents and related conversations
        agents = Agent.objects.filter(user=request.user)
        conversations = Conversation.objects.filter(agent__in=agents)
        
        # Calculate basic counts
        no_of_agents = agents.count()
        no_of_conversations = conversations.count()
        no_of_active_conversations = conversations.filter(is_active=True).count()
        no_of_messages = ConversationMessages.objects.filter(conversation__in=conversations).count()
        
        # Calculate trend data with proper date handling
        today = timezone.now().date()
        two_days_ago = today - timezone.timedelta(days=2)
        one_day_ago = today - timezone.timedelta(days=1)
        
        # Get conversation counts for trend calculation
        two_days_ago_conversations = conversations.filter(created_at__date=two_days_ago).count()
        one_day_ago_conversations = conversations.filter(created_at__date=one_day_ago).count()
        
        # Calculate conversation change rate with division by zero protection
        conversations_change_rate = 0
        if two_days_ago_conversations > 0:
            conversations_change_rate = ((one_day_ago_conversations - two_days_ago_conversations) / 
                                       two_days_ago_conversations * 100)
        
        # Get message counts for trend calculation
        two_days_ago_messages = ConversationMessages.objects.filter(
            conversation__in=conversations, 
            created_at__date=two_days_ago
        ).count()
        one_day_ago_messages = ConversationMessages.objects.filter(
            conversation__in=conversations, 
            created_at__date=one_day_ago
        ).count()
        
        # Calculate message change rate with division by zero protection
        messages_change_rate = 0
        if two_days_ago_messages > 0:
            messages_change_rate = ((one_day_ago_messages - two_days_ago_messages) / 
                                  two_days_ago_messages * 100)
        
        response_data = {
            "totalAgents": no_of_agents,
            "totalConversations": no_of_conversations,
            "activeConversations": no_of_active_conversations,
            "totalMessages": no_of_messages,
            "trends": {
                "conversations": {
                    "value": round(conversations_change_rate, 2), 
                    "isPositive": conversations_change_rate > 0
                },
                "messages": {
                    "value": round(messages_change_rate, 2), 
                    "isPositive": messages_change_rate > 0
                }
            }
        }
        
        return Response(response_data, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response(
            {"error": f"Failed to retrieve dashboard stats: {str(e)}"}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def recent_chats(request):
    try:
        from django.db.models import Max
        
        agents = Agent.objects.filter(user=request.user, is_deleted=False)
        # Order by the most recent message time using the related messages field
        conversations = Conversation.objects.filter(
            agent__in=agents,
            is_archived=False,
            is_deleted=False
        ).annotate(
            last_msg_time=Max('messages__message_time')
        ).order_by('-last_msg_time')[:10]
        
        if not conversations:
            return Response(
                {"error": "No recent conversations found."}, 
                status=status.HTTP_404_NOT_FOUND
            )
        
        response_data = [{
            "id": conv.conversation_id,
            "name": conv.conversation_name,
            "lastMessage": conv.last_message_text(),
            "timestamp": conv.last_message_time(),
            "unread": conv.check_last_message_is_read(),
            "status": conv.check_is_active(),
        } for conv in conversations]
        
        return Response(response_data, status=status.HTTP_200_OK)
    except Conversation.DoesNotExist:
        return Response(
            {"error": "Recent conversations not found."}, 
            status=status.HTTP_404_NOT_FOUND
        )


@api_view(['GET'])
def api_health_check(request):
    return Response({"status": "API is working"}, status=status.HTTP_200_OK)