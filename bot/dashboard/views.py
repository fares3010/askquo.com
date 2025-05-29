from django.shortcuts import render # type: ignore
from rest_framework.views import APIView # type: ignore
from rest_framework.response import Response # type: ignore
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from .models import DashboardStats
from conversations.models import Conversation # type: ignore
from rest_framework import status # type: ignore

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def dashboard_stats(request):
    try:
        stats = DashboardStats.objects.filter(user=request.user).first()
        if not stats:
            return Response(
                {"error": "No dashboard stats found."}, 
                status=status.HTTP_404_NOT_FOUND
            )

        response_data = {
            "totalConversations": stats.total_conversations(),
            "activeUsers": stats.active_conversations(),
            "avgResponseTime": stats.avg_response_time(),
            "userSatisfaction": stats.user_satisfaction_rate(),
            "trends": {
                "conversations": {
                    "value": stats.conversations_change_rate(), 
                    "isPositive": stats.check_conversations_rate_ispositive()
                },
                "users": {
                    "value": stats.active_conversations_change_rate(), 
                    "isPositive": stats.check_active_conversations_rate_ispositive()
                },
                "responseTime": {
                    "value": stats.avg_response_time_change_rate(), 
                    "isPositive": stats.check_rate_ispositive_avg_response_time()
                },
                "satisfaction": {
                    "value": stats.user_satisfaction_change_rate(), 
                    "isPositive": stats.check_rate_ispositive_user_satisfaction()
                }
            }
        }
        return Response(response_data, status=status.HTTP_200_OK)
    except DashboardStats.DoesNotExist:
        return Response(
            {"error": "Dashboard stats not found."}, 
            status=status.HTTP_404_NOT_FOUND
        )

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def recent_chats(request):
    try:
        conversations = Conversation.objects.filter(
            user=request.user
        ).order_by('-last_message_time')[:5]
        
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
@permission_classes([IsAuthenticated])
def engagement_stats(request):
    try:
        stats = DashboardStats.objects.filter(user=request.user).first()
        if not stats:
            return Response(
                {"error": "No engagement stats found."}, 
                status=status.HTTP_404_NOT_FOUND
            )

        conversations_data = stats.last_week_conversations()
        responses_data = stats.last_week_responses()
        
        response_data = [
            {
                "name": day,
                "conversations": conversations_data[day],
                "responses": responses_data[i]
            }
            for i, day in enumerate(conversations_data.keys())
        ]
        
        return Response(response_data, status=status.HTTP_200_OK)
    except DashboardStats.DoesNotExist:
        return Response(
            {"error": "Engagement stats not found."}, 
            status=status.HTTP_404_NOT_FOUND
        )

@api_view(['GET'])
def api_health_check(request):
    return Response({"status": "API is working"}, status=status.HTTP_200_OK)