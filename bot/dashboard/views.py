from django.shortcuts import render # type: ignore
from rest_framework.views import APIView # type: ignore
from rest_framework.response import Response # type: ignore
from rest_framework.decorators import api_view
from .models import DashboardStats
from conversations.models import Conversation # type: ignore
from rest_framework import status # type: ignore

@api_view(['GET'])
def dashboard_stats(request):
    # Check if the user is authenticated
    if not request.user.is_authenticated:
        return Response({"error": "Authentication credentials were not provided."}, status=status.HTTP_401_UNAUTHORIZED)

    # Fetch the dashboard stats from the database
    try:
        stats = DashboardStats.objects.filter(user=request.user).first()  # Assuming you have a user field in DashboardStats
        if not stats:
            return Response({"error": "No dashboard stats found."}, status=status.HTTP_404_NOT_FOUND)
    except DashboardStats.DoesNotExist:
        return Response({"error": "Dashboard stats not found."}, status=status.HTTP_404_NOT_FOUND)

    # Prepare the response data
    response_data = {
        "totalConversations": stats.total_conversations(),
        "activeUsers": stats.active_conversations(),
        "avgResponseTime": stats.avg_response_time(),
        "userSatisfaction": stats.user_satisfaction_rate(),
        "trends": {
            "conversations": {"value": stats.conversations_change_rate(), "isPositive": stats.check_conversations_rate_ispositive()},
            "users": {"value": stats.active_conversations_change_rate(), "isPositive": stats.check_active_conversations_rate_ispositive()},
            "responseTime": {"value": stats.avg_response_time_change_rate(), "isPositive": stats.check_rate_ispositive_avg_response_time()},
            "satisfaction": {"value": stats.user_satisfaction_change_rate(), "isPositive": stats.check_rate_ispositive_user_satisfaction()}
        }
    }
 
    return Response(response_data, status=status.HTTP_200_OK)

@api_view(['GET'])
def recent_chats(request):
    # Check if the user is authenticated
    if not request.user.is_authenticated:
        return Response({"error": "Authentication credentials were not provided."}, status=status.HTTP_401_UNAUTHORIZED)

    try:
        # Fetch recent conversations from the database
        conversations = Conversation.objects.filter(user=request.user).order_by('-last_message_time')[:5]  # Fetch the last 5 conversations
        if not conversations:
            return Response({"error": "No recent conversations found."}, status=status.HTTP_404_NOT_FOUND)
    except Conversation.DoesNotExist:
        return Response({"error": "Recent conversations not found."}, status=status.HTTP_404_NOT_FOUND)
    
    response_data = []
    for conversation in conversations:
        response_data.append({
            "id": conversation.conversation_id,
            "name": conversation.conversation_name,  # Assuming you have a user field in Conversation
            "lastMessage": conversation.last_message_text(),  # Assuming you have a last_message field in Conversation
            "timestamp": conversation.last_message_time(),  # Assuming you have a last_message_time field in Conversation
            "unread": conversation.check_last_message_is_read(),  # Assuming you have an unread field in Conversation
            "status": conversation.check_is_active(),  # Assuming you have a status field in Conversation
        })
    return Response(response_data, status=status.HTTP_200_OK)        
    


@api_view(['GET'])
def engagement_stats(request):
    # Check if the user is authenticated
    if not request.user.is_authenticated:
        return Response({"error": "Authentication credentials were not provided."}, status=status.HTTP_401_UNAUTHORIZED)
    
    try:
        # Fetch engagement stats from the database
        stats = DashboardStats.objects.filter(user=request.user).first()  # Assuming you have a user field in DashboardStats
        if not stats:
            return Response({"error": "No engagement stats found."}, status=status.HTTP_404_NOT_FOUND)
    except DashboardStats.DoesNotExist:
        return Response({"error": "Engagement stats not found."}, status=status.HTTP_404_NOT_FOUND)
    response_data=[]
    data=stats.last_week_conversations()
    for key in data :
        i = list(data.keys()).index(key)
        response_data.append({
            "name":key,  # Get the first three letters of the day name
            "conversations": data[key] ,  # Get the number of conversations for that day
            "responses": stats.last_week_responses()[i] # Get the number of responses for that day
        })
    # Prepare the response data
    return Response(response_data, status=status.HTTP_200_OK)

@api_view(['GET'])
def api_health_check(request):
    return Response({"status": "Api is working"}, status=status.HTTP_200_OK)
    