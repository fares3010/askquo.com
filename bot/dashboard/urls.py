from django.urls import path # type: ignore
from . import views

app_name = 'dashboard'

urlpatterns = [
    path("api/stats/", views.dashboard_stats, name="dashboard_stats"),
    path("api/recent-chats/", views.recent_chats, name="recent_chats"),
    path("api/engagement/", views.engagement_stats, name="engagement_stats"),
    path("api/health", views.api_health_check, name="api_health_check"),
]
    
