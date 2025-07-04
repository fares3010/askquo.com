"""
URL configuration for bot project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
"""

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

# Define URL patterns with clear organization and documentation
urlpatterns = [
    # Admin interface - restricted access
    path("admin/", admin.site.urls, name="admin"),
    
    # Authentication routes
    path("accounts/", include("accounts.urls", namespace="accounts")),  # Custom auth views
    #path("auth/", include("allauth.urls")),  # Third-party auth
    
    # Core application routes - using namespaced URLs
    path("conversations/", include("conversations.urls", namespace="conversations")),  # Chat functionality
    path("dashboard/", include("dashboard.urls", namespace="dashboard")),  # User dashboard
    path("plans/", include("plans.urls", namespace="plans")),  # Subscription plans
    path("agents/", include("create_agent.urls", namespace="agents")),  # Agent management
    path("integrations/", include("integrations.urls", namespace="integrations")),  # Third-party integrations
]

# Add media serving in development only
#if settings.DEBUG:
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)