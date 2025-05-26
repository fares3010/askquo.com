from django.urls import path
from . import views

app_name = 'plans'

urlpatterns = [
    path('', views.get_subscription_plans, name='get_subscription_plans'),
    path('subscriptions/', views.get_user_subscriptions, name='get_user_subscriptions'),
    path('subscriptions/create/', views.create_user_subscription, name='create_user_subscription'),
    path('subscriptions/<int:subscription_id>/', views.update_user_subscription, name='update_user_subscription'),
    path('subscriptions/<int:subscription_id>/delete/', views.delete_user_subscription, name='delete_user_subscription'),
    path('subscriptions/<int:subscription_id>/status/', views.get_user_subscription_status, name='get_user_subscription_status'),
    path('subscriptions/<int:subscription_id>/usage/', views.get_user_subscription_usage, name='get_user_subscription_usage'),
    path('subscriptions/<int:subscription_id>/limits/', views.get_user_subscription_usage_limits, name='get_user_subscription_usage_limits'),
    path('subscriptions/<int:subscription_id>/history/', views.get_user_subscription_usage_history, name='get_user_subscription_usage_history'),
]
