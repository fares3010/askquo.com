from django.urls import path
from . import views

app_name = 'plans'

urlpatterns = [ 
    path('create-checkout-session/', views.create_checkout_session, name='create_checkout_session'),
    path('get-customer-portal-url/', views.get_customer_portal_url, name='get_customer_portal_url'),
    path('get-user-subscription/', views.get_user_subscription, name='get_user_subscription'),
    path('cancel-user-subscription/', views.cancel_user_subscription, name='cancel_user_subscription'),
    path('get-user-subscription-history/', views.get_user_subscription_history, name='get_user_subscription_history'),
]
