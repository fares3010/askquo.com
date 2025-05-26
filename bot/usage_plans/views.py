from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .models import SubscriptionPlan, UserSubscription, PlanFeature
from django.utils import timezone


# Create your views here.

@api_view(["GET"])
def get_subscription_plans(request):
    """
    Retrieve all subscription plans.

    Returns:
        Response: JSON response containing all subscription plans
    """
    plans = SubscriptionPlan.objects.all()
    plan_list = []
    for plan in plans:
        plan_list.append(plan.to_dict())

    return Response(plan_list, status=status.HTTP_200_OK)

@api_view(["GET"])
def get_user_subscriptions(request):
    """
    Retrieve all user subscriptions for a specific user.

    Returns:
        Response: JSON response containing all user subscriptions
    """

    user_id = request.query_params.get("user_id")
    if not user_id:
        return Response({"error": "User ID is required"}, status=status.HTTP_400_BAD_REQUEST)

    subscriptions = UserSubscription.objects.filter(user_id=user_id)
    subscription_list = []
    for subscription in subscriptions:
        subscription_list.append(subscription.to_dict())

    return Response(subscription_list, status=status.HTTP_200_OK)

@api_view(["POST"])
def create_user_subscription(request):
    """
    Create a new user subscription.

    Returns:
        Response: JSON response containing the new user subscription
    """
    user_id = request.data.get("user_id")
    plan_id = request.data.get("plan_id")
    stripe_subscription_id = request.data.get("stripe_subscription_id")

    if not user_id or not plan_id or not stripe_subscription_id:
        return Response({"error": "User ID, plan ID, and stripe subscription ID are required"}, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        plan = SubscriptionPlan.objects.get(plan_id=plan_id)
    except SubscriptionPlan.DoesNotExist:
        return Response({"error": "Plan not found"}, status=status.HTTP_404_NOT_FOUND)
    
    user_subscription = UserSubscription.objects.create(
        user_id=user_id,
        plan=plan,
        stripe_subscription_id=stripe_subscription_id,
        usage_start_date=timezone.now()
    )

    return Response(user_subscription.to_dict(), status=status.HTTP_201_CREATED)

@api_view(["PUT"])
def update_user_subscription(request, subscription_id):
    """
    Update an existing user subscription.

    Returns:
        Response: JSON response containing the updated user subscription
    """
    try:
        user_subscription = UserSubscription.objects.get(subscription_id=subscription_id)
    except UserSubscription.DoesNotExist:
        return Response({"error": "Subscription not found"}, status=status.HTTP_404_NOT_FOUND)
    
    user_subscription.update(request.data)
    return Response(user_subscription.to_dict(), status=status.HTTP_200_OK)

@api_view(["DELETE"])
def delete_user_subscription(request, subscription_id):
    """
    Delete a user subscription.
    
    Returns:
        Response: JSON response containing the deleted user subscription
    """
    try:
        user_subscription = UserSubscription.objects.get(subscription_id=subscription_id)
    except UserSubscription.DoesNotExist:
        return Response({"error": "Subscription not found"}, status=status.HTTP_404_NOT_FOUND)
    
    user_subscription.delete()  
    return Response({"message": "Subscription deleted successfully"}, status=status.HTTP_200_OK)

@api_view(["GET"])
def get_user_subscription_status(request, subscription_id):
    """
    Get the status of a user subscription.
    
    Returns:
        Response: JSON response containing the user subscription status
    """
    try:
        user_subscription = UserSubscription.objects.get(subscription_id=subscription_id)
    except UserSubscription.DoesNotExist:
        return Response({"error": "Subscription not found"}, status=status.HTTP_404_NOT_FOUND)
    
    return Response(user_subscription.to_dict(), status=status.HTTP_200_OK)

@api_view(["GET"])
def get_user_subscription_usage(request, subscription_id):
    """
    Get the usage of a user subscription.
    
    Returns:
        Response: JSON response containing the user subscription usage
    """
    try:
        user_subscription = UserSubscription.objects.get(subscription_id=subscription_id)
    except UserSubscription.DoesNotExist:
        return Response({"error": "Subscription not found"}, status=status.HTTP_404_NOT_FOUND)
    
    return Response(user_subscription.to_dict(), status=status.HTTP_200_OK)

@api_view(["GET"])
def get_user_subscription_usage_limits(request, subscription_id):
    """
    Get the usage limits of a user subscription.
    
    Returns:
        Response: JSON response containing the user subscription usage limits
    """
    try:
        user_subscription = UserSubscription.objects.get(subscription_id=subscription_id)
    except UserSubscription.DoesNotExist:
        return Response({"error": "Subscription not found"}, status=status.HTTP_404_NOT_FOUND)
    
    return Response(user_subscription.to_dict(), status=status.HTTP_200_OK)

@api_view(["GET"])
def get_user_subscription_usage_history(request, subscription_id):
    """
    Get the usage history of a user subscription.
    
    Returns:
        Response: JSON response containing the user subscription usage history
    """
    try:
        user_subscription = UserSubscription.objects.get(subscription_id=subscription_id)
    except UserSubscription.DoesNotExist:
        return Response({"error": "Subscription not found"}, status=status.HTTP_404_NOT_FOUND)  
    
    return Response(user_subscription.to_dict(), status=status.HTTP_200_OK)

@api_view(["GET"])
def get_user_subscription_usage_limits(request, subscription_id):
    """
    Get the usage limits of a user subscription.
    
    Returns:
        Response: JSON response containing the user subscription usage limits
    """
    try:
        user_subscription = UserSubscription.objects.get(subscription_id=subscription_id)
    except UserSubscription.DoesNotExist:
        return Response({"error": "Subscription not found"}, status=status.HTTP_404_NOT_FOUND)
    
    return Response(user_subscription.to_dict(), status=status.HTTP_200_OK)
