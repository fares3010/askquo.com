from django.shortcuts import get_object_or_404
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework import status 
from django.utils import timezone
from django.db import transaction
import logging
import os 
from .models import SubscriptionPlan, UserSubscription, PlanPrice
from .payment.stripe_gateway import StripeGateway
from .payment.paypal import create_paypal_subscription
from django.views.decorators.csrf import csrf_exempt
from dotenv import load_dotenv
load_dotenv()
    
logger = logging.getLogger(__name__)



@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_user_subscription(request):
    """
    Get the current user's active subscription details.
    
    Returns:
        - 200: Subscription details
        - 401: User not authenticated
        - 404: No active subscription found
        - 500: Internal server error
    """
    try:
        user = request.user
        
        # Get the most recent active subscription for the user
        user_subscription = UserSubscription.objects.filter(
            user=user, 
            is_active=True,
            is_deleted=False
        ).select_related('price').first()
        
        
        if not user_subscription:
            return Response(
                {"error": "No active subscription found"}, 
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Check if subscription is still valid
        if user_subscription.is_expired():
            return Response(
                {"error": "Subscription has expired"}, 
                status=status.HTTP_410_GONE
            )
        
        data = {
            "plan_id": user_subscription.price.plan.plan_id,
            "plan_name": user_subscription.price.plan.plan_name,
            "features_limit": {feature.feature_name: feature.feature_limit for feature in user_subscription.price.plan.features.all()},
            "subscription_id": user_subscription.subscription_id,
            "subscription_start_date": user_subscription.usage_start_date,
            "subscription_end_date": user_subscription.usage_end_date,
            "subscription_status": user_subscription.is_active,
            "subscription_updated_at": user_subscription.updated_at,
            "remaining_days": user_subscription.get_remaining_days(),
            "is_expired": user_subscription.is_expired(),
            "tokens_usage": user_subscription.tokens_usage,
            "conversations_usage": user_subscription.conversations_usage,
            "agents_created": user_subscription.agents_usage,
            "widgets_created": user_subscription.widgets_usage,
            "team_members_created": user_subscription.team_members_usage,
        }
        
        return Response(data, status=status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"Error in get_user_subscription: {str(e)}")
        return Response(
            {"error": "Internal server error"}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    
    
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def cancel_user_subscription(request):
    """
    Cancel the current user's active subscription.
    
    Returns:
        - 200: Subscription cancelled successfully
        - 400: Invalid request data
        - 401: User not authenticated
        - 500: Internal server error
    """
    try:
        user = request.user
        subscription_id = request.data.get('subscription_id')
        
        if not subscription_id:
            return Response(
                {"error": "Subscription ID is required"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        subscription = UserSubscription.objects.filter(
            user=user,
            subscription_id=subscription_id,
            is_active=True,
            is_deleted=False
        ).first()
        
        if not subscription:
            return Response(
                {"error": "Subscription not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        
        with transaction.atomic():
            stripe_gateway = StripeGateway()
            immediate = request.data.get('immediate', False)
            stripe_subscription = stripe_gateway.cancel_subscription(
                subscription, 
                user.stripe_customers.first(), 
                immediate
            )
            
            if not stripe_subscription.success:
                return Response(
                    {"error": stripe_subscription.error},
                    status=status.HTTP_400_BAD_REQUEST  
                )   
            
            subscription.is_active = False
            subscription.is_deleted = True
            subscription.save()
            
            logger.info(f"Subscription {subscription_id} cancelled successfully for user {user.id}")
            return Response(
                {"message": "Subscription cancelled successfully"},
                status=status.HTTP_200_OK
            )
            
    except Exception as e:
        logger.error(f"Error in cancel_user_subscription: {str(e)}")
        return Response(
            {"error": "Internal server error"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_user_subscription_history(request):
    """
    Get the current user's subscription history.
    
    Returns:
        - 200: Subscription history
        - 401: User not authenticated
        - 500: Internal server error
    """
    try:
        user = request.user
        subscription_history = UserSubscription.objects.filter(
            user=user,
            is_active=False,
            is_deleted=True
        ).select_related('price').order_by('-updated_at')
        
        if not subscription_history.exists():
            return Response(
                {"error": "No subscription history found"},
                status=status.HTTP_404_NOT_FOUND
            )
        
        data = [{
            "subscription_id": subscription.subscription_id,
            "plan_id": subscription.price.plan.plan_id,
            "subscription_start_date": subscription.usage_start_date,
            "subscription_end_date": subscription.usage_end_date,
            "subscription_status": subscription.is_active,
            "subscription_updated_at": subscription.updated_at,
        } for subscription in subscription_history]

        return Response(data, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Error in get_user_subscription_history: {str(e)}")
        return Response(
            {"error": "Internal server error"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@csrf_exempt
def webhook_raw(request):
    """
    Raw webhook handler that bypasses Django's request processing
    to preserve payload integrity for Stripe signature verification.
    
    Returns:
        - 200: Webhook processed successfully
        - 400: Invalid webhook signature or payload
        - 500: Internal server error
    """
    import json
    from django.http import HttpResponse
    
    stripe_gateway = StripeGateway()
    
    try:
        # Check request method
        if request.method != 'POST':
            logging.warning(f"Webhook received with invalid method: {request.method}")
            return HttpResponse(
                json.dumps({"error": "Method not allowed"}),
                status=405,
                content_type='application/json'
            )
        
        # Get raw payload directly from request.body (bypasses Django processing)
        payload = request.body
        signature = request.META.get('HTTP_STRIPE_SIGNATURE')
        
        # Additional debugging for payload integrity
        content_type = request.META.get('CONTENT_TYPE', '')
        content_length = request.META.get('CONTENT_LENGTH', '')
        user_agent = request.META.get('HTTP_USER_AGENT', '')
        remote_addr = request.META.get('REMOTE_ADDR', '')
        
        # Enhanced debug logging
        logging.info(f"Raw webhook received - Method: {request.method}")
        logging.info(f"Payload type: {type(payload)}, length: {len(payload) if payload else 'None'}")
        logging.info(f"Content-Type: {content_type}")
        logging.info(f"Content-Length: {content_length}")
        logging.info(f"User-Agent: {user_agent}")
        logging.info(f"Remote-Addr: {remote_addr}")
        logging.info(f"Stripe signature: {'Present' if signature else 'Missing'}")
        
        # Validate required components
        if not signature:
            logging.warning("Webhook received without Stripe signature")
            return HttpResponse(
                json.dumps({"error": "Missing Stripe signature header"}),
                status=400,
                content_type='application/json'
            )
        
        if not payload:
            logging.error("Webhook received with empty payload")
            return HttpResponse(
                json.dumps({"error": "Empty payload"}),
                status=400,
                content_type='application/json'
            )
        
        # Verify payload integrity
        if content_length:
            try:
                expected_length = int(content_length)
                actual_length = len(payload)
                if expected_length != actual_length:
                    logging.warning(f"Content-Length mismatch: expected {expected_length}, got {actual_length}")
            except ValueError:
                logging.warning(f"Invalid Content-Length header: {content_length}")
        
        # Log payload characteristics for debugging
        if isinstance(payload, bytes):
            logging.info(f"Payload is bytes - first 100 chars: {payload[:100]}")
            try:
                payload_str = payload.decode('utf-8')
                logging.info(f"Payload decodes to UTF-8 successfully")
            except UnicodeDecodeError as e:
                logging.error(f"Payload cannot be decoded as UTF-8: {e}")
        else:
            logging.warning(f"Payload is not bytes but {type(payload)}")
        
        # Process webhook with raw payload
        webhook_response = stripe_gateway.handle_webhook(payload, signature)
        
        if not webhook_response.success:
            logging.error(f"Webhook processing failed: {webhook_response.error}")
            
            # Return appropriate error response
            if "signature" in webhook_response.error.lower():
                return HttpResponse(
                    json.dumps({
                        "error": "Signature verification failed",
                        "details": webhook_response.error
                    }),
                    status=400,
                    content_type='application/json'
                )
            else:
                return HttpResponse(
                    json.dumps({
                        "error": "Webhook processing error",
                        "details": webhook_response.error
                    }),
                    status=400,
                    content_type='application/json'
                )
        
        logging.info(f"Webhook processed successfully: {webhook_response.data}")
        return HttpResponse(
            json.dumps({
                "success": True,
                "event_type": webhook_response.data.get('event', 'unknown'),
                "message": "Webhook processed successfully"
            }),
            status=200,
            content_type='application/json'
        )
        
    except Exception as e:
        logging.error(f"Unexpected error in raw webhook handler: {str(e)}")
        logging.error(f"Error type: {type(e)}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        
        return HttpResponse(
            json.dumps({
                "error": "Internal server error",
                "details": "An unexpected error occurred processing the webhook"
            }),
            status=500,
            content_type='application/json'
        )

@csrf_exempt    
@api_view(['POST'])  
def webhook(request):
    """
    Handle webhook from Stripe.
    
    Returns:
        - 200: Webhook processed successfully
        - 400: Invalid webhook signature or payload
        - 500: Internal server error
    """
    stripe_gateway = StripeGateway()
    try:
        # Get raw payload and signature
        payload = request.body
        signature = request.headers.get('stripe-signature') or request.headers.get('Stripe-Signature')
        
        # Additional debugging for payload integrity
        content_type = request.headers.get('content-type', '')
        content_length = request.headers.get('content-length', '')
        user_agent = request.headers.get('user-agent', '')
        
        # Debug logging
        logging.info(f"Webhook received - Payload type: {type(payload)}, Payload length: {len(payload) if payload else 'None'}")
        logging.info(f"Content-Type: {content_type}")
        logging.info(f"Content-Length: {content_length}")
        logging.info(f"User-Agent: {user_agent}")
        logging.info(f"Stripe signature header: {'Present' if signature else 'Missing'}")
        logging.info(f"Webhook secret configured: {'Yes' if stripe_gateway.stripe_webhook_secret else 'No'}")
        
        # Validate required headers
        if not signature:
            logging.warning("Webhook received without Stripe signature")
            return Response(
                {"error": "Missing Stripe signature header"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Validate that this looks like a Stripe request
        if not user_agent.startswith('Stripe'):
            logging.warning(f"Suspicious user agent: {user_agent}")
            # Don't reject immediately, but log for monitoring
        
        # Validate payload exists and is not empty
        if not payload:
            logging.error("Webhook received with empty payload")
            return Response(
                {"error": "Empty payload"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Ensure payload is bytes (most common issue)
        if isinstance(payload, str):
            logging.warning("Payload is string, converting to bytes")
            payload = payload.encode('utf-8')
        elif not isinstance(payload, bytes):
            logging.error(f"Invalid payload type: {type(payload)}")
            return Response(
                {"error": f"Invalid payload type: {type(payload)}"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Log payload integrity check
        if content_length:
            try:
                expected_length = int(content_length)
                actual_length = len(payload)
                if expected_length != actual_length:
                    logging.warning(f"Content-Length mismatch: expected {expected_length}, got {actual_length}")
            except ValueError:
                logging.warning(f"Invalid Content-Length header: {content_length}")
        
        # Process webhook
        webhook_response = stripe_gateway.handle_webhook(payload, signature)
        
        if not webhook_response.success:
            logging.error(f"Webhook processing failed: {webhook_response.error}")
            
            # Return specific error codes for different failure types
            if "signature" in webhook_response.error.lower():
                return Response(
                    {
                        "error": "Signature verification failed",
                        "details": webhook_response.error
                    }, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            elif "configuration" in webhook_response.error.lower():
                return Response(
                    {
                        "error": "Webhook configuration error",
                        "details": webhook_response.error
                    }, 
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            else:
                return Response(
                    {
                        "error": "Webhook processing error",
                        "details": webhook_response.error
                    }, 
                    status=status.HTTP_400_BAD_REQUEST
                )
        
        logging.info(f"Webhook processed successfully: {webhook_response.data}")
        return Response(
            {
                "success": True,
                "event_type": webhook_response.data.get('event', 'unknown'),
                "message": "Webhook processed successfully"
            },
            status=status.HTTP_200_OK
        )
        
    except Exception as e:
        logging.error(f"Unexpected error in webhook handler: {str(e)}")
        logging.error(f"Error type: {type(e)}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        
        return Response(
            {
                "error": "Internal server error",
                "details": "An unexpected error occurred processing the webhook"
            }, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def create_checkout_session(request):
    """
    Create a checkout session for the current user.
    
    Returns:
        - 200: Checkout session created successfully
        - 400: Invalid request data
        - 401: User not authenticated
        - 500: Internal server error
    """
    try:
        user = request.user
        price_id = request.data.get('price_id')
        success_url = request.data.get('success_url') or os.getenv('STRIPE_SUCCESS_URL')
        cancel_url = request.data.get('cancel_url') or os.getenv('STRIPE_CANCEL_URL')

        if not price_id or not success_url or not cancel_url:
            return Response(
                {"error": "Missing required fields"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        price = get_object_or_404(PlanPrice, price_id=price_id, is_active=True, is_deleted=False)
        if price.paypal_price_id is None:
            return Response(
                {"error": "Paypal price ID is not set"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        checkout_url = create_paypal_subscription(plan_id=price.paypal_price_id)
        return Response({"checkout_url": checkout_url}, status=status.HTTP_200_OK)
        
        #stripe_gateway = StripeGateway()
        #checkout_session = stripe_gateway.checkout_session(user, price, success_url, cancel_url)


        #if not checkout_session.success:    
            #return Response(
                #{"error": checkout_session.error},
                #status=status.HTTP_400_BAD_REQUEST
            #)
        
        #return Response(
            #{"checkout_url": checkout_session.data['checkout_url']},
            #status=status.HTTP_200_OK
        #)   
    except Exception as e:
        logging.error(f"Unexpected error in create_checkout_session: {str(e)}")
        return Response(
            {"error": "Internal server error"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_customer_portal_url(request):
    """
    Get the customer portal URL for the current user.
    
    Returns:
        - 200: Customer portal URL
        - 400: Invalid request data
        - 401: User not authenticated
        - 500: Internal server error    
    """
    try:
        user = request.user
        stripe_gateway = StripeGateway()
        customer_portal_url = stripe_gateway.get_customer_portal_url(user)
        return Response({"customer_portal_url": customer_portal_url.data['url']}, status=status.HTTP_200_OK)
    except Exception as e:
        logging.error(f"Unexpected error in get_customer_portal_url: {str(e)}")
        return Response(
            {"error": "Internal server error"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR        
        )
    
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def create_paypal_subscription_view(request):
    """
    Create a PayPal subscription for the current user.
    
    Returns:
        - 200: Subscription created successfully
        - 400: Invalid request data
        - 401: User not authenticated
        - 500: Internal server error
    """
    try:
        user = request.user
        plan_id = request.data.get('plan_id')
        
        if not plan_id:
            return Response(
                {"error": "Missing required fields"},
                status=status.HTTP_400_BAD_REQUEST
            )
        plan = SubscriptionPlan.objects.get(plan_id=plan_id)
        if plan.is_active == False or plan.is_deleted == True:
            return Response(
                {"error": "Plan is not active or deleted"},
                status=status.HTTP_400_BAD_REQUEST
            )
        price = PlanPrice.objects.get(plan=plan, is_active=True, is_deleted=False)
        if price.paypal_price_id is None:
            return Response(
                {"error": "Paypal price ID is not set"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        approval_url = create_paypal_subscription(plan_id=price.paypal_price_id)
        
        return Response({"approval_url": approval_url}, status=status.HTTP_200_OK)
    except Exception as e:
        logging.error(f"Unexpected error in create_paypal_subscription_view: {str(e)}")
        return Response(
            {"error": "Internal server error"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    