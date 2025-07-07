import logging
from typing import Dict, Any
from paypalcheckoutsdk.core import PayPalHttpClient, SandboxEnvironment, LiveEnvironment
from django.core.exceptions import ImproperlyConfigured
import os
from dotenv import load_dotenv
import requests
from rest_framework.response import Response
from rest_framework import status
import json
import hashlib
import base64
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
from django.utils.dateparse import parse_datetime
from django.db import transaction
from django.core.exceptions import ValidationError
from django.urls import reverse
from plans.models import PayPalSubscription, PayPalWebhookEvent, PlanPrice, UserSubscription
from django.utils import timezone


load_dotenv()

# Environment variables
PAYPAL_CLIENT_ID = os.getenv('PAYPAL_CLIENT_ID')
PAYPAL_SECRET_ID = os.getenv('PAYPAL_SECRET_ID')
PAYPAL_ENVIRONMENT = os.getenv('PAYPAL_ENVIRONMENT', 'sandbox')

logger = logging.getLogger(__name__)

# PayPal API endpoints
PAYPAL_BASE_URL = "https://api-m.paypal.com" if PAYPAL_ENVIRONMENT == 'live' else "https://api-m.sandbox.paypal.com"
PAYPAL_TOKEN_URL = f"{PAYPAL_BASE_URL}/v1/oauth2/token"
PAYPAL_PRODUCTS_URL = f"{PAYPAL_BASE_URL}/v1/catalogs/products"
PAYPAL_PLANS_URL = f"{PAYPAL_BASE_URL}/v1/billing/plans"
PAYPAL_SUBSCRIPTIONS_URL = f"{PAYPAL_BASE_URL}/v1/billing/subscriptions"

def debug_paypal_auth():
    """
    Debug function to test PayPal authentication and see exactly what's returned.
    """
    print(f"🔧 PayPal Debug Information:")
    print(f"   Environment: {PAYPAL_ENVIRONMENT}")
    print(f"   Client ID exists: {bool(PAYPAL_CLIENT_ID)}")
    print(f"   Client ID (first 10 chars): {PAYPAL_CLIENT_ID[:10] if PAYPAL_CLIENT_ID else 'None'}...")
    print(f"   Secret exists: {bool(PAYPAL_SECRET_ID)}")
    print(f"   Secret (first 10 chars): {PAYPAL_SECRET_ID[:10] if PAYPAL_SECRET_ID else 'None'}...")
    print(f"   Token URL: {PAYPAL_TOKEN_URL}")
    
    if not PAYPAL_CLIENT_ID or not PAYPAL_SECRET_ID:
        print("❌ Missing PayPal credentials!")
        return None
    
    try:
        print("\n🚀 Making PayPal token request...")
        response = requests.post(
            PAYPAL_TOKEN_URL,
            headers={"Accept": "application/json"},
            data={"grant_type": "client_credentials"},
            auth=(PAYPAL_CLIENT_ID, PAYPAL_SECRET_ID),
            timeout=30
        )
        
        print(f"📊 Response Status: {response.status_code}")
        print(f"📊 Response Headers: {dict(response.headers)}")
        
        try:
            response_json = response.json()
            print(f"📊 Response JSON: {response_json}")
            
            if 'access_token' in response_json:
                print("✅ Access token found!")
                return response_json['access_token']
            else:
                print("❌ No access_token in response!")
                if 'error' in response_json:
                    print(f"❌ Error: {response_json['error']}")
                if 'error_description' in response_json:
                    print(f"❌ Error Description: {response_json['error_description']}")
                return None
                
        except Exception as json_error:
            print(f"❌ Failed to parse JSON response: {json_error}")
            print(f"❌ Raw response text: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")
        return None

def get_paypal_access_token() -> str:
    """
    Get PayPal access token for API authentication.
    
    Returns:
        str: Access token for PayPal API calls
        
    Raises:
        Exception: If token retrieval fails
    """
    try:
        if not PAYPAL_CLIENT_ID or not PAYPAL_SECRET_ID:
            raise ImproperlyConfigured("PayPal client ID and secret are required")
        
        logger.debug(f"Using PayPal environment: {PAYPAL_ENVIRONMENT}")
        logger.debug(f"PayPal token URL: {PAYPAL_TOKEN_URL}")
        logger.debug(f"PayPal client ID exists: {bool(PAYPAL_CLIENT_ID)}")
        logger.debug(f"PayPal secret exists: {bool(PAYPAL_SECRET_ID)}")
            
        response = requests.post(
            PAYPAL_TOKEN_URL,
            headers={"Accept": "application/json"},
            data={"grant_type": "client_credentials"},
            auth=(PAYPAL_CLIENT_ID, PAYPAL_SECRET_ID),
            timeout=30
        )
        
        logger.debug(f"PayPal token response status: {response.status_code}")
        logger.debug(f"PayPal token response headers: {dict(response.headers)}")
        
        response.raise_for_status()
        
        token_data = response.json()
        logger.debug(f"PayPal token response data keys: {list(token_data.keys())}")
        
        if 'access_token' not in token_data:
            logger.error(f"Access token not found in response. Full response: {token_data}")
            raise Exception(f"Access token not found in response: {token_data}")
            
        logger.debug("PayPal access token retrieved successfully")
        return token_data['access_token']
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get PayPal access token: {str(e)}")
        raise Exception(f"PayPal token request failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error getting PayPal access token: {str(e)}")
        raise

def create_paypal_product(name: str = "Basic Plans", description: str = "Access to Basic Plans") -> Dict[str, Any]:
    """
    Create a PayPal product.
    
    Args:
        name: Product name
        description: Product description
        
    Returns:
        Dict containing product creation response
        
    Raises:
        Exception: If product creation fails
    """
    try:
        token = get_paypal_access_token()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        data = {
            "name": name,
            "description": description,
            "type": "SERVICE",
            "category": "SOFTWARE"
        }

        response = requests.post(
            PAYPAL_PRODUCTS_URL,
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        logger.info(f"PayPal product created successfully: {result.get('id', 'Unknown ID')}")
        return result
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to create PayPal product: {str(e)}")
        raise Exception(f"PayPal product creation failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error creating PayPal product: {str(e)}")
        raise

def create_paypal_plan(
    product_id: str, 
    name: str, 
    description: str, 
    price: str,
    currency: str = "USD",
    interval_unit: str = "MONTH",
    interval_count: int = 1
) -> Dict[str, Any]:
    """
    Create a PayPal subscription plan.
    
    Args:
        product_id: PayPal product ID
        name: Plan name
        description: Plan description
        price: Plan price as string
        currency: Currency code (default: USD)
        interval_unit: Billing interval unit (default: MONTH)
        interval_count: Billing interval count (default: 1)
        
    Returns:
        Dict containing plan creation response
        
    Raises:
        Exception: If plan creation fails
    """
    try:
        # Input validation
        if not product_id or not isinstance(product_id, str):
            raise ValueError("product_id must be a non-empty string")
        
        # Validate that the product exists
        if not validate_paypal_product(product_id):
            raise ValueError(f"PayPal product {product_id} does not exist or is not accessible")
        
        if not name or not isinstance(name, str):
            raise ValueError("name must be a non-empty string")
            
        if not description or not isinstance(description, str):
            raise ValueError("description must be a non-empty string")
        
        
        # Validate currency
        valid_currencies = ["USD", "EUR", "GBP", "AUD", "CAD", "JPY"]
        if currency not in valid_currencies:
            raise ValueError(f"currency must be one of {valid_currencies}, got: {currency}")
        
        # Validate interval_unit
        valid_intervals = ["DAY", "WEEK", "MONTH", "YEAR"]
        if interval_unit not in valid_intervals:
            raise ValueError(f"interval_unit must be one of {valid_intervals}, got: {interval_unit}")
        
        # Validate interval_count
        if not isinstance(interval_count, int) or interval_count < 1:
            raise ValueError("interval_count must be a positive integer")
        
        logger.debug(f"Creating PayPal plan with: product_id={product_id}, name={name}, price={price}, currency={currency}")
        
        token = get_paypal_access_token()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            'Accept': 'application/json',
            'PayPal-Request-Id': 'plan_1234567890',
            'Prefer':'return=representation',
        }

        data = {
            "product_id": product_id,
            "name": name,
            "description": description,
            "status": "ACTIVE",
            "billing_cycles": [
                {
                    "frequency": {
                        "interval_unit": interval_unit,
                        "interval_count": interval_count
                    },
                    "tenure_type": "REGULAR",
                    "sequence": 1,
                    "total_cycles": 3,  # 0 = infinite until canceled
                    "pricing_scheme": {
                        "fixed_price": {
                            "value": price,
                            "currency_code": currency
                        }
                    }
                }
            ],
            "payment_preferences": {
                "auto_bill_outstanding": True,
                "setup_fee": {
                    "value": "0",
                    "currency_code": currency
                },
                "setup_fee_failure_action": "CONTINUE",
                "payment_failure_threshold": 3
            },
            "taxes": {
                "percentage": "0",
                "inclusive": False
            }
        }

        logger.debug(f"PayPal plan request data: {data}")

        response = requests.post(
            PAYPAL_PLANS_URL,
            headers=headers,
            data=data,
            timeout=30
        )
        
        logger.debug(f"PayPal plan response status: {response.status_code}")
        logger.debug(f"PayPal plan response headers: {dict(response.headers)}")
        
        # Enhanced error handling for 422 errors
        if response.status_code == 422:
            try:
                error_details = response.json()
                logger.error(f"PayPal plan creation failed with 422 - Full error response: {error_details}")
                
                # Extract specific error details
                error_message = "PayPal plan creation failed (422 - Unprocessable Entity)"
                if 'details' in error_details:
                    for detail in error_details['details']:
                        error_message += f"\n- {detail.get('field', 'Unknown field')}: {detail.get('description', 'Unknown error')}"
                
                raise Exception(error_message)
            except Exception as json_error:
                logger.error(f"Failed to parse 422 error response: {json_error}")
                logger.error(f"Raw 422 response: {response.text}")
                raise Exception(f"PayPal plan creation failed with 422 error. Raw response: {response.text}")
        
        response.raise_for_status()
        
        result = response.json()
        logger.info(f"PayPal plan created successfully: {result.get('id', 'Unknown ID')}")
        return result
        
    except ValueError as e:
        logger.error(f"Validation error in create_paypal_plan: {str(e)}")
        raise Exception(f"Invalid input parameters: {str(e)}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to create PayPal plan: {str(e)}")
        raise Exception(f"PayPal plan creation failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error creating PayPal plan: {str(e)}")
        raise

def create_paypal_subscription(
    plan_id: str,
    return_url: str = "https://askquo.lovable.app/subscription/success",
    cancel_url: str = "https://askquo.lovable.app/subscription/cancel",
    brand_name: str = "AskQuo"
) -> str:
    """
    Create a PayPal subscription and return approval URL.
    
    Args:
        plan_id: PayPal plan ID
        return_url: URL to redirect after successful subscription
        cancel_url: URL to redirect after cancellation
        brand_name: Brand name for the subscription
        
    Returns:
        str: Approval URL for user redirection
        
    Raises:
        Exception: If subscription creation fails
    """
    try:
        token = get_paypal_access_token()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Prefer": "return=representation"
        }

        data = {
            "plan_id": plan_id,
            'quantity': '1',
            "application_context": {
                "brand_name": brand_name,
                "locale": "fr-EG",
                "shipping_preference": "NO_SHIPPING",
                "user_action": "SUBSCRIBE_NOW",
                "return_url": return_url,
                "cancel_url": cancel_url
            }
        }

        response = requests.post(
            PAYPAL_SUBSCRIPTIONS_URL,
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        
        subscription = response.json()
        
        # Find approval URL from links
        approval_url = None
        for link in subscription.get('links', []):
            if link.get('rel') == 'approve':
                approval_url = link.get('href')
                break
                
        if not approval_url:
            raise Exception("Approval URL not found in subscription response")
            
        logger.info(f"PayPal subscription created successfully: {subscription.get('id', 'Unknown ID')}")
        return approval_url
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to create PayPal subscription: {str(e)}")
        raise Exception(f"PayPal subscription creation failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error creating PayPal subscription: {str(e)}")
        raise

def get_paypal_client() -> PayPalHttpClient:
    """
    Create and return a PayPal HTTP client with appropriate environment configuration.
    
    Returns:
        PayPalHttpClient: Configured PayPal client for API interactions
        
    Raises:
        ImproperlyConfigured: If required PayPal settings are missing
    """
    # Validate required settings
    if not PAYPAL_CLIENT_ID or not PAYPAL_SECRET_ID:
        raise ImproperlyConfigured("PayPal client ID and secret are required")
        
    
    # Determine environment
    try:
        if PAYPAL_ENVIRONMENT == 'live':
            environment = LiveEnvironment(
                client_id=PAYPAL_CLIENT_ID,
                client_secret=PAYPAL_SECRET_ID
            )
            logger.info("Initialized PayPal Live Environment")
        else:
            environment = SandboxEnvironment(
                client_id=PAYPAL_CLIENT_ID,
                client_secret=PAYPAL_SECRET_ID
            )
            logger.info("Initialized PayPal Sandbox Environment")
        
        client = PayPalHttpClient(environment)
        logger.debug("PayPal client created successfully")
        return client
        
    except Exception as e:
        logger.error(f"Failed to create PayPal client: {str(e)}")
        raise ImproperlyConfigured(f"PayPal client initialization failed: {str(e)}")

def get_paypal_environment_name() -> str:
    """
    Get the current PayPal environment name.
    
    Returns:
        str: Environment name ('live' or 'sandbox')
    """
    return PAYPAL_ENVIRONMENT.lower()

def is_paypal_live() -> bool:
    """
    Check if PayPal is configured for live environment.
    
    Returns:
        bool: True if live environment, False if sandbox
    """
    return get_paypal_environment_name() == 'live'


# Simple wrapper functions for backward compatibility
def create_paypal_product_simple():
    """Simple wrapper - use create_paypal_product() for better error handling."""
    return create_paypal_product()

def create_paypal_plan_simple(product_id, name, description, price):
    """Simple wrapper - use create_paypal_plan() for better error handling."""
    return create_paypal_plan(product_id, name, description, str(price))

def create_subscription_simple(plan_id):
    """Simple wrapper - use create_paypal_subscription() for better error handling."""
    return create_paypal_subscription(plan_id)

def validate_paypal_product(product_id: str) -> bool:
    """
    Validate that a PayPal product exists.
    
    Args:
        product_id: PayPal product ID to validate
        
    Returns:
        bool: True if product exists, False otherwise
    """
    try:
        token = get_paypal_access_token()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        
        response = requests.get(
            f"{PAYPAL_PRODUCTS_URL}/{product_id}",
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            product = response.json()
            logger.debug(f"Product validation successful: {product.get('name', 'Unknown')}")
            return True
        elif response.status_code == 404:
            logger.warning(f"Product {product_id} not found")
            return False
        else:
            logger.warning(f"Product validation failed with status {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"Error validating product {product_id}: {str(e)}")
        return False

def debug_paypal_plan_creation(
    product_id: str,
    name: str = "Debug Test Plan",
    description: str = "Debug test plan for troubleshooting",
    price: str = "9.99",
    currency: str = "USD"
):
    """
    Debug function to test PayPal plan creation with detailed logging.
    """
    print(f"🔧 PayPal Plan Creation Debug:")
    print(f"   Product ID: {product_id}")
    print(f"   Plan Name: {name}")
    print(f"   Description: {description}")
    print(f"   Price: {price}")
    print(f"   Currency: {currency}")
    
    try:
        # Validate product first
        print(f"\n🔍 Validating product {product_id}...")
        if not validate_paypal_product(product_id):
            print(f"❌ Product {product_id} does not exist or is not accessible")
            return None
        else:
            print(f"✅ Product {product_id} is valid")
        
        # Get token
        print(f"\n🔑 Getting access token...")
        token = get_paypal_access_token()
        print(f"✅ Access token retrieved")
        
        # Prepare request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        
        data = {
            "product_id": product_id,
            "name": name,
            "description": description,
            "billing_cycles": [
                {
                    "frequency": {
                        "interval_unit": "MONTH",
                        "interval_count": 1
                    },
                    "tenure_type": "REGULAR",
                    "sequence": 1,
                    "total_cycles": 0,
                    "pricing_scheme": {
                        "fixed_price": {
                            "value": price,
                            "currency_code": currency
                        }
                    }
                }
            ],
            "payment_preferences": {
                "auto_bill_outstanding": True,
                "setup_fee": {
                    "value": "0.00",
                    "currency_code": currency
                },
                "setup_fee_failure_action": "CONTINUE",
                "payment_failure_threshold": 3
            },
            "taxes": {
                "percentage": "0.00",
                "inclusive": False
            }
        }
        
        print(f"\n📝 Request URL: {PAYPAL_PLANS_URL}")
        print(f"📝 Request Headers: {headers}")
        print(f"📝 Request Data: {json.dumps(data, indent=2)}")
        
        # Make request
        print(f"\n🚀 Making PayPal plan creation request...")
        response = requests.post(
            PAYPAL_PLANS_URL,
            headers=headers,
            json=data,
            timeout=30
        )
        
        print(f"📊 Response Status: {response.status_code}")
        print(f"📊 Response Headers: {dict(response.headers)}")
        
        try:
            response_json = response.json()
            print(f"📊 Response JSON: {json.dumps(response_json, indent=2)}")
            
            if response.status_code == 201:
                print("✅ Plan created successfully!")
                return response_json
            else:
                print(f"❌ Plan creation failed with status {response.status_code}")
                if 'details' in response_json:
                    print("❌ Error details:")
                    for detail in response_json['details']:
                        print(f"   - Field: {detail.get('field', 'Unknown')}")
                        print(f"     Issue: {detail.get('issue', 'Unknown')}")
                        print(f"     Description: {detail.get('description', 'Unknown')}")
                return None
                
        except Exception as json_error:
            print(f"❌ Failed to parse JSON response: {json_error}")
            print(f"❌ Raw response text: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")
        return None




# views.py


logger = logging.getLogger(__name__)



@method_decorator(csrf_exempt, name='dispatch')
class PayPalWebhookView(View):
    
    def post(self, request):
        try:
            # Validate request content type
            if request.content_type != 'application/json':
                logger.warning("Invalid content type for webhook")
                return JsonResponse({'error': 'Invalid content type'}, status=400)
            
            # Get webhook data
            webhook_data = json.loads(request.body)
            event_type = webhook_data.get('event_type')
            event_id = webhook_data.get('id')
            
            if not event_id or not event_type:
                logger.warning("Missing required webhook fields")
                return JsonResponse({'error': 'Missing required fields'}, status=400)
            
            # Verify webhook signature
            if not self.verify_webhook_signature(request):
                logger.warning(f"Invalid webhook signature for event {event_id}")
                return JsonResponse({'error': 'Invalid signature'}, status=400)
            
            # Process webhook with transaction
            with transaction.atomic():
                # Check if event already processed
                if PayPalWebhookEvent.objects.filter(event_id=event_id).exists():
                    logger.info(f"Event {event_id} already processed")
                    return JsonResponse({'status': 'already_processed'}, status=200)
                
                # Store webhook event
                webhook_event = PayPalWebhookEvent.objects.create(
                    event_id=event_id,
                    event_type=event_type,
                    resource_id=webhook_data.get('resource', {}).get('id', ''),
                    raw_data=webhook_data
                )
                
                # Process the webhook based on event type
                if event_type in self.get_subscription_events():
                    try:
                        self.process_subscription_event(webhook_data)
                        webhook_event.processed = True
                    except Exception as e:
                        webhook_event.processing_error = str(e)
                        logger.error(f"Error processing webhook {event_id}: {str(e)}")
                        raise
                
                webhook_event.save()
            
            logger.info(f"Successfully processed webhook event {event_id} of type {event_type}")
            return JsonResponse({'status': 'success'}, status=200)
            
        except json.JSONDecodeError:
            logger.error("Invalid JSON in webhook payload")
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
        except ValidationError as e:
            logger.error(f"Validation error: {str(e)}")
            return JsonResponse({'error': 'Validation error'}, status=400)
        except Exception as e:
            logger.error(f"Error processing webhook: {str(e)}")
            return JsonResponse({'error': 'Internal server error'}, status=500)
    
    def verify_webhook_signature(self, request) -> bool:
        """Verify PayPal webhook signature with improved security"""
        try:
            # Get required headers
            transmission_id = request.headers.get('PAYPAL-TRANSMISSION-ID')
            cert_id = request.headers.get('PAYPAL-CERT-ID')
            transmission_time = request.headers.get('PAYPAL-TRANSMISSION-TIME')
            transmission_sig = request.headers.get('PAYPAL-TRANSMISSION-SIG')
            
            if not all([transmission_id, cert_id, transmission_time, transmission_sig]):
                logger.warning("Missing required webhook headers")
                return False
            
            # Get webhook configuration
            webhook_id = os.getenv('PAYPAL_WEBHOOK_CLIENT_ID')
            if not webhook_id:
                logger.warning("PAYPAL_WEBHOOK_ID not configured")
                return False
            
            # Create expected signature
            expected_sig = self.create_signature(
                transmission_id, transmission_time, webhook_id, request.body
            )

            if transmission_sig != expected_sig:
                logger.warning(f"Webhook signature mismatch: {transmission_sig} != {expected_sig}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error verifying webhook signature: {str(e)}")
            return False
    
    def create_signature(self, transmission_id: str, timestamp: str, webhook_id: str, body: bytes) -> str:
        """Create signature for webhook verification"""
        message = f"{transmission_id}|{timestamp}|{webhook_id}|{hashlib.sha256(body).hexdigest()}"
        return base64.b64encode(message.encode()).decode()
    
    def get_subscription_events(self) -> list:
        """Return list of subscription-related events to process"""
        return [
            'BILLING.SUBSCRIPTION.CREATED',
            'BILLING.SUBSCRIPTION.ACTIVATED',
            'BILLING.SUBSCRIPTION.UPDATED',
            'BILLING.SUBSCRIPTION.EXPIRED',
            'BILLING.SUBSCRIPTION.CANCELLED',
            'BILLING.SUBSCRIPTION.SUSPENDED',
            'BILLING.SUBSCRIPTION.PAYMENT.FAILED',
            'PAYMENT.SALE.COMPLETED',
            'PAYMENT.SALE.DENIED',
        ]
    
    def process_subscription_event(self, webhook_data: Dict[str, Any]) -> None:
        """Process subscription-related webhook events with improved error handling"""
        event_type = webhook_data.get('event_type')
        resource = webhook_data.get('resource', {})
        
        event_handlers = {
            'BILLING.SUBSCRIPTION.CREATED': self.handle_subscription_created,
            'BILLING.SUBSCRIPTION.ACTIVATED': self.handle_subscription_activated,
            'BILLING.SUBSCRIPTION.UPDATED': self.handle_subscription_updated,
            'BILLING.SUBSCRIPTION.EXPIRED': self.handle_subscription_expired,
            'BILLING.SUBSCRIPTION.CANCELLED': self.handle_subscription_cancelled,
            'BILLING.SUBSCRIPTION.SUSPENDED': self.handle_subscription_suspended,
            'BILLING.SUBSCRIPTION.PAYMENT.FAILED': self.handle_payment_failed,
            'PAYMENT.SALE.COMPLETED': self.handle_payment_completed,
            'PAYMENT.SALE.DENIED': self.handle_payment_denied,
        }
        if event_type == 'BILLING.SUBSCRIPTION.CREATED':
            self.handle_subscription_created(resource)
        elif event_type == 'BILLING.SUBSCRIPTION.ACTIVATED':
            self.handle_subscription_activated(resource)
        elif event_type == 'BILLING.SUBSCRIPTION.UPDATED':
            self.handle_subscription_updated(resource)
        elif event_type == 'BILLING.SUBSCRIPTION.EXPIRED':
            self.handle_subscription_expired(resource)
        elif event_type == 'BILLING.SUBSCRIPTION.CANCELLED':
            self.handle_subscription_cancelled(resource)
        elif event_type == 'BILLING.SUBSCRIPTION.SUSPENDED':
            self.handle_subscription_suspended(resource)
        elif event_type == 'BILLING.SUBSCRIPTION.PAYMENT.FAILED':
            self.handle_payment_failed(resource)
        elif event_type == 'PAYMENT.SALE.COMPLETED':
            self.handle_payment_completed(resource)
        elif event_type == 'PAYMENT.SALE.DENIED':
            self.handle_payment_denied(resource)
    
    def handle_subscription_created(self, resource: Dict[str, Any]) -> None:
        """Handle subscription creation with improved error handling and validation"""
        subscription_id = resource.get('id')
        plan_id = resource.get('plan_id')
        
        if not subscription_id:
            logger.error("Missing subscription ID in webhook resource")
            return
        
        try:
            if PayPalSubscription.objects.filter(paypal_subscription_id=subscription_id).exists():
                subscription = PayPalSubscription.objects.get(paypal_subscription_id=subscription_id)
                subscription.status = 'APPROVAL_PENDING'
                subscription.save(update_fields=['status', 'updated_at'])
                logger.info(f"Subscription {subscription_id} marked as created")
            else:
                # For new subscriptions, we can't create them without a user
                # The user will be associated when the subscription is activated
                logger.info(f"New subscription {subscription_id} created, waiting for activation")
                # Note: We'll create the subscription record when the user completes the approval process
        except PayPalSubscription.DoesNotExist:
            logger.warning(f"Subscription {subscription_id} not found in database")
        except Exception as e:
            logger.error(f"Error updating subscription {subscription_id}: {str(e)}")
    
    def handle_subscription_activated(self, resource: Dict[str, Any]) -> None:
        """Handle subscription activation with comprehensive billing info update"""
        subscription_id = resource.get('id')
        
        if not subscription_id:
            logger.error("Missing subscription ID in webhook resource")
            return
        
        try:
            subscription = PayPalSubscription.objects.get(paypal_subscription_id=subscription_id)
            subscription.status = 'ACTIVE'
            
            billing_info = resource.get('billing_info', {})
            if billing_info.get('next_billing_time'):
                try:
                    next_billing_time = parse_datetime(billing_info['next_billing_time'])
                    if next_billing_time:
                        subscription.next_billing_time = next_billing_time
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid next_billing_time format: {billing_info['next_billing_time']}")
            
            subscription.save(update_fields=['status', 'next_billing_time', 'updated_at'])
            logger.info(f"Subscription {subscription_id} activated")
            
            self.send_subscription_activated_notification(subscription)
            
        except PayPalSubscription.DoesNotExist:
            logger.warning(f"Subscription {subscription_id} not found in database")
        except Exception as e:
            logger.error(f"Error activating subscription {subscription_id}: {str(e)}")
    
    def handle_subscription_updated(self, resource: Dict[str, Any]) -> None:
        """Handle subscription updates with validation"""
        subscription_id = resource.get('id')
        
        if not subscription_id:
            logger.error("Missing subscription ID in webhook resource")
            return
        
        try:
            subscription = PayPalSubscription.objects.get(paypal_subscription_id=subscription_id)
            
            billing_info = resource.get('billing_info', {})
            if billing_info.get('next_billing_time'):
                try:
                    next_billing_time = parse_datetime(billing_info['next_billing_time'])
                    if next_billing_time:
                        subscription.next_billing_time = next_billing_time
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid next_billing_time format: {billing_info['next_billing_time']}")
            
            subscription.save(update_fields=['next_billing_time', 'updated_at'])
            logger.info(f"Subscription {subscription_id} updated")
            
        except PayPalSubscription.DoesNotExist:
            logger.warning(f"Subscription {subscription_id} not found in database")
        except Exception as e:
            logger.error(f"Error updating subscription {subscription_id}: {str(e)}")
    
    def handle_subscription_expired(self, resource: Dict[str, Any]) -> None:
        """Handle subscription expiration with notification"""
        subscription_id = resource.get('id')
        
        if not subscription_id:
            logger.error("Missing subscription ID in webhook resource")
            return
        
        try:
            subscription = PayPalSubscription.objects.get(paypal_subscription_id=subscription_id)
            subscription.status = 'EXPIRED'
            subscription.save(update_fields=['status', 'updated_at'])
            logger.info(f"Subscription {subscription_id} expired")
            
            self.send_subscription_expired_notification(subscription)
            
        except PayPalSubscription.DoesNotExist:
            logger.warning(f"Subscription {subscription_id} not found in database")
        except Exception as e:
            logger.error(f"Error expiring subscription {subscription_id}: {str(e)}")
    
    def handle_subscription_cancelled(self, resource: Dict[str, Any]) -> None:
        """Handle subscription cancellation with notification"""
        subscription_id = resource.get('id')
        
        if not subscription_id:
            logger.error("Missing subscription ID in webhook resource")
            return
        
        try:
            subscription = PayPalSubscription.objects.get(paypal_subscription_id=subscription_id)
            subscription.status = 'CANCELLED'
            subscription.save(update_fields=['status', 'updated_at'])
            logger.info(f"Subscription {subscription_id} cancelled")
            
            self.send_subscription_cancelled_notification(subscription)
            
        except PayPalSubscription.DoesNotExist:
            logger.warning(f"Subscription {subscription_id} not found in database")
        except Exception as e:
            logger.error(f"Error cancelling subscription {subscription_id}: {str(e)}")
    
    def handle_subscription_suspended(self, resource: Dict[str, Any]) -> None:
        """Handle subscription suspension with notification"""
        subscription_id = resource.get('id')
        
        if not subscription_id:
            logger.error("Missing subscription ID in webhook resource")
            return
        
        try:
            subscription = PayPalSubscription.objects.get(paypal_subscription_id=subscription_id)
            subscription.status = 'SUSPENDED'
            subscription.save(update_fields=['status', 'updated_at'])
            logger.info(f"Subscription {subscription_id} suspended")
            
            self.send_subscription_suspended_notification(subscription)
            
        except PayPalSubscription.DoesNotExist:
            logger.warning(f"Subscription {subscription_id} not found in database")
        except Exception as e:
            logger.error(f"Error suspending subscription {subscription_id}: {str(e)}")
    
    def handle_payment_failed(self, resource: Dict[str, Any]) -> None:
        """Handle failed payment with enhanced logging"""
        subscription_id = resource.get('id')
        
        if not subscription_id:
            logger.error("Missing subscription ID in payment failed webhook")
            return
        
        try:
            subscription = PayPalSubscription.objects.get(paypal_subscription_id=subscription_id)
            logger.warning(f"Payment failed for subscription {subscription_id}")
            
            self.send_payment_failed_notification(subscription)
            
        except PayPalSubscription.DoesNotExist:
            logger.warning(f"Subscription {subscription_id} not found in database")
        except Exception as e:
            logger.error(f"Error handling payment failure for subscription {subscription_id}: {str(e)}")
    
    def handle_payment_completed(self, resource: Dict[str, Any]) -> None:
        """Handle successful payment with subscription lookup"""
        payment_id = resource.get('id')
        billing_agreement_id = resource.get('billing_agreement_id')
        plan_id = resource.get('plan_id')
        
        if not payment_id:
            logger.error("Missing payment ID in webhook resource")
            return
        
        logger.info(f"Payment completed: {payment_id}")
        
        if billing_agreement_id:
            try:
                plan_price = PlanPrice.objects.get(paypal_price_id=plan_id)
                subscription = PayPalSubscription.objects.get(paypal_subscription_id=billing_agreement_id)
                logger.info(f"Payment {payment_id} associated with subscription {billing_agreement_id}")
                UserSubscription.objects.create(
                    user=subscription.user,
                    plan_name=plan_price.plan.plan_name,
                    price=plan_price,
                    paypal_subscription_id=billing_agreement_id,
                    created_at=timezone.now(),
                    updated_at=timezone.now(),
                )
            except PayPalSubscription.DoesNotExist:
                logger.warning(f"No subscription found for billing agreement {billing_agreement_id}")
    
    def handle_payment_denied(self, resource: Dict[str, Any]) -> None:
        """Handle denied payment with detailed logging"""
        payment_id = resource.get('id')
        billing_agreement_id = resource.get('billing_agreement_id')
        
        if not payment_id:
            logger.error("Missing payment ID in webhook resource")
            return
        
        logger.warning(f"Payment denied: {payment_id}")
        
        if billing_agreement_id:
            try:
                subscription = PayPalSubscription.objects.get(paypal_subscription_id=billing_agreement_id)
                self.send_payment_failed_notification(subscription)
            except PayPalSubscription.DoesNotExist:
                logger.warning(f"No subscription found for billing agreement {billing_agreement_id}")
    
    def send_subscription_activated_notification(self, subscription: PayPalSubscription) -> None:
        """Send notification when subscription is activated"""
        try:
            logger.info(f"Sending activation notification for subscription {subscription.paypal_subscription_id}")
            # TODO: Implement actual notification logic (email, SMS, etc.)
            
        except Exception as e:
            logger.error(f"Failed to send activation notification: {str(e)}")
    def send_subscription_expired_notification(self, subscription: PayPalSubscription) -> None:
        """Send notification when subscription expires"""
        try:
            logger.info(f"Sending expiration notification for subscription {subscription.paypal_subscription_id}")
            # TODO: Implement actual notification logic (email, SMS, etc.)
        except Exception as e:
            logger.error(f"Failed to send expiration notification: {str(e)}")
    
    
    def send_subscription_cancelled_notification(self, subscription):
        """Send notification when subscription is cancelled"""
        try:
            logger.info(f"Sending cancellation notification for subscription {subscription.paypal_subscription_id}")
            # TODO: Implement actual notification logic (email, SMS, etc.)
        except Exception as e:
            logger.error(f"Failed to send cancellation notification: {str(e)}")
    
    def send_subscription_suspended_notification(self, subscription):
        """Send notification when subscription is suspended"""
        try:
            logger.info(f"Sending suspension notification for subscription {subscription.paypal_subscription_id}")
            # TODO: Implement actual notification logic (email, SMS, etc.)
        except Exception as e:
            logger.error(f"Failed to send suspension notification: {str(e)}")
    
    def send_payment_failed_notification(self, subscription):
        """Send notification when payment fails"""
        try:
            logger.info(f"Sending payment failure notification for subscription {subscription.paypal_subscription_id}")
            # TODO: Implement actual notification logic (email, SMS, etc.)
        except Exception as e:
            logger.error(f"Failed to send payment failure notification: {str(e)}")



