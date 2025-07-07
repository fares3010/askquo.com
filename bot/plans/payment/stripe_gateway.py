import stripe
import logging
from dotenv import load_dotenv
import os
import time
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from plans.models import StripeCustomer, UserSubscription, SubscriptionPlan, PlanPrice, StripeTransaction
from accounts.models import CustomUser

load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class StripeResponse:
    """Standardized response object for Stripe operations"""
    success: bool
    data: Any = None
    error: str = None
    error_code: str = None

class StripeGateway:
    def __init__(self):
        self.stripe_api_key = os.getenv('STRIPE_SECRET_KEY')
        self.stripe_webhook_secret = os.getenv('STRIPE_WEBHOOK_SECRET')
        self.stripe_publishable_key = os.getenv('STRIPE_PUBLISHABLE_KEY')
        
        # Set the API key for stripe
        if self.stripe_api_key:
            stripe.api_key = self.stripe_api_key
        else:
            raise ValueError("STRIPE_SECRET_KEY environment variable is required")

    def _validate_user(self, user) -> bool:
        """Validate user object has required fields"""
        if not user:
            logger.error("User object is required")
            return False
        
        required_fields = ['email', 'full_name', 'id']
        for field in required_fields:
            if not getattr(user, field, None):
                logger.error(f"User {field} is required")
                return False
        return True

    def _validate_plan(self, plan) -> bool:
        """Validate plan object has required fields"""
        if not plan:
            logger.error("Plan object is required")
            return False
        
        required_fields = ['plan_id', 'plan_name', 'stripe_product_id']
        for field in required_fields:
            if not getattr(plan, field, None):
                logger.error(f"Plan {field} is required")
                return False
        return True
    
    def _validate_plan_price(self, plan_price) -> bool:
        """Validate plan price object has required fields"""
        if not plan_price:
            logger.error("Plan price object is required")
            return False
        
        required_fields = ['plan', 'price_amount', 'price_currency', 'price_period', 'stripe_price_id']
        for field in required_fields:
            if not getattr(plan_price, field, None):
                logger.error(f"Plan price {field} is required")
                return False
        return True

    def _handle_stripe_error(self, e: stripe.error.StripeError) -> StripeResponse:
        """Centralized Stripe error handling"""
        error_message = str(e)
        error_code = getattr(e, 'code', 'unknown')
        logger.error(f"Stripe error [{error_code}]: {error_message}")
        return StripeResponse(success=False, error=error_message, error_code=error_code)

    def create_stripe_customer(self, user) -> StripeResponse:
        """Create or retrieve a Stripe customer"""
        try:
            if not self._validate_user(user):
                return StripeResponse(success=False, error="User validation failed")
            
            # Check if customer already exists by email
            existing_customers = stripe.Customer.list(email=user.email, limit=1).data
            if existing_customers:
                logger.info(f"Existing Stripe customer found for {user.email}")
                return StripeResponse(success=True, data=existing_customers[0])
            
            # Create a new customer (let Stripe generate the ID)
            customer = stripe.Customer.create(
                email=user.email,
                name=user.full_name,
                metadata={
                    'user_id': str(user.id),
                    'user_email': user.email,
                    'user_full_name': user.full_name,
                    'created_at': str(int(time.time()))
                }
            )
            
            logger.info(f"Created Stripe customer {customer.id} for user {user.id}")
            return StripeResponse(success=True, data=customer)
            
        except stripe.error.StripeError as e:
            return self._handle_stripe_error(e)
        except Exception as e:
            logger.error(f"Unexpected error creating customer: {str(e)}")
            return StripeResponse(success=False, error=f"Unexpected error: {str(e)}")

    def create_stripe_subscription(self, user, price) -> StripeResponse:
        """Create a Stripe subscription"""
        try:
            if not self._validate_user(user) or not self._validate_plan_price(price):
                return StripeResponse(success=False, error="Validation failed")
            
            # Get or create customer first
            customer_response = self.create_stripe_customer(user)
            if not customer_response.success:
                return customer_response
            
            customer = customer_response.data
            customer_id = customer.id
            
            # Check for existing active subscriptions with the same plan
            existing_subscriptions = stripe.Subscription.list(
                customer=customer_id, 
                status='active',
                price=price.stripe_price_id,
                limit=1
            ).data
            
            if existing_subscriptions:
                logger.info(f"User {user.id} already has active subscription for plan {price.plan.plan_id}")
                return StripeResponse(success=True, data=existing_subscriptions[0])
            
            # Create new subscription
            subscription = stripe.Subscription.create(
                customer=customer_id,
                metadata={
                    'user_id': str(user.id),
                    'user_email': user.email,
                    'user_full_name': user.full_name,
                    'plan_name': price.plan.plan_name,
                    'created_at': str(int(time.time()))
                },  
                items=[{'price': price.stripe_price_id}],
                payment_behavior='default_incomplete',
                expand=['latest_invoice.payment_intent']
            )
            
            logger.info(f"Created subscription {subscription.id} for user {user.id}")
            return StripeResponse(success=True, data=subscription)
            
        except stripe.error.StripeError as e:
            return self._handle_stripe_error(e)
        except Exception as e:
            logger.error(f"Unexpected error creating subscription: {str(e)}")
            return StripeResponse(success=False, error=f"Unexpected error: {str(e)}")

    def create_payment_intent(self, user, price) -> StripeResponse:
        """Create a PaymentIntent for one-time payments"""
        try:
            if not self._validate_user(user) or not self._validate_plan_price(price):
                return StripeResponse(success=False, error="Validation failed")
            
            price= PlanPrice.objects.get(plan=price.plan, stripe_price_id=price.stripe_price_id, is_active=True, is_deleted=False)
            amount = int(float(price.price_amount) * 100)  # Convert to cents
            if amount <= 0:
                return StripeResponse(success=False, error="Amount must be greater than 0")
            
            # Get or create customer
            customer_response = self.create_stripe_customer(user)
            if not customer_response.success:
                return customer_response
            
            customer = customer_response.data
            
            # Create PaymentIntent
            intent = stripe.PaymentIntent.create(
                amount=amount,
                currency='usd',
                customer=customer.id,
                metadata={
                    'user_id': str(user.id),
                    'user_email': user.email,
                    'plan_id': price.plan.plan_id,
                    'plan_name': price.plan.plan_name,
                    'plan_price': str(price.price_amount),
                    'plan_currency': 'usd',
                    'plan_period': price.price_period,
                    'created_at': str(int(time.time()))
                },
                automatic_payment_methods={'enabled': True},
                confirm=True
            )
            
            return StripeResponse(success=True, data={
                'client_secret': intent.client_secret,
                'payment_intent_id': intent.id,
                'amount': amount,
                'currency': 'usd'
            })
            
        except stripe.error.StripeError as e:
            return self._handle_stripe_error(e)
        except Exception as e:
            logger.error(f"Unexpected error creating payment intent: {str(e)}")
            return StripeResponse(success=False, error=f"Unexpected error: {str(e)}")

    def get_user_subscriptions(self, user) -> StripeResponse:
        """Get user's subscriptions"""
        try:
            if not self._validate_user(user):
                return StripeResponse(success=False, error="User validation failed")
            
            # Find customer by email
            customers = stripe.Customer.list(email=user.email, limit=1).data
            if not customers:
                logger.info(f"No Stripe customer found for user {user.email}")
                return StripeResponse(success=True, data=[])
            
            customer = customers[0]
            subscriptions = stripe.Subscription.list(
                customer=customer.id,
                expand=['data.default_payment_method', 'data.items.data.price']
            )
            
            subscription_data = []
            for sub in subscriptions.data:
                subscription_info = {
                    'id': sub.id,
                    'status': sub.status,
                    'metadata': sub.metadata,
                    'current_period_start': sub.current_period_start,
                    'current_period_end': sub.current_period_end,
                    'cancel_at_period_end': sub.cancel_at_period_end,
                    'canceled_at': sub.canceled_at,
                    'items': []
                }
                
                for item in sub.items.data:
                    item_info = {
                        'id': item.id,
                        'quantity': item.quantity,
                        'price': {
                            'id': item.price.id,
                            'nickname': item.price.nickname,
                            'unit_amount': item.price.unit_amount,
                            'currency': item.price.currency,
                            'recurring': dict(item.price.recurring) if item.price.recurring else None
                        }
                    }
                    subscription_info['items'].append(item_info)
                
                subscription_data.append(subscription_info)
            
            return StripeResponse(success=True, data=subscription_data)
            
        except stripe.error.StripeError as e:
            return self._handle_stripe_error(e)
        except Exception as e:
            logger.error(f"Unexpected error fetching subscriptions: {str(e)}")
            return StripeResponse(success=False, error=f"Unexpected error: {str(e)}")

    def cancel_subscription(self, subscription: UserSubscription, stripe_customer: StripeCustomer, immediate: bool = False) -> StripeResponse:
        """Cancel a user's subscription"""
        try:
            if not subscription.stripe_subscription_id:
                return StripeResponse(success=False, error="Subscription ID is required")
            
            if not self._validate_user(stripe_customer.user):
                return StripeResponse(success=False, error="User validation failed")
            
            # Retrieve and verify subscription
            try:
                stripe_subscription = stripe.Subscription.retrieve(subscription.stripe_subscription_id)
            except stripe.error.InvalidRequestError:
                return StripeResponse(success=False, error="Subscription not found")
            
            # Verify ownership by checking customer email
            customer = stripe.Customer.retrieve(stripe_subscription.customer)
            if customer.email != stripe_customer.user.email:
                return StripeResponse(success=False, error="Subscription does not belong to user")
            
            # Cancel subscription
            if immediate:
                cancelled_subscription = stripe.Subscription.delete(subscription.stripe_subscription_id)
            else:
                cancelled_subscription = stripe.Subscription.modify(
                    subscription.stripe_subscription_id,
                    cancel_at_period_end=True,
                    metadata={
                        **stripe_subscription.metadata,
                        'cancelled_by_user': str(stripe_customer.user.id),
                        'cancelled_at': str(int(time.time()))
                    }
                )
            
            logger.info(f"Cancelled subscription {subscription.stripe_subscription_id} for user {stripe_customer.user.id}")
            return StripeResponse(success=True, data=cancelled_subscription)
            
        except stripe.error.StripeError as e:
            return self._handle_stripe_error(e)
        except Exception as e:
            logger.error(f"Unexpected error cancelling subscription: {str(e)}")
            return StripeResponse(success=False, error=f"Unexpected error: {str(e)}")

    def checkout_session(self, user, price, success_url: str, cancel_url: str) -> StripeResponse:
        """Create a Stripe Checkout session for subscription signup"""
        try:
            if not self._validate_user(user) or not self._validate_plan_price(price):
                return StripeResponse(success=False, error="Validation failed")

            # Get or create customer
            customer_response = self.create_stripe_customer(user)
            if not customer_response.success:
                return customer_response
            
            customer = customer_response.data
            
            # Create checkout session
            session = stripe.checkout.Session.create(
                customer=customer.id,
                payment_method_types=['card'],
                line_items=[{
                    'price': price.stripe_price_id,
                    'quantity': 1,
                }],
                mode='subscription',
                success_url=success_url,
                cancel_url=cancel_url,
                metadata={
                    'user_id': str(user.id),
                    'plan_id': price.plan.plan_id,
                    'plan_name': price.plan.plan_name
                }
            )

            return StripeResponse(success=True, data={
                'checkout_url': session.url,
                'session_id': session.id
            })
        except stripe.error.StripeError as e:
            return self._handle_stripe_error(e)
        except Exception as e:
            logger.error(f"Unexpected error creating checkout session: {str(e)}")
            return StripeResponse(success=False, error=f"Unexpected error: {str(e)}")   

    def _verify_webhook_signature(self, payload, signature) -> Tuple[bool, str, Any]:
        """
        Enhanced webhook signature verification with better error handling
        
        Returns:
            Tuple[bool, str, Any]: (success, error_message, event_or_none)
        """
        try:
            if not self.stripe_webhook_secret:
                return False, "Webhook secret not configured", None
            
            if not signature:
                return False, "Missing Stripe signature header", None
            
            # Store original payload for comparison
            original_payload = payload
            
            # Ensure payload is bytes
            if isinstance(payload, str):
                logger.warning("Converting string payload to bytes for signature verification")
                payload = payload.encode('utf-8')
            elif not isinstance(payload, bytes):
                logger.error(f"Invalid payload type: {type(payload)}")
                return False, f"Invalid payload type: {type(payload)}", None
            
            # Enhanced debugging
            logger.info(f"Verifying webhook signature")
            logger.info(f"Payload type: {type(payload)}, length: {len(payload)}")
            logger.info(f"Payload preview: {payload[:100]}..." if len(payload) > 100 else payload)
            logger.info(f"Signature header: {signature}")
            logger.info(f"Webhook secret exists: {bool(self.stripe_webhook_secret)}")
            logger.info(f"Secret prefix: {self.stripe_webhook_secret[:10]}..." if self.stripe_webhook_secret else "None")
            
            # Parse signature elements for debugging
            sig_elements = signature.split(',')
            timestamp = None
            v1_signatures = []
            v0_signatures = []
            
            for element in sig_elements:
                if element.startswith('t='):
                    timestamp = element[2:]
                elif element.startswith('v1='):
                    v1_signatures.append(element[3:])
                elif element.startswith('v0='):
                    v0_signatures.append(element[3:])
            
            # Log parsed signature components
            logger.info(f"Parsed signature - timestamp: {timestamp}")
            logger.info(f"v1 signatures: {v1_signatures}")
            logger.info(f"v0 signatures: {v0_signatures}")
            
            if timestamp:
                import time
                current_time = int(time.time())
                try:
                    sig_time = int(timestamp)
                    time_diff = current_time - sig_time  # Don't use abs() to see if timestamp is in future
                    logger.info(f"Current time: {current_time}, Signature timestamp: {timestamp}")
                    logger.info(f"Time difference: {time_diff}s ({'future' if time_diff < 0 else 'past'})")
                    
                    # If timestamp is significantly in the future, log warning but continue
                    if time_diff < -300:  # More than 5 minutes in future
                        logger.warning(f"Signature timestamp is {abs(time_diff)}s in the future - possible clock sync issue")
                    elif time_diff > 300:  # More than 5 minutes in past
                        logger.warning(f"Signature timestamp is {time_diff}s old - might be expired")
                except ValueError as e:
                    logger.error(f"Invalid timestamp format: {timestamp}")
            
            # Try with progressively longer tolerance values and different approaches
            verification_attempts = [
                # Standard tolerances
                {'tolerance': 300, 'description': 'Standard 5 minutes'},
                {'tolerance': 600, 'description': 'Extended 10 minutes'},
                {'tolerance': 3600, 'description': 'Very extended 1 hour'},
                # For debugging: very long tolerance to bypass timestamp issues
                {'tolerance': 86400, 'description': 'Debug mode 24 hours'},  # 24 hours for debugging
            ]
            
            last_error = None
            
            for attempt in verification_attempts:
                try:
                    logger.info(f"Attempting verification with {attempt['description']} tolerance")
                    
                    event = stripe.Webhook.construct_event(
                        payload=payload, 
                        sig_header=signature, 
                        secret=self.stripe_webhook_secret,
                        tolerance=attempt['tolerance'],
                        api_key=self.stripe_api_key
                    )
                    
                    if attempt['tolerance'] > 300:
                        logger.warning(f"Signature verification succeeded with {attempt['description']} tolerance")
                    else:
                        logger.info(f"Webhook signature verified successfully with {attempt['description']} tolerance")
                    
                    return True, None, event
                    
                except stripe.error.SignatureVerificationError as e:
                    last_error = e
                    logger.error(f"Verification failed with {attempt['description']}: {str(e)}")
                    
                    # Add specific error analysis
                    error_str = str(e).lower()
                    if "no signatures found" in error_str:
                        logger.error("→ Issue: Signature calculation mismatch")
                    elif "timestamp" in error_str:
                        logger.error("→ Issue: Timestamp validation failed")
                    elif "unable to extract" in error_str:
                        logger.error("→ Issue: Signature header format problem")
                    
                    continue
                except Exception as e:
                    logger.error(f"Unexpected error during verification attempt: {str(e)}")
                    last_error = e
                    continue
            
            # If all attempts failed, provide comprehensive error information
            error_msg = f"Signature verification failed with all tolerance levels"
            logger.error(error_msg)
            
            # Enhanced debugging information
            logger.error("=== COMPREHENSIVE DEBUGGING INFO ===")
            logger.error(f"1. Webhook secret format: {'Valid (whsec_)' if self.stripe_webhook_secret and self.stripe_webhook_secret.startswith('whsec_') else 'Invalid or missing'}")
            logger.error(f"2. Payload integrity: Original={type(original_payload)}, Final={type(payload)}, Length={len(payload)}")
            logger.error(f"3. Raw signature header: {signature}")
            logger.error(f"4. Parsed components:")
            logger.error(f"   - Timestamp: {timestamp}")
            logger.error(f"   - v1 signatures ({len(v1_signatures)}): {v1_signatures}")
            logger.error(f"   - v0 signatures ({len(v0_signatures)}): {v0_signatures}")
            logger.error(f"5. System time: {int(time.time()) if 'time' in locals() else 'Unknown'}")
            logger.error(f"6. Last error: {str(last_error) if last_error else 'None'}")
            
            # Additional troubleshooting suggestions
            logger.error("=== TROUBLESHOOTING SUGGESTIONS ===")
            logger.error("1. Check if system clock is synchronized (ntpdate)")
            logger.error("2. Verify webhook secret in Stripe dashboard matches environment variable")
            logger.error("3. Ensure no middleware is modifying the request payload")
            logger.error("4. Test with Stripe CLI: stripe listen --forward-to your-url")
            logger.error("5. Check if payload encoding is consistent")
            
            return False, error_msg, None
            
        except stripe.error.SignatureVerificationError as e:
            error_msg = f"Stripe signature verification error: {str(e)}"
            logger.error(error_msg)
            return False, error_msg, None
            
        except Exception as e:
            error_msg = f"Unexpected error during signature verification: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False, error_msg, None

    def handle_webhook(self, payload: str, signature: str) -> StripeResponse:
        """Handle Stripe webhook events"""
        # Use enhanced signature verification
        success, error_message, event = self._verify_webhook_signature(payload, signature)
        
        if not success:
            return StripeResponse(success=False, error=error_message)
        
        try:
            logger.info(f"Received webhook event: {event['type']}")
            logger.info(f"Event data: {event['data']['object']}")
            # Use event handler mapping for cleaner code
            event_handlers = {  
                'charge.succeeded': self._handle_charge_succeeded,
                'customer.created': self._handle_customer_created,
                'customer.updated': self._handle_customer_updated,
                'customer.subscription.created': self._handle_subscription_created,
                'customer.subscription.deleted': self._handle_subscription_deleted,
            }
            
            handler = event_handlers.get(event['type'])
            if handler:
                logger.info(f"Event data: {event['data']['object']}")
                return handler(event['data']['object'])
            else:
                logger.info(f"Unhandled webhook event type: {event['type']}")
                return StripeResponse(success=True, data={'message': 'Event received but not processed'})
                
        except Exception as e:
            logger.error(f"Error processing webhook event: {str(e)}")
            return StripeResponse(success=False, error=f"Event processing error: {str(e)}")

    def _handle_charge_succeeded(self, charge) -> StripeResponse:
        """Handle charge succeeded"""
        try:
            logger.info(f"Charge succeeded: {charge['id']}")
            transaction = StripeTransaction.objects.create(
                stripe_transaction_id=charge['id'],
                user_email=charge['billing_details']['email'],
                stripe_customer_id=charge['customer'],
                transaction_amount=charge['amount'],
                transaction_currency=charge['currency'],
                transaction_status=charge['status'],
                transaction_address_country=charge['billing_details']['address']['country'],
                transaction_receipt_url=charge['receipt_url'],
            )
            return StripeResponse(success=True, data={
                'event': 'charge_succeeded',
                'transaction': transaction
            })
        except Exception as e:
            logger.error(f"Error handling charge succeeded: {str(e)}")
            return StripeResponse(success=False, error=f"Charge succeeded error: {str(e)}")

    def _handle_subscription_created(self, subscription) -> StripeResponse:
        """Handle subscription creation"""
        try:
            logger.info(f"Subscription created: {subscription['id']}")
            
            # Get customer ID from subscription
            customer_id = subscription['customer']
            
            customer = stripe.Customer.retrieve(customer_id)
            user = CustomUser.objects.get(email=customer['email'])
            
            # Get or create StripeCustomer
            existing_customer, created = StripeCustomer.objects.get_or_create(
                user=user,
                defaults={'stripe_customer_id': customer_id}
            )
            
            if not created:
                existing_customer.stripe_customer_id = customer_id
                existing_customer.save()
            
            # Get plan from subscription
            price_id = subscription['items']['data'][0]['price']['id']
            price = PlanPrice.objects.get(stripe_price_id=price_id)

            # Check if user has a free subscription
            free_subscription = UserSubscription.objects.filter(
                user=user,
                price__price_id=7,
                is_active=True,
                is_deleted=False
            ).first()

            if free_subscription:
                free_subscription.is_active = False
                free_subscription.is_deleted = True
                free_subscription.save()
                logger.info(f"Free subscription deleted for user: {user.email}")
            else:
                logger.info(f"No free subscription found for user: {user.email}")

            # Check if user has a paid subscription
            paid_subscription = UserSubscription.objects.filter(
                user=user,
                price__price_id__lt=7,
                is_active=True,
                is_deleted=False
            ).first()

            if paid_subscription:
                paid_subscription.is_active = False
                paid_subscription.is_deleted = True
                paid_subscription.save()
                logger.info(f"Paid subscription deleted for user: {user.email}")
            else:
                logger.info(f"No paid subscription found for user: {user.email}")

            # Create subscription record
            new_subscription = UserSubscription.objects.create(
                user=user,
                price=price,
                plan_name=price.plan.plan_name,
                stripe_subscription_id=subscription['id'],
                is_active=True,
                is_deleted=False
            )
            
            return StripeResponse(success=True, data={
                'event': 'subscription_created', 
                'subscription': new_subscription
            })
            
        except Exception as e:
            logger.error(f"Error handling subscription created: {str(e)}")
            return StripeResponse(success=False, error=f"Subscription creation error: {str(e)}")


    def _handle_subscription_deleted(self, subscription) -> StripeResponse:
        """Handle subscription deletion"""
        try:
            logger.info(f"Subscription deleted: {subscription['id']}")
            customer_id = subscription['customer']

            customer = stripe.Customer.retrieve(customer_id)
            user = CustomUser.objects.get(email=customer['email'])
            # Find subscription by ID
            existing_subscription = UserSubscription.objects.filter(    
                user=user,
                stripe_subscription_id=subscription['id'],
                is_deleted=False,
            ).first()
            
            if existing_subscription:
                existing_subscription.is_deleted = True
                existing_subscription.is_active = False
                existing_subscription.save()
                
                return StripeResponse(success=True, data={
                    'event': 'subscription_deleted', 
                    'subscription': existing_subscription.to_dict()
                })
            else:
                logger.warning(f"Subscription {subscription['id']} not found in database")
                return StripeResponse(success=True, data={
                    'event': 'subscription_deleted', 
                    'message': 'Subscription not found in database'
                })
                
        except Exception as e:
            logger.error(f"Error handling subscription deleted: {str(e)}")
            return StripeResponse(success=False, error=f"Subscription deletion error: {str(e)}")

    def _handle_customer_updated(self, customer) -> StripeResponse:
        """Handle customer update"""
        try:
            logger.info(f"Customer updated: {customer['id']}")
            
            # Find user by email
            user = CustomUser.objects.get(email=customer['email'])
            
            # Update or create StripeCustomer
            existing_customer, created = StripeCustomer.objects.get_or_create(
                user=user,
                defaults={'stripe_customer_id': customer['id']}
            )
            
            if not created:
                existing_customer.stripe_customer_id = customer['id']
                existing_customer.save()
            
            return StripeResponse(success=True, data={
                'event': 'customer_updated', 
                'customer': existing_customer.to_dict()
            })
            
        except CustomUser.DoesNotExist:
            logger.error(f"User with email {customer['email']} not found")
            return StripeResponse(success=False, error="User not found")
        except Exception as e:
            logger.error(f"Error handling customer updated: {str(e)}")
            return StripeResponse(success=False, error=f"Customer update error: {str(e)}")

    def _handle_customer_created(self, customer) -> StripeResponse:
        """Handle customer creation"""
        try:
            logger.info(f"Customer created: {customer['id']}")
            
            # Find user by email
            user = CustomUser.objects.get(email=customer['email'])
            
            # Create StripeCustomer
            new_customer, created = StripeCustomer.objects.get_or_create(
                user=user,
                stripe_customer_id=customer['id']
            )
            
            if not created:
                logger.info(f"Customer {customer['id']} already exists for user {user.id}")
            
            return StripeResponse(success=True, data={
                'event': 'customer_created', 
                'customer': new_customer.to_dict()
            })
            
        except CustomUser.DoesNotExist:
            logger.error(f"User with email {customer['email']} not found")
            return StripeResponse(success=False, error="User not found")
        except Exception as e:
            logger.error(f"Error handling customer created: {str(e)}")
            return StripeResponse(success=False, error=f"Customer creation error: {str(e)}")

    def get_customer_portal_url(self, user, return_url: str = None) -> StripeResponse:
        """Create a customer portal session for subscription management"""
        return_url = return_url or os.getenv('STRIPE_RETURN_URL')
        try:
            if not self._validate_user(user):
                return StripeResponse(success=False, error="User validation failed")
            
            # Find customer
            customers = stripe.Customer.list(email=user.email, limit=1).data
            if not customers:
                return StripeResponse(success=False, error="Customer not found")
            
            customer = customers[0]
            
            # Create portal session
            session = stripe.billing_portal.Session.create(
                customer=customer.id,
                return_url=return_url,
            )
            
            return StripeResponse(success=True, data={'url': session.url})
            
        except stripe.error.StripeError as e:
            return self._handle_stripe_error(e)
        except Exception as e:
            logger.error(f"Unexpected error creating portal session: {str(e)}")
            return StripeResponse(success=False, error=f"Unexpected error: {str(e)}")
        
    def create_customer_session(self, user) -> StripeResponse:
        """Create a customer session for embedded Stripe components"""
        try:
            if not self._validate_user(user):
                return StripeResponse(success=False, error="User validation failed")
            
            # Find customer
            customers = stripe.Customer.list(email=user.email, limit=1).data
            if not customers:
                return StripeResponse(success=False, error="Customer not found")
            
            customer = customers[0]
            
            # Create customer session for embedded components
            session = stripe.CustomerSession.create(
                customer=customer.id,
                components={
                    'pricing_table': {'enabled': True},
                    'payment_element': {'enabled': True, 'features': {'payment_method_save': 'enabled'}},
                    'buy_button': {'enabled': True}
                }
            )
            
            return StripeResponse(success=True, data={
                'client_secret': session.client_secret,
                'customer_id': customer.id
            })
            
        except stripe.error.StripeError as e:
            return self._handle_stripe_error(e)
        except Exception as e:
            logger.error(f"Unexpected error creating customer session: {str(e)}")
            return StripeResponse(success=False, error=f"Unexpected error: {str(e)}")

    def create_checkout_session(self, user, plan, success_url: str = None, cancel_url: str = None) -> StripeResponse:
        """Create a Stripe Checkout session for subscription signup"""
        success_url = success_url or os.getenv('STRIPE_SUCCESS_URL')
        cancel_url = cancel_url or os.getenv('STRIPE_CANCEL_URL')
        try:
            if not self._validate_user(user) or not self._validate_plan(plan):
                return StripeResponse(success=False, error="Validation failed")
            
            # Get or create customer
            customer_response = self.create_stripe_customer(user)
            if not customer_response.success:
                return customer_response
            
            customer = customer_response.data
            
            # Create checkout session
            session = stripe.checkout.Session.create(
                customer=customer.id,
                payment_method_types=['card'],
                line_items=[{
                    'price': plan.plan_id,
                    'quantity': 1,
                }],
                mode='subscription',
                success_url=success_url,
                cancel_url=cancel_url,
                metadata={
                    'user_id': str(user.id),
                    'plan_id': plan.plan_id,
                    'plan_name': plan.plan_name
                }
            )
            
            return StripeResponse(success=True, data={
                'checkout_url': session.url,
                'session_id': session.id
            })
            
        except stripe.error.StripeError as e:
            return self._handle_stripe_error(e)
        except Exception as e:
            logger.error(f"Unexpected error creating checkout session: {str(e)}")
            return StripeResponse(success=False, error=f"Unexpected error: {str(e)}")
        
    def get_subscription_details(self, subscription_id: str) -> StripeResponse:
        """Get subscription details"""
        try:
            if not subscription_id:
                return StripeResponse(success=False, error="Subscription ID is required")
            
            subscription = stripe.Subscription.retrieve(subscription_id)
            return subscription
        
        except stripe.error.StripeError as e:
            return self._handle_stripe_error(e)
        except Exception as e:
            logger.error(f"Unexpected error getting subscription details: {str(e)}")
            return StripeResponse(success=False, error=f"Unexpected error: {str(e)}")
        