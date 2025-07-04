import json
import os
import stripe
import time
import hashlib
import hmac
from django.core.management.base import BaseCommand
from plans.payment.stripe_gateway import StripeGateway


class Command(BaseCommand):
    help = 'Test webhook signature verification and configuration'

    def add_arguments(self, parser):
        parser.add_argument(
            '--check-config',
            action='store_true',
            help='Check webhook configuration'
        )
        parser.add_argument(
            '--test-payload',
            type=str,
            help='Test with a sample payload'
        )
        parser.add_argument(
            '--test-signature',
            type=str,
            help='Test with a sample signature'
        )
        parser.add_argument(
            '--create-test-event',
            action='store_true',
            help='Create a test webhook event using Stripe API'
        )
        parser.add_argument(
            '--generate-test-signature',
            action='store_true',
            help='Generate a test signature for sample payload'
        )
        parser.add_argument(
            '--validate-secret',
            action='store_true',
            help='Validate webhook secret format and connectivity'
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Testing Webhook Configuration'))
        
        # Initialize gateway
        try:
            gateway = StripeGateway()
            self.stdout.write(self.style.SUCCESS('✓ StripeGateway initialized successfully'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'✗ Failed to initialize StripeGateway: {e}'))
            return

        if options['check_config']:
            self.check_configuration(gateway)
            
        if options['validate_secret']:
            self.validate_webhook_secret(gateway)
            
        if options['generate_test_signature']:
            self.generate_test_signature(gateway)
            
        if options['create_test_event']:
            self.create_test_webhook_event(gateway)
            
        if options['test_payload'] and options['test_signature']:
            self.test_signature_verification(
                gateway, 
                options['test_payload'], 
                options['test_signature']
            )
        elif options['test_payload'] or options['test_signature']:
            self.stdout.write(
                self.style.WARNING('Both --test-payload and --test-signature are required for testing')
            )

    def check_configuration(self, gateway):
        """Check basic configuration"""
        self.stdout.write('\n' + '='*50)
        self.stdout.write('CONFIGURATION CHECK')
        self.stdout.write('='*50)
        
        # Check environment variables
        stripe_secret = os.getenv('STRIPE_SECRET_KEY')
        webhook_secret = os.getenv('STRIPE_WEBHOOK_SECRET')
        publishable_key = os.getenv('STRIPE_PUBLISHABLE_KEY')
        
        self.stdout.write(f'Stripe Secret Key: {"✓ Present" if stripe_secret else "✗ Missing"}')
        if stripe_secret:
            self.stdout.write(f'  Format: {"✓ Valid" if stripe_secret.startswith("sk_") else "✗ Invalid"}')
            self.stdout.write(f'  Length: {len(stripe_secret)} chars')
        
        self.stdout.write(f'Webhook Secret: {"✓ Present" if webhook_secret else "✗ Missing"}')
        if webhook_secret:
            self.stdout.write(f'  Format: {"✓ Valid" if webhook_secret.startswith("whsec_") else "✗ Invalid"}')
            self.stdout.write(f'  Length: {len(webhook_secret)} chars')
        
        self.stdout.write(f'Publishable Key: {"✓ Present" if publishable_key else "✗ Missing"}')
        if publishable_key:
            self.stdout.write(f'  Format: {"✓ Valid" if publishable_key.startswith("pk_") else "✗ Invalid"}')
        
        # Test Stripe API connection
        try:
            stripe.api_key = gateway.stripe_api_key
            account = stripe.Account.retrieve()
            self.stdout.write(self.style.SUCCESS('✓ Stripe API connection: Working'))
            self.stdout.write(f'  Account ID: {account.id}')
            self.stdout.write(f'  Country: {account.country}')
            self.stdout.write(f'  Business Type: {account.business_type}')
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'✗ Stripe API connection: {e}'))

    def validate_webhook_secret(self, gateway):
        """Validate webhook secret format and basic functionality"""
        self.stdout.write('\n' + '='*50)
        self.stdout.write('WEBHOOK SECRET VALIDATION')
        self.stdout.write('='*50)
        
        if not gateway.stripe_webhook_secret:
            self.stdout.write(self.style.ERROR('✗ Webhook secret not configured'))
            return
        
        secret = gateway.stripe_webhook_secret
        
        # Check format
        if secret.startswith('whsec_'):
            self.stdout.write(self.style.SUCCESS('✓ Webhook secret format: Valid'))
        else:
            self.stdout.write(self.style.ERROR('✗ Webhook secret format: Invalid (should start with whsec_)'))
            return
        
        # Check length (Stripe webhook secrets are typically 32+ chars after whsec_)
        if len(secret) > 32:
            self.stdout.write(self.style.SUCCESS(f'✓ Webhook secret length: {len(secret)} chars (Valid)'))
        else:
            self.stdout.write(self.style.WARNING(f'⚠ Webhook secret length: {len(secret)} chars (Suspiciously short)'))
        
        self.stdout.write(f'Secret preview: {secret[:15]}...')

    def generate_test_signature(self, gateway):
        """Generate a test signature for a sample payload"""
        self.stdout.write('\n' + '='*50)
        self.stdout.write('GENERATE TEST SIGNATURE')
        self.stdout.write('='*50)
        
        if not gateway.stripe_webhook_secret:
            self.stdout.write(self.style.ERROR('✗ Cannot generate signature: Webhook secret not configured'))
            return
        
        # Sample test payload
        test_payload = json.dumps({
            "id": "evt_test_webhook",
            "object": "event",
            "type": "customer.created",
            "created": int(time.time()),
            "data": {
                "object": {
                    "id": "cus_test_customer",
                    "object": "customer",
                    "email": "test@example.com"
                }
            }
        })
        
        # Generate signature
        timestamp = str(int(time.time()))
        payload_bytes = test_payload.encode('utf-8')
        
        # Create signature the same way Stripe does
        secret_bytes = gateway.stripe_webhook_secret.encode('utf-8')
        signed_payload = f"{timestamp}.{test_payload}".encode('utf-8')
        signature = hmac.new(secret_bytes, signed_payload, hashlib.sha256).hexdigest()
        
        stripe_signature = f"t={timestamp},v1={signature}"
        
        self.stdout.write(f'Generated test payload ({len(test_payload)} bytes):')
        self.stdout.write(test_payload)
        self.stdout.write(f'\nGenerated signature:')
        self.stdout.write(stripe_signature)
        
        # Test the generated signature
        self.stdout.write(f'\nTesting generated signature...')
        success, error_message, event = gateway._verify_webhook_signature(payload_bytes, stripe_signature)
        
        if success:
            self.stdout.write(self.style.SUCCESS('✓ Generated signature verification: PASSED'))
        else:
            self.stdout.write(self.style.ERROR(f'✗ Generated signature verification: FAILED - {error_message}'))

    def create_test_webhook_event(self, gateway):
        """Create a test webhook event using Stripe API"""
        self.stdout.write('\n' + '='*50)
        self.stdout.write('CREATE TEST WEBHOOK EVENT')
        self.stdout.write('='*50)
        
        try:
            # Create a test event
            test_event = stripe.Event.construct_from({
                "id": "evt_test_webhook",
                "object": "event",
                "type": "customer.created",
                "created": int(time.time()),
                "data": {
                    "object": {
                        "id": "cus_test_customer",
                        "object": "customer",
                        "email": "test@example.com",
                        "created": int(time.time())
                    }
                },
                "livemode": False,
                "pending_webhooks": 1,
                "request": {
                    "id": None,
                    "idempotency_key": None
                }
            }, stripe.api_key)
            
            self.stdout.write(self.style.SUCCESS(f'✓ Test event created: {test_event.id}'))
            self.stdout.write(f'Event type: {test_event.type}')
            self.stdout.write(f'Event created: {test_event.created}')
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'✗ Failed to create test event: {e}'))

    def test_signature_verification(self, gateway, payload, signature):
        """Test signature verification with provided payload and signature"""
        self.stdout.write('\n' + '='*50)
        self.stdout.write('SIGNATURE VERIFICATION TEST')
        self.stdout.write('='*50)
        
        self.stdout.write(f'Payload length: {len(payload)}')
        self.stdout.write(f'Payload type: {type(payload)}')
        self.stdout.write(f'Signature: {signature}')
        
        # Parse signature components
        sig_elements = signature.split(',')
        timestamp = None
        signatures = []
        
        for element in sig_elements:
            if element.startswith('t='):
                timestamp = element[2:]
            elif element.startswith('v1='):
                signatures.append(element[3:])
        
        self.stdout.write(f'Timestamp: {timestamp}')
        self.stdout.write(f'Signatures: {len(signatures)} found')
        
        if timestamp:
            sig_time = int(timestamp)
            current_time = int(time.time())
            age = current_time - sig_time
            self.stdout.write(f'Signature age: {age} seconds')
        
        # Convert payload to bytes if needed
        if isinstance(payload, str):
            payload_bytes = payload.encode('utf-8')
        else:
            payload_bytes = payload
        
        # Test the verification
        success, error_message, event = gateway._verify_webhook_signature(payload_bytes, signature)
        
        if success:
            self.stdout.write(self.style.SUCCESS('✓ Signature verification: PASSED'))
            self.stdout.write(f'Event type: {event.get("type", "Unknown")}')
            self.stdout.write(f'Event ID: {event.get("id", "Unknown")}')
        else:
            self.stdout.write(self.style.ERROR('✗ Signature verification: FAILED'))
            self.stdout.write(f'Error: {error_message}')
            
            # Provide troubleshooting suggestions
            self.stdout.write('\n' + '-'*30)
            self.stdout.write('Troubleshooting Suggestions:')
            self.stdout.write('-'*30)
            self.stdout.write('1. Verify webhook secret is correct (starts with whsec_)')
            self.stdout.write('2. Check that payload is raw/unmodified')
            self.stdout.write('3. Ensure signature header is from Stripe')
            self.stdout.write('4. Verify webhook endpoint URL in Stripe dashboard')
            self.stdout.write('5. Check system clock synchronization')
            self.stdout.write('6. Try using --generate-test-signature to test with known good signature') 