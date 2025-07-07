import os
import requests
import json
from django.core.management.base import BaseCommand
from django.urls import reverse
from django.conf import settings


class Command(BaseCommand):
    help = 'Diagnose and fix webhook 404 errors'

    def add_arguments(self, parser):
        parser.add_argument(
            '--ngrok-url',
            type=str,
            help='Your ngrok URL (e.g., https://abc123.ngrok-free.app)',
            default="https://9c4e-196-137-69-71.ngrok-free.app"
        )
        parser.add_argument(
            '--test-urls',
            action='store_true',
            help='Test webhook URLs accessibility'
        )
        parser.add_argument(
            '--show-correct-url',
            action='store_true',
            help='Show the correct webhook URL for Stripe configuration'
        )
        parser.add_argument(
            '--test-webhook-endpoint',
            action='store_true',
            help='Test webhook endpoint with a sample payload'
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('🔍 Webhook 404 Diagnosis Tool'))
        self.stdout.write('=' * 50)
        
        ngrok_url = options.get('ngrok_url', "https://9c4e-196-137-69-71.ngrok-free.app")
        
        if options['show_correct_url']:
            self.show_correct_webhook_url(ngrok_url)
        
        if options['test_urls']:
            self.test_webhook_urls(ngrok_url)
        
        if options['test_webhook_endpoint']:
            self.test_webhook_endpoint_locally()
        
        # Always show the diagnosis summary
        self.show_diagnosis_summary(ngrok_url)

    def show_correct_webhook_url(self, ngrok_url):
        """Show the correct webhook URL that should be configured in Stripe"""
        self.stdout.write('\n' + self.style.SUCCESS('🎯 Correct Webhook URL Configuration'))
        self.stdout.write('-' * 50)
        
        # Get the correct URL path
        try:
            webhook_path = reverse('plans:webhook')
            correct_url = f"{ngrok_url.rstrip('/')}{webhook_path}"
            
            self.stdout.write(f'✓ Correct webhook URL: {correct_url}')
            self.stdout.write('')
            self.stdout.write(self.style.WARNING('Configure this URL in Stripe Dashboard:'))
            self.stdout.write(f'  1. Go to https://dashboard.stripe.com/webhooks')
            self.stdout.write(f'  2. Click on your webhook endpoint')
            self.stdout.write(f'  3. Update the URL to: {correct_url}')
            self.stdout.write(f'  4. Make sure to include the trailing slash')
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error getting webhook URL: {e}'))

    def test_webhook_urls(self, ngrok_url):
        """Test if webhook URLs are accessible"""
        self.stdout.write('\n' + self.style.SUCCESS('🔍 Testing URL Accessibility'))
        self.stdout.write('-' * 50)
        
        # Test various URL patterns
        test_paths = [
            '/',
            '/plans/',
            '/plans/webhook/',
            '/admin/',
        ]
        
        for path in test_paths:
            full_url = f"{ngrok_url.rstrip('/')}{path}"
            self.stdout.write(f'Testing: {full_url}')
            
            try:
                response = requests.get(full_url, timeout=10)
                status_icon = '✓' if response.status_code < 400 else '✗'
                self.stdout.write(f'  {status_icon} {response.status_code} - {response.reason}')
                
                # For webhook, also test POST
                if 'webhook' in path:
                    try:
                        post_response = requests.post(
                            full_url,
                            json={"test": "data"},
                            headers={
                                "Content-Type": "application/json",
                                "User-Agent": "Stripe/1.0"
                            },
                            timeout=10
                        )
                        post_icon = '✓' if post_response.status_code < 500 else '✗'
                        self.stdout.write(f'  {post_icon} POST {post_response.status_code} - {post_response.reason}')
                    except Exception as e:
                        self.stdout.write(f'  ✗ POST failed: {e}')
                
            except Exception as e:
                self.stdout.write(f'  ✗ Request failed: {e}')
            
            self.stdout.write('')

    def test_webhook_endpoint_locally(self):
        """Test webhook endpoint locally"""
        self.stdout.write('\n' + self.style.SUCCESS('🧪 Testing Webhook Endpoint Locally'))
        self.stdout.write('-' * 50)
        
        try:
            from plans.views import webhook_raw
            from django.test import RequestFactory
            
            factory = RequestFactory()
            
            # Create a test request
            test_payload = json.dumps({
                "id": "evt_test_webhook",
                "object": "event",
                "type": "test.event",
                "data": {"object": {"id": "test"}}
            })
            
            request = factory.post(
                '/plans/webhook/',
                data=test_payload,
                content_type='application/json',
                HTTP_STRIPE_SIGNATURE='t=1234567890,v1=test_signature'
            )
            
            # This will likely fail signature verification, but should not 404
            response = webhook_raw(request)
            
            self.stdout.write(f'✓ Webhook endpoint is accessible locally')
            self.stdout.write(f'  Response status: {response.status_code}')
            
            if response.status_code == 400:
                self.stdout.write(f'  ✓ Expected 400 (signature verification failure)')
            elif response.status_code == 404:
                self.stdout.write(f'  ✗ Unexpected 404 - URL routing issue')
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'✗ Error testing webhook endpoint: {e}'))

    def show_diagnosis_summary(self, ngrok_url):
        """Show diagnosis summary and next steps"""
        self.stdout.write('\n' + self.style.SUCCESS('📋 Diagnosis Summary'))
        self.stdout.write('-' * 50)
        
        # Check if Django server is likely running
        try:
            response = requests.get(f"{ngrok_url}/admin/", timeout=5)
            if response.status_code in [200, 302, 301]:
                self.stdout.write('✓ Django server appears to be running')
            else:
                self.stdout.write(f'⚠ Django server response: {response.status_code}')
        except:
            self.stdout.write('✗ Django server not accessible via ngrok')
        
        self.stdout.write('')
        self.stdout.write(self.style.WARNING('Most Common Issues:'))
        self.stdout.write('1. Webhook URL in Stripe Dashboard is incorrect')
        self.stdout.write('2. Django server is not running')
        self.stdout.write('3. ngrok is not forwarding to the correct port')
        self.stdout.write('4. URL path is missing trailing slash')
        
        self.stdout.write('')
        self.stdout.write(self.style.SUCCESS('Quick Fix Commands:'))
        self.stdout.write('1. Make sure Django is running:')
        self.stdout.write('   python manage.py runserver 8000')
        self.stdout.write('')
        self.stdout.write('2. Make sure ngrok is forwarding correctly:')
        self.stdout.write('   ngrok http 8000')
        self.stdout.write('')
        self.stdout.write('3. Update Stripe webhook URL to:')
        webhook_url = f"{ngrok_url.rstrip('/')}/plans/webhook/"
        self.stdout.write(f'   {webhook_url}')
        
        self.stdout.write('')
        self.stdout.write(self.style.SUCCESS('Next Steps:'))
        self.stdout.write('1. Run: python manage.py fix_webhook_404 --test-urls')
        self.stdout.write('2. Fix any issues found')
        self.stdout.write('3. Update Stripe webhook URL')
        self.stdout.write('4. Test webhook delivery again') 