import pytest
from django.test import TestCase
from django.contrib.auth import get_user_model
from django.urls import reverse
from rest_framework.test import APITestCase
from rest_framework import status
from unittest.mock import patch, MagicMock
import stripe

from plans.models import (
    SubscriptionPlan, 
    PlanFeature, 
    UserSubscription, 
    StripeCustomer,
    PlanPrice
)
from plans.payment.stripe_gateway import StripeGateway, StripeResponse

User = get_user_model()


class SubscriptionPlanModelTest(TestCase):
    """Test cases for SubscriptionPlan model"""
    
    def setUp(self):
        self.plan_data = {
            'plan_name': 'Test Plan',
            'plan_description': 'A test plan',
            'is_active': True
        }
        self.plan = SubscriptionPlan.objects.create(**self.plan_data)
        # Create a price for the plan
        self.plan_price = PlanPrice.objects.create(
            plan=self.plan,
            price_amount=29.99,
            price_currency='USD',
            price_period='monthly'
        )
    
    def test_plan_creation(self):
        """Test that a plan can be created successfully"""
        self.assertEqual(self.plan.plan_name, 'Test Plan')
        self.assertTrue(self.plan.is_active)
    
    def test_plan_str_representation(self):
        """Test the string representation of a plan"""
        expected = "Test Plan"
        self.assertEqual(str(self.plan), expected)
    
    def test_plan_to_dict(self):
        """Test the to_dict method returns correct data"""
        plan_dict = self.plan.to_dict()
        self.assertEqual(plan_dict['plan_name'], 'Test Plan')
        self.assertTrue(plan_dict['is_active'])


class PlanFeatureModelTest(TestCase):
    """Test cases for PlanFeature model"""
    
    def setUp(self):
        self.plan = SubscriptionPlan.objects.create(
            plan_name='Test Plan'
        )
        self.feature_data = {
            'plan': self.plan,
            'feature_name': 'tokens',
            'feature_type': 'limit',
            'feature_description': 'Monthly token limit',
            'feature_limit': 1000,
            'is_active': True
        }
        self.feature = PlanFeature.objects.create(**self.feature_data)
    
    def test_feature_creation(self):
        """Test that a feature can be created successfully"""
        self.assertEqual(self.feature.feature_name, 'tokens')
        self.assertEqual(self.feature.feature_limit, 1000)
        self.assertTrue(self.feature.is_active)
    
    def test_feature_str_representation(self):
        """Test the string representation of a feature"""
        expected = "tokens (Test Plan)"
        self.assertEqual(str(self.feature), expected)
    
    def test_feature_to_dict(self):
        """Test the to_dict method returns correct data"""
        feature_dict = self.feature.to_dict()
        self.assertEqual(feature_dict['feature_name'], 'tokens')
        self.assertEqual(feature_dict['feature_type'], 'limit')
        self.assertEqual(feature_dict['feature_limit'], 1000)
        self.assertTrue(feature_dict['is_active'])


class UserSubscriptionModelTest(TestCase):
    """Test cases for UserSubscription model"""
    
    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        self.plan = SubscriptionPlan.objects.create(
            plan_name='Test Plan'
        )
        self.stripe_customer = StripeCustomer.objects.create(
            user=self.user,
            stripe_customer_id='cus_test123'
        )
        self.subscription_data = {
            'user': self.user,
            'plan': self.plan,
            'subscription_id': 'sub_test123',
            'stripe_customer_id': self.stripe_customer,
            'is_active': True
        }
        self.subscription = UserSubscription.objects.create(**self.subscription_data)
    
    def test_subscription_creation(self):
        """Test that a subscription can be created successfully"""
        self.assertEqual(self.subscription.user, self.user)
        self.assertEqual(self.subscription.plan, self.plan)
        self.assertEqual(self.subscription.subscription_id, 'sub_test123')
        self.assertTrue(self.subscription.is_active)
    
    def test_subscription_usage_fields(self):
        """Test that usage fields are properly initialized"""
        self.assertEqual(self.subscription.tokens_usage, 0)
        self.assertEqual(self.subscription.conversations_usage, 0)
        self.assertEqual(self.subscription.agents_usage, 0)
        self.assertEqual(self.subscription.integrations_usage, 0)


class StripeGatewayTest(TestCase):
    """Test cases for StripeGateway class"""
    
    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        self.plan = SubscriptionPlan.objects.create(
            plan_name='Test Plan'
        )
        self.stripe_gateway = StripeGateway()
    
    @patch('stripe.Customer.create')
    def test_create_stripe_customer_success(self, mock_create):
        """Test successful customer creation"""
        mock_customer = MagicMock()
        mock_customer.id = 'cus_test123'
        mock_customer.email = 'test@example.com'
        mock_create.return_value = mock_customer
        
        response = self.stripe_gateway.create_stripe_customer(self.user)
        
        self.assertTrue(response.success)
        self.assertEqual(response.data['stripe_customer_id'], 'cus_test123')
        mock_create.assert_called_once_with(
            email='test@example.com',
            metadata={'user_id': str(self.user.id)}
        )
    
    @patch('stripe.Customer.create')
    def test_create_stripe_customer_failure(self, mock_create):
        """Test customer creation failure"""
        mock_create.side_effect = stripe.error.StripeError("Stripe error")
        
        response = self.stripe_gateway.create_stripe_customer(self.user)
        
        self.assertFalse(response.success)
        self.assertIn("Stripe error", response.error)
    
    def test_validate_user_success(self):
        """Test user validation success"""
        result = self.stripe_gateway._validate_user(self.user)
        self.assertTrue(result)
    
    def test_validate_user_failure(self):
        """Test user validation failure"""
        result = self.stripe_gateway._validate_user(None)
        self.assertFalse(result)
    
    def test_validate_plan_success(self):
        """Test plan validation success"""
        result = self.stripe_gateway._validate_plan(self.plan)
        self.assertTrue(result)
    
    def test_validate_plan_failure(self):
        """Test plan validation failure"""
        result = self.stripe_gateway._validate_plan(None)
        self.assertFalse(result)


class StripeResponseTest(TestCase):
    """Test cases for StripeResponse class"""
    
    def test_successful_response(self):
        """Test successful response creation"""
        data = {'test': 'data'}
        response = StripeResponse(success=True, data=data)
        
        self.assertTrue(response.success)
        self.assertEqual(response.data, data)
        self.assertIsNone(response.error)
    
    def test_error_response(self):
        """Test error response creation"""
        error_message = "Something went wrong"
        response = StripeResponse(success=False, error=error_message)
        
        self.assertFalse(response.success)
        self.assertEqual(response.error, error_message)
        self.assertIsNone(response.data)


class SubscriptionViewsTest(APITestCase):
    """Test cases for subscription views"""
    
    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        self.plan = SubscriptionPlan.objects.create(
            plan_name='Test Plan'
        )
        self.client.force_authenticate(user=self.user)
    
    @patch('usage_plans.views.StripeGateway')
    def test_create_checkout_session_success(self, mock_stripe_gateway):
        """Test successful checkout session creation"""
        mock_gateway = MagicMock()
        mock_gateway.checkout_session.return_value = StripeResponse(
            success=True,
            data={'checkout_url': 'https://checkout.stripe.com/test'}
        )
        mock_stripe_gateway.return_value = mock_gateway
        
        url = reverse('create_checkout_session')
        data = {
            'plan_id': 'test_plan_001',
            'success_url': 'https://example.com/success',
            'cancel_url': 'https://example.com/cancel'
        }
        

