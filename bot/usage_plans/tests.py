from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APIClient
from .models import SubscriptionPlan, UserSubscription, PlanFeature
from django.utils import timezone
from datetime import timedelta
import json

class SubscriptionPlanAPITests(TestCase):
    def setUp(self):
        """Set up test data for API tests"""
        self.client = APIClient()
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        self.client.force_authenticate(user=self.user)
        
        # Create test plan and feature
        self.plan = SubscriptionPlan.objects.create(
            plan_name="Test Plan",
            plan_price=10.00,
            plan_period="monthly",
            is_active=True
        )
        self.feature = PlanFeature.objects.create(
            plan=self.plan,
            feature_name="Test Feature",
            feature_limit=100
        )
        
        self.plan_data = {
            'plan_name': 'New Plan',
            'plan_price': 20.00,
            'plan_period': 'monthly',
            'is_active': True
        }

    def test_get_plans_list(self):
        """Test retrieving list of subscription plans"""
        url = reverse('subscription-plan-list')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 1)
        
        # Verify response data
        plan = response.data[0]
        self.assertEqual(plan['plan_name'], 'Test Plan')
        self.assertEqual(float(plan['plan_price']), 10.00)
        self.assertEqual(plan['plan_period'], 'monthly')

    def test_get_plan_detail(self):
        """Test retrieving specific plan details"""
        url = reverse('subscription-plan-detail', args=[self.plan.plan_id])
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        # Verify response data
        self.assertEqual(response.data['plan_name'], 'Test Plan')
        self.assertEqual(float(response.data['plan_price']), 10.00)
        self.assertEqual(response.data['plan_period'], 'monthly')

    def test_create_plan(self):
        """Test creating a new subscription plan"""
        url = reverse('subscription-plan-list')
        response = self.client.post(url, self.plan_data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertTrue(SubscriptionPlan.objects.filter(plan_name='New Plan').exists())
        
        # Verify response data
        self.assertEqual(response.data['plan_name'], 'New Plan')
        self.assertEqual(float(response.data['plan_price']), 20.00)
        self.assertEqual(response.data['plan_period'], 'monthly')

    def test_update_plan(self):
        """Test updating a subscription plan"""
        url = reverse('subscription-plan-detail', args=[self.plan.plan_id])
        update_data = self.plan_data.copy()
        update_data['plan_name'] = 'Updated Plan'
        
        response = self.client.put(url, update_data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        # Verify update
        self.plan.refresh_from_db()
        self.assertEqual(self.plan.plan_name, 'Updated Plan')
        self.assertEqual(float(self.plan.plan_price), 20.00)

    def test_delete_plan(self):
        """Test deleting a subscription plan"""
        url = reverse('subscription-plan-detail', args=[self.plan.plan_id])
        response = self.client.delete(url)
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)
        self.assertFalse(SubscriptionPlan.objects.filter(plan_id=self.plan.plan_id).exists())

    def test_get_user_subscriptions(self):
        """Test retrieving user's subscriptions"""
        # Create test subscription
        subscription = UserSubscription.objects.create(
            user=self.user,
            plan=self.plan,
            stripe_subscription_id="sub_1234567890"
        )
        
        url = reverse('user-subscription-list')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 1)
        
        # Verify response data
        sub = response.data[0]
        self.assertEqual(sub['plan']['plan_name'], 'Test Plan')
        self.assertEqual(sub['stripe_subscription_id'], 'sub_1234567890')

    def test_create_user_subscription(self):
        """Test creating a new user subscription"""
        url = reverse('user-subscription-list')
        data = {
            'plan': self.plan.plan_id,
            'stripe_subscription_id': 'sub_new123'
        }
        
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertTrue(UserSubscription.objects.filter(stripe_subscription_id='sub_new123').exists())

    def test_unauthorized_access(self):
        """Test unauthorized access to subscription endpoints"""
        self.client.force_authenticate(user=None)
        
        # Test plan list endpoint
        url = reverse('subscription-plan-list')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
        
        # Test subscription list endpoint
        url = reverse('user-subscription-list')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_invalid_plan_data(self):
        """Test API with invalid plan data"""
        url = reverse('subscription-plan-list')
        invalid_data = {
            'plan_name': None,  # Invalid data
            'plan_price': -10.00,  # Invalid price
            'plan_period': 'invalid'  # Invalid period
        }
        
        response = self.client.post(url, invalid_data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
