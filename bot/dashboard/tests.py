from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.utils import timezone
from django.urls import reverse
from django.core.cache import cache
from dashboard.models import Dashboard
from conversations.models import Conversation
from create_agent.models import Agent
import pytest
import json

class DashboardAPITestCase(TestCase):
    @classmethod
    def setUpTestData(cls):
        # Create test data that will be used by all test methods
        cls.user = User.objects.create_user(
            username='testuser',
            password='testpassword',
            email='test@example.com'
        )
        cls.dashboard = Dashboard.objects.create(
            user=cls.user,
            dashboard_type='test_dashboard',
            is_active=True
        )
        cls.client = Client()
        cls.client.login(username='testuser', password='testpassword')

    def setUp(self):
        cache.clear()
        self.dashboard.refresh_from_db()

    def tearDown(self):
        cache.clear()

    def test_dashboard_api_list(self):
        """Test retrieving dashboard list via API"""
        url = reverse('dashboard-list')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['dashboard_type'], 'test_dashboard')

    def test_dashboard_api_detail(self):
        """Test retrieving specific dashboard details via API"""
        url = reverse('dashboard-detail', args=[self.dashboard.dashboard_id])
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertEqual(data['dashboard_type'], 'test_dashboard')
        self.assertTrue(data['is_active'])

    def test_dashboard_api_create(self):
        """Test creating a new dashboard via API"""
        url = reverse('dashboard-list')
        data = {
            'dashboard_type': 'new_dashboard',
            'is_active': True
        }
        response = self.client.post(url, data, content_type='application/json')
        self.assertEqual(response.status_code, 201)
        self.assertTrue(Dashboard.objects.filter(dashboard_type='new_dashboard').exists())

    def test_dashboard_api_update(self):
        """Test updating dashboard via API"""
        url = reverse('dashboard-detail', args=[self.dashboard.dashboard_id])
        data = {
            'dashboard_type': 'updated_dashboard',
            'is_active': False
        }
        response = self.client.patch(url, data, content_type='application/json')
        self.assertEqual(response.status_code, 200)
        self.dashboard.refresh_from_db()
        self.assertEqual(self.dashboard.dashboard_type, 'updated_dashboard')
        self.assertFalse(self.dashboard.is_active)

    def test_dashboard_api_delete(self):
        """Test deleting dashboard via API"""
        url = reverse('dashboard-detail', args=[self.dashboard.dashboard_id])
        response = self.client.delete(url)
        self.assertEqual(response.status_code, 204)
        self.assertFalse(Dashboard.objects.filter(dashboard_id=self.dashboard.dashboard_id).exists())

    def test_dashboard_api_stats(self):
        """Test retrieving dashboard statistics via API"""
        # Create test conversation
        Conversation.objects.create(
            user=self.user,
            is_active=True,
            number_of_agent_messages=5
        )
        
        url = reverse('dashboard-stats', args=[self.dashboard.dashboard_id])
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        
        # Verify stats data structure
        self.assertIn('total_conversations', data)
        self.assertIn('active_conversations', data)
        self.assertIn('total_responses', data)
        self.assertEqual(data['total_conversations'], 1)
        self.assertEqual(data['total_responses'], 5)

    def test_dashboard_api_unauthorized(self):
        """Test unauthorized access to dashboard API"""
        self.client.logout()
        
        # Test list endpoint
        url = reverse('dashboard-list')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 401)
        
        # Test detail endpoint
        url = reverse('dashboard-detail', args=[self.dashboard.dashboard_id])
        response = self.client.get(url)
        self.assertEqual(response.status_code, 401)

    def test_dashboard_api_invalid_data(self):
        """Test API with invalid data"""
        url = reverse('dashboard-list')
        data = {
            'dashboard_type': None,  # Invalid data
            'is_active': True
        }
        response = self.client.post(url, data, content_type='application/json')
        self.assertEqual(response.status_code, 400)