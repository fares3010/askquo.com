from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.urls import reverse
from .models import Agent, AgentDocuments, AgentIntegrations, AgentWebsites
from django.utils import timezone
from datetime import timedelta
import json

class AgentAPITests(TestCase):
    def setUp(self):
        """Set up test data for API tests"""
        self.client = Client()
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        self.client.login(username='testuser', password='testpass123')
        self.agent = Agent.objects.create(
            user=self.user,
            name='Test Agent',
            description='A test agent',
            visibility='private'
        )

    def test_create_agent(self):
        """Test creating a new agent via API"""
        url = reverse('create_agent')
        data = {
            'name': 'New Test Agent',
            'description': 'A new test agent',
            'visibility': 'private'
        }
        response = self.client.post(url, data, content_type='application/json')
        self.assertEqual(response.status_code, 201)
        self.assertTrue(Agent.objects.filter(name='New Test Agent').exists())
        
        # Verify response data
        response_data = json.loads(response.content)
        self.assertEqual(response_data['name'], 'New Test Agent')
        self.assertEqual(response_data['visibility'], 'private')
        self.assertIn('agent_id', response_data)
        self.assertIn('created_at', response_data)
        self.assertIn('updated_at', response_data)

    def test_get_agent_list(self):
        """Test retrieving the list of agents"""
        # Create additional agent
        Agent.objects.create(
            user=self.user,
            name='Another Agent',
            description='Another test agent',
            visibility='public'
        )

        url = reverse('agent_list')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        
        # Verify response data
        response_data = json.loads(response.content)
        self.assertEqual(len(response_data), 2)
        self.assertEqual(response_data[0]['name'], 'Test Agent')

    def test_get_agent_detail(self):
        """Test retrieving a specific agent's details"""
        url = reverse('agent_detail', args=[self.agent.agent_id])
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        
        # Verify response data
        response_data = json.loads(response.content)
        self.assertEqual(response_data['name'], 'Test Agent')
        self.assertContains(response, 'Test Agent')
        self.assertContains(response, 'A test agent')


    def test_update_agent(self):
        """Test updating an existing agent via API"""
        url = reverse('agent_detail', args=[self.agent.agent_id])
        data = {
            'name': 'Updated Test Agent',
            'description': 'An updated test agent',
            'visibility': 'public'
        }
        response = self.client.put(url, data, content_type='application/json')
        self.assertEqual(response.status_code, 200)
        self.agent.refresh_from_db()
        self.assertEqual(self.agent.name, 'Updated Test Agent')
        self.assertEqual(self.agent.visibility, 'public')

    def test_delete_agent(self):
        """Test deleting an agent via API"""
        url = reverse('agent_detail', args=[self.agent.agent_id])
        response = self.client.delete(url)
        self.assertEqual(response.status_code, 204)
        self.assertTrue(Agent.objects.filter(agent_id=self.agent.agent_id, is_deleted=True).exists())

    def test_get_nonexistent_agent(self):
        """Test retrieving a non-existent agent"""
        url = reverse('agent_detail', args=[99999])  # Non-existent ID
        response = self.client.get(url)
        self.assertEqual(response.status_code, 404)

    def test_create_agent_validation(self):
        """Test agent creation validation"""
        url = reverse('create_agent')
        # Test missing required field
        data = {'description': 'Missing name field'}
        response = self.client.post(url, data, content_type='application/json')
        self.assertEqual(response.status_code, 400)

        # Test invalid visibility value
        data = {
            'name': 'Invalid Agent',
            'description': 'Test agent',
            'visibility': 'invalid_value'
        }
        response = self.client.post(url, data, content_type='application/json')
        self.assertEqual(response.status_code, 400)

    def test_unauthorized_access(self):
        """Test unauthorized access to agent endpoints"""
        self.client.logout()
        
        # Test getting agent list
        url = reverse('agent_list')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 401)

        # Test getting agent detail
        url = reverse('agent_detail', args=[self.agent.agent_id])
        response = self.client.get(url)
        self.assertEqual(response.status_code, 401)

