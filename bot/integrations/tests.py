from django.test import TestCase
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APIClient
from django.contrib.auth import get_user_model
from .models import Integration, IntegrationCategory

User = get_user_model()

class IntegrationTests(TestCase):
    def setUp(self):
        # Create test user and authenticate
        self.user = User.objects.create_user(
            email='test@example.com',
            password='testpass123',
            full_name='Test User'
        )
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)
        
        # Create test category first
        self.category = IntegrationCategory.objects.create(
            category_name="Test Category",
            description="Test description",
            is_active=True
        )
        
        # Create test integration
        self.integration = Integration.objects.create(
            user=self.user,
            integration_name="Test Integration",
            integration_type="api",
            integration_url="https://test.com/api",
            integration_key="test_key",
            integration_secret="test_secret",
            integration_token="test_token",
            description="Test description",
            is_active=True,
            category=self.category
        )
        
        self.integration_data = {
            "integration_name": "New Integration",
            "integration_type": "api",
            "integration_url": "https://new-test.com/api",
            "integration_key": "new_key",
            "integration_secret": "new_secret",
            "integration_token": "new_token",
            "description": "New test description",
            "is_active": True,
            "category": self.category.id
        }
        
        self.category_data = {
            "category_name": "New Category",
            "description": "New test description",
            "is_active": True
        }

    def test_create_integration(self):
        """Test creating a new integration"""
        url = reverse('integration-list')
        response = self.client.post(url, self.integration_data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(Integration.objects.count(), 2)
        self.assertTrue(Integration.objects.filter(integration_name='New Integration').exists())
        
        # Verify response data
        self.assertEqual(response.data['integration_name'], 'New Integration')
        self.assertEqual(response.data['integration_type'], 'api')
        self.assertEqual(response.data['category'], self.category.id)

    def test_get_integrations(self):
        """Test retrieving list of integrations"""
        url = reverse('integration-list')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 1)
        
        # Verify response data
        integration = response.data[0]
        self.assertEqual(integration['integration_name'], 'Test Integration')
        self.assertEqual(integration['integration_type'], 'api')
        self.assertEqual(integration['category'], self.category.id)

    def test_get_integration_detail(self):
        """Test retrieving specific integration details"""
        url = reverse('integration-detail', args=[self.integration.id])
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        # Verify response data
        self.assertEqual(response.data['integration_name'], 'Test Integration')
        self.assertEqual(response.data['integration_type'], 'api')
        self.assertEqual(response.data['category'], self.category.id)

    def test_update_integration(self):
        """Test updating an integration"""
        url = reverse('integration-detail', args=[self.integration.id])
        update_data = self.integration_data.copy()
        update_data['integration_name'] = 'Updated Integration'
        
        response = self.client.put(url, update_data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        # Verify update
        self.integration.refresh_from_db()
        self.assertEqual(self.integration.integration_name, 'Updated Integration')
        self.assertEqual(response.data['integration_name'], 'Updated Integration')

    def test_delete_integration(self):
        """Test deleting an integration"""
        url = reverse('integration-detail', args=[self.integration.id])
        response = self.client.delete(url)
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)
        self.assertEqual(Integration.objects.count(), 0)

    def test_create_integration_category(self):
        """Test creating a new integration category"""
        url = reverse('integration-category-list')
        response = self.client.post(url, self.category_data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(IntegrationCategory.objects.count(), 2)
        
        # Verify response data
        self.assertEqual(response.data['category_name'], 'New Category')
        self.assertTrue(response.data['is_active'])

    def test_get_integration_categories(self):
        """Test retrieving list of integration categories"""
        url = reverse('integration-category-list')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 1)
        
        # Verify response data
        category = response.data[0]
        self.assertEqual(category['category_name'], 'Test Category')
        self.assertTrue(category['is_active'])

    def test_update_integration_category(self):
        """Test updating an integration category"""
        url = reverse('integration-category-detail', args=[self.category.id])
        update_data = self.category_data.copy()
        update_data['category_name'] = 'Updated Category'
        
        response = self.client.put(url, update_data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        # Verify update
        self.category.refresh_from_db()
        self.assertEqual(self.category.category_name, 'Updated Category')
        self.assertEqual(response.data['category_name'], 'Updated Category')

    def test_delete_integration_category(self):
        """Test deleting an integration category"""
        url = reverse('integration-category-detail', args=[self.category.id])
        response = self.client.delete(url)
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)
        self.assertEqual(IntegrationCategory.objects.count(), 0)

    def test_unauthorized_access(self):
        """Test unauthorized access to integration endpoints"""
        self.client.force_authenticate(user=None)
        
        # Test list endpoint
        url = reverse('integration-list')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
        
        # Test detail endpoint
        url = reverse('integration-detail', args=[self.integration.id])
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
