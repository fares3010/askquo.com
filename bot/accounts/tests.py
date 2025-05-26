from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth import get_user_model
from rest_framework.test import APITestCase, APIClient
from rest_framework import status
from .forms import UserRegistrationForm, UserProfileForm
from django.core.files.uploadedfile import SimpleUploadedFile
import os

User = get_user_model()

class UserAPITest(APITestCase):
    def setUp(self):
        self.client = APIClient()
        self.user_data = {
            'email': 'test@example.com',
            'password': 'testpass123',
            'full_name': 'Test User'
        }
        self.user = User.objects.create_user(**self.user_data)
        self.client.force_authenticate(user=self.user)
        self.base_url = '/api/accounts/'

    def test_user_registration_api(self):
        url = f'{self.base_url}register/'
        data = {
            'email': 'new@example.com',
            'password': 'newpass123',
            'password2': 'newpass123',
            'full_name': 'New API User'
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertTrue(User.objects.filter(email='new@example.com').exists())

    def test_user_login_api(self):
        url = f'{self.base_url}login/'
        data = {
            'email': 'test@example.com',
            'password': 'testpass123'
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('token', response.data)

    def test_user_profile_api(self):
        url = f'{self.base_url}profile/'
        # Test GET profile
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['email'], self.user.email)

        # Test PUT profile update
        update_data = {
            'full_name': 'Updated API User'
        }
        response = self.client.put(url, update_data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['full_name'], 'Updated API User')

    def test_user_logout_api(self):
        url = f'{self.base_url}logout/'
        response = self.client.post(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_unauthorized_access(self):
        self.client.force_authenticate(user=None)
        url = f'{self.base_url}profile/'
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

class UserProfileImageAPITest(APITestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            email='test@example.com',
            password='testpass123',
            full_name='Test User'
        )
        self.client.force_authenticate(user=self.user)
        self.profile_url = '/api/accounts/profile/'

    def test_profile_image_upload(self):
        test_image = SimpleUploadedFile(
            name='test_image.jpg',
            content=b'fake image content',
            content_type='image/jpeg'
        )
        data = {
            'profile_image': test_image
        }
        response = self.client.put(
            self.profile_url,
            data,
            format='multipart'
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('profile_image', response.data)
