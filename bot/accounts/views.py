from django.contrib.auth.decorators import login_required
from django.core.files.uploadedfile import UploadedFile
from .models import CustomUser
from rest_framework.response import Response # type: ignore
from rest_framework.decorators import api_view, permission_classes # type: ignore
from rest_framework import status # type: ignore
from rest_framework.permissions import IsAuthenticated, AllowAny # type: ignore
from rest_framework_simplejwt.tokens import RefreshToken # type: ignore
from django.utils.translation import gettext_lazy as _
from django.contrib.auth import authenticate


@api_view(['POST'])
@permission_classes([AllowAny])  # Explicitly allow unauthenticated access
def register_view(request) -> Response:
    """Handle user registration via API"""
    email = request.data.get('email', '').strip()
    password = request.data.get('password', '')
    full_name = request.data.get('full_name', '').strip()
    
    # Validate required fields
    if not all([email, password, full_name]):
        return Response({
            'error': _('Please provide email, password and full name.')
        }, status=status.HTTP_400_BAD_REQUEST)
    
    # Check if user already exists
    if CustomUser.objects.filter(email=email).exists():
        return Response({
            'error': _('User with this email already exists.')
        }, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        # Create new user
        user = CustomUser.objects.create_user(
            email=email,
            password=password,
            full_name=full_name
        )
        
        # Generate tokens
        refresh = RefreshToken.for_user(user)
        
        return Response({
            'message': _('Registration successful!'),
            'user': {
                'email': user.email,
                'full_name': user.full_name,
            },
            'tokens': {
                'refresh': str(refresh),
                'access': str(refresh.access_token),
            }
        }, status=status.HTTP_201_CREATED)
        
    except Exception as e:
        return Response({
            'error': _('Registration failed. Please try again.')
        }, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
def login_view(request) -> Response:
    """Handle user login via API"""
    email = request.data.get('email', '').strip()
    password = request.data.get('password', '')
    
    if not email or not password:
        return Response({
            'error': _('Please provide both email and password.')
        }, status=status.HTTP_400_BAD_REQUEST)
        
    user = authenticate(request, email=email, password=password)
    
    if user is not None:
        if user.is_active:
            refresh = RefreshToken.for_user(user)
            return Response({
                'message': _('Login successful!'),
                'user': {
                    'email': user.email,
                    'full_name': user.full_name,
                    'profile_image': user.profile_image.url if user.profile_image else None,
                },
                'tokens': {
                    'refresh': str(refresh),
                    'access': str(refresh.access_token),
                }
            })
        return Response({
            'error': _('Your account is inactive.')
        }, status=status.HTTP_403_FORBIDDEN)
    return Response({
        'error': _('Invalid email or password.')
    }, status=status.HTTP_401_UNAUTHORIZED)

@api_view(['POST'])
@login_required
def logout_view(request) -> Response:
    """Handle user logout via API"""
    try:
        refresh_token = request.data.get('refresh_token')
        token = RefreshToken(refresh_token)
        token.blacklist()
        return Response({
            'message': _('Successfully logged out.')
        })
    except Exception:
        return Response({
            'error': _('Error logging out.')
        }, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET', 'PUT'])
@login_required
def profile_view(request) -> Response:
    """Handle user profile via API"""
    if request.method == 'GET':
        user = request.user
        return Response({
            'email': user.email,
            'full_name': user.full_name,
            'profile_image': user.profile_image.url if user.profile_image else None,
            'profile_updated_at': user.profile_updated_at,
        })
    
    # Handle PUT request for profile update
    email = request.data.get('email', '').strip()
    full_name = request.data.get('full_name', '').strip()
    
    if not all([email, full_name]):
        return Response({
            'error': _('Please provide both email and full name.')
        }, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        user = request.user
        user.email = email
        user.full_name = full_name
        user.save()
        
        return Response({
            'message': _('Profile updated successfully!'),
            'user': {
                'email': user.email,
                'full_name': user.full_name,
                'profile_image': user.profile_image.url if user.profile_image else None,
                'profile_updated_at': user.profile_updated_at,
            }
        })
    except Exception as e:
        return Response({
            'error': _('Failed to update profile.')
        }, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
@login_required
def update_profile_image(request) -> Response:
    """Handle profile image updates via API"""
    if not request.FILES.get('profile_image'):
        return Response({
            'error': _('No image file provided')
        }, status=status.HTTP_400_BAD_REQUEST)
    
    profile_image: UploadedFile = request.FILES['profile_image']
    
    if not profile_image.content_type.startswith('image/'):
        return Response({
            'error': _('Invalid file type. Please upload an image.')
        }, status=status.HTTP_400_BAD_REQUEST)
    
    if profile_image.size > 5 * 1024 * 1024:
        return Response({
            'error': _('File too large. Maximum size is 5MB.')
        }, status=status.HTTP_400_BAD_REQUEST)
        
    try:
        request.user.profile_image = profile_image
        request.user.save()
        return Response({
            'message': _('Profile image updated successfully!'),
            'profile_image': request.user.profile_image.url
        })
    except Exception as e:
        return Response({
            'error': _('Failed to update profile image.')
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@login_required
def delete_account(request) -> Response:
    """Handle account deletion via API"""
    password = request.data.get('password', '')
    if not request.user.check_password(password):
        return Response({
            'error': _('Invalid password.')
        }, status=status.HTTP_401_UNAUTHORIZED)
        
    try:
        user = request.user
        user.delete()
        return Response({
            'message': _('Account deleted successfully.')
        })
    except Exception as e:
        return Response({
            'error': _('Failed to delete account.')
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
