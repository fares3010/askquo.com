from django.contrib.auth.decorators import login_required
from django.core.files.uploadedfile import UploadedFile
from django.contrib.auth.hashers import make_password, check_password
from django.core.validators import validate_email
from django.core.exceptions import ValidationError
from .models import CustomUser
from rest_framework.response import Response # type: ignore
from rest_framework.decorators import api_view, permission_classes # type: ignore
from rest_framework import status # type: ignore
from rest_framework.permissions import IsAuthenticated, AllowAny # type: ignore
from rest_framework_simplejwt.tokens import RefreshToken, UntypedToken # type: ignore
from rest_framework_simplejwt.token_blacklist.models import BlacklistedToken, OutstandingToken # type: ignore
from django.utils.translation import gettext_lazy as _
from django.contrib.auth import authenticate
from django.middleware.csrf import get_token
import logging
from plans.models import UserSubscription, PlanPrice, PlanFeature
from .models import Team, TeamMember, TeamAgent
from create_agent.models import Agent
from django.db import transaction
from django.utils import timezone
from rest_framework_simplejwt.exceptions import TokenError, InvalidToken

logger = logging.getLogger(__name__)

@api_view(['POST'])
@permission_classes([AllowAny])
def register_view(request) -> Response:
    """Handle user registration via API with improved validation and security"""
    email = request.data.get('email', '').strip().lower()
    password = request.data.get('password', '')
    full_name = request.data.get('full_name', '').strip()
    
    # Enhanced validation
    if not all([email, password, full_name]):
        return Response({
            'error': _('Please provide email, password and full name.')
        }, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        validate_email(email)
    except ValidationError:
        return Response({
            'error': _('Please provide a valid email address.')
        }, status=status.HTTP_400_BAD_REQUEST)
    
    if len(password) < 8:
        return Response({
            'error': _('Password must be at least 8 characters long.')
        }, status=status.HTTP_400_BAD_REQUEST)
    
    if CustomUser.objects.filter(email=email).exists():
        return Response({
            'error': _('User with this email already exists.')
        }, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        user = CustomUser.objects.create(
            email=email,
            password=make_password(password),  # Properly hash the password
            full_name=full_name
        )

        # Create a free subscription for the user
        UserSubscription.objects.create(
            user=user,
            price=PlanPrice.objects.get(price_id=7),
            plan_name='Free',
            is_active=True,
            is_trial=True,
        )
                
        refresh = RefreshToken.for_user(user)
        logger.info(f"New user registered: {email}")
        logger.info(f"free subscription created for user: {email}")
        
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
        logger.error(f"Registration failed for {email}: {str(e)}")
        return Response({
            'error': _('Registration failed. Please try again.')
        }, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
@permission_classes([AllowAny])
def login_view(request) -> Response:
    """Handle user login via API with improved security"""
    email = request.data.get('email', '').strip().lower()
    password = request.data.get('password', '')
    
    if not email or not password:
        return Response({
            'error': _('Please provide both email and password.')
        }, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        user = authenticate(email=email, password=password)
        if user is None:
            return Response({
                'error': _('Invalid email or password.')
            }, status=status.HTTP_401_UNAUTHORIZED)
            
        if not user.is_active:
            return Response({
                'error': _('Your account is inactive.')
            }, status=status.HTTP_403_FORBIDDEN)
        
        refresh = RefreshToken.for_user(user)
        logger.info(f"User logged in: {email}")
        
        # Handle user subscription with proper exception handling
        try:
            user_subscription = UserSubscription.objects.get(user=user, is_active=True, is_deleted=False)
            if user_subscription.price and user_subscription.price.price_id == 7 and user_subscription.is_expired():
                user_subscription.is_active = False
                user_subscription.is_deleted = True
                user_subscription.save()
                logger.info(f"User subscription expired for user: {email}, deleting subscription")
                UserSubscription.objects.create(
                    user=user,
                    price=PlanPrice.objects.get(price_id=7),
                    plan_name='Free',
                    is_active=True,
                    is_trial=True,
                )
                logger.info(f"New free subscription created for user: {email}")
        except UserSubscription.DoesNotExist:
            logger.warning(f"No active subscription found for user {email} during login, creating free subscription")
            UserSubscription.objects.create(
                user=user,
                price=PlanPrice.objects.get(price_id=7),
                plan_name='Free',
                is_active=True,
                is_trial=True,
            )
            logger.info(f"Free subscription created for user: {email}")
        except Exception as e:
            logger.error(f"Error handling subscription for user {email}: {str(e)}")
            # Continue with login even if subscription handling fails
        
        return Response({
            'message': _('Login successful!'),
            'user': {
                'email': user.email,
                'full_name': user.full_name,
                'profile_image': user.get_profile_image_url(),
            },
            'tokens': {
                'refresh': str(refresh),
                'access': str(refresh.access_token),
            }
        })
        
    except Exception as e:
        logger.error(f"Login failed for {email}: {str(e)}")
        return Response({
            'error': _('Login failed. Please try again.')
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
@permission_classes([AllowAny])
def auth_csrf_token(request) -> Response:
    """Handle CSRF token generation via API with improved error handling"""
    try:
        csrf_token = get_token(request)
        return Response({
            'X-CSRFToken': csrf_token,
            'message': _('CSRF token generated successfully')
        })
    except Exception as e:
        logger.error(f"CSRF token generation failed: {str(e)}")
        return Response({
            'error': _('Failed to generate CSRF token')
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@permission_classes([AllowAny])
def logout_view(request) -> Response:
    """Handle user logout via API with token blacklisting, cookie deletion, and improved error handling"""
    try:
        # Validate refresh token presence
        refresh_token = request.data.get('refresh_token')
        if not refresh_token:
            user_email = request.user.email if request.user.is_authenticated else "anonymous"
            logger.warning(f"Logout attempt without refresh token for user: {user_email}")
            
            # Even if no refresh token, still clear cookies
            response = Response({
                'error': _('Refresh token is required for logout.')
            }, status=status.HTTP_400_BAD_REQUEST)
            _clear_auth_cookies(response)
            return response
        
        # Validate refresh token format
        if not isinstance(refresh_token, str) or len(refresh_token.strip()) == 0:
            user_email = request.user.email if request.user.is_authenticated else "anonymous"
            logger.warning(f"Invalid refresh token format for user: {user_email}")
            
            response = Response({
                'error': _('Invalid refresh token format.')
            }, status=status.HTTP_400_BAD_REQUEST)
            _clear_auth_cookies(response)
            return response
        
        # Blacklist the refresh token
        try:
            # Parse the token to validate it
            token = UntypedToken(refresh_token.strip())
            logger.info(f"Token JTI: {token['jti']}")
            
            # Get the outstanding token and blacklist it
            outstanding_token = OutstandingToken.objects.get(jti=token['jti'])
            blacklisted_token, created = BlacklistedToken.objects.get_or_create(token=outstanding_token)
            
            user_email = request.user.email if request.user.is_authenticated else "anonymous"
            if not created:
                logger.info(f"Token was already blacklisted for user: {user_email}")
            else:
                logger.info(f"User successfully logged out: {user_email}")
                
        except OutstandingToken.DoesNotExist:
            user_email = request.user.email if request.user.is_authenticated else "anonymous"
            logger.error(f"Outstanding token not found for user {user_email}")
            
            response = Response({
                'error': _('Invalid refresh token.')
            }, status=status.HTTP_401_UNAUTHORIZED)
            _clear_auth_cookies(response)
            return response
            
        except Exception as token_error:
            user_email = request.user.email if request.user.is_authenticated else "anonymous"
            logger.error(f"Token blacklisting failed for user {user_email}: {str(token_error)}")
            
            response = Response({
                'error': _('Invalid or expired refresh token.')
            }, status=status.HTTP_401_UNAUTHORIZED)
            _clear_auth_cookies(response)
            return response
        
        # Successful logout - create response and clear cookies
        response = Response({
            'message': _('Successfully logged out.'),
            'timestamp': timezone.now().isoformat()
        }, status=status.HTTP_200_OK)
        
        # Clear all authentication-related cookies
        _clear_auth_cookies(response)
        
        return response
        
    except Exception as e:
        user_email = request.user.email if request.user.is_authenticated else "anonymous"
        logger.error(f"Unexpected error during logout for user {user_email}: {str(e)}", exc_info=True)
        
        response = Response({
            'error': _('An unexpected error occurred during logout. Please try again.')
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        _clear_auth_cookies(response)
        return response


def _clear_auth_cookies(response: Response) -> None:
    """
    Clear all authentication-related cookies from the response.
    This ensures complete logout even if token blacklisting fails.
    """
    # Common cookie names for JWT tokens
    cookie_names = [
        'access_token',
        'refresh_token',
        'accessToken',
        'refreshToken',
        'jwt_access',
        'jwt_refresh',
        'auth_token',
        'session_token',
        'csrftoken',
        'sessionid',
    ]
    
    # Cookie deletion settings
    cookie_settings = {
        'path': '/',
        'domain': None,
    }
    
    # Clear each cookie
    for cookie_name in cookie_names:
        response.delete_cookie(cookie_name, **cookie_settings)
    
    # Also try to clear cookies with different path settings
    # Some cookies might be set with specific paths
    for cookie_name in ['access_token', 'refresh_token', 'auth_token']:
        response.delete_cookie(cookie_name, path='/api/', **{k: v for k, v in cookie_settings.items() if k != 'path'})
        response.delete_cookie(cookie_name, path='/auth/', **{k: v for k, v in cookie_settings.items() if k != 'path'})
    
    # Add security headers
    response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response['Pragma'] = 'no-cache'
    response['Expires'] = '0'
    
    logger.info("Authentication cookies cleared successfully")
@api_view(['GET', 'PUT'])
@permission_classes([IsAuthenticated])
def profile_view(request) -> Response:
    """Handle user profile via API with improved validation"""
    if request.method == 'GET':
        user = request.user
        return Response({
            'email': user.email,
            'full_name': user.full_name,
            'profile_image': user.get_profile_image_url(),
            'profile_updated_at': user.profile_updated_at,
        })
    
    email = request.data.get('email', '').strip().lower()
    full_name = request.data.get('full_name', '').strip()
    
    if not all([email, full_name]):
        return Response({
            'error': _('Please provide both email and full name.')
        }, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        validate_email(email)
    except ValidationError:
        return Response({
            'error': _('Please provide a valid email address.')
        }, status=status.HTTP_400_BAD_REQUEST)
    
    if CustomUser.objects.exclude(pk=request.user.pk).filter(email=email).exists():
        return Response({
            'error': _('Email already in use by another account.')
        }, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        user = request.user
        user.email = email
        user.full_name = full_name
        user.save(update_fields=['email', 'full_name'])
        logger.info(f"Profile updated for user: {email}")
        
        return Response({
            'message': _('Profile updated successfully!'),
            'user': {
                'email': user.email,
                'full_name': user.full_name,
                'profile_image': user.get_profile_image_url(),
                'profile_updated_at': user.profile_updated_at,
            }
        })
    except Exception as e:
        logger.error(f"Profile update failed: {str(e)}")
        return Response({
            'error': _('Failed to update profile.')
        }, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def update_profile_image(request) -> Response:
    """Handle profile image updates via API with improved validation"""
    if not request.FILES.get('profile_image'):
        return Response({
            'error': _('No image file provided')
        }, status=status.HTTP_400_BAD_REQUEST)
    
    profile_image: UploadedFile = request.FILES['profile_image']
    allowed_types = ['image/jpeg', 'image/png', 'image/gif']
    
    if profile_image.content_type not in allowed_types:
        return Response({
            'error': _('Invalid file type. Please upload a JPEG, PNG, or GIF image.')
        }, status=status.HTTP_400_BAD_REQUEST)
    
    if profile_image.size > 5 * 1024 * 1024:  # 5MB limit
        return Response({
            'error': _('File too large. Maximum size is 5MB.')
        }, status=status.HTTP_400_BAD_REQUEST)
        
    try:
        request.user.update_profile_image(profile_image)
        logger.info(f"Profile image updated for user: {request.user.email}")
        
        return Response({
            'message': _('Profile image updated successfully!'),
            'profile_image': request.user.get_profile_image_url()
        })
    except Exception as e:
        logger.error(f"Profile image update failed: {str(e)}")
        return Response({
            'error': _('Failed to update profile image.')
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def delete_account(request) -> Response:
    """Handle account deletion via API with confirmation"""
    password = request.data.get('password', '')
    
    if not password:
        return Response({
            'error': _('Password is required for account deletion.')
        }, status=status.HTTP_400_BAD_REQUEST)
    
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
    
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def create_team(request) -> Response:
    """Handle team creation via API with improved validation"""
    team_name = request.data.get('team_name', '').strip()
    
    # Get user subscription with proper exception handling
    user_subscription = UserSubscription.objects.get(user=request.user, is_active=True, is_deleted=False)
    if not user_subscription:
        return Response({
            'error': _('No active subscription found. Please subscribe to a plan to create teams.')
        }, status=status.HTTP_403_FORBIDDEN)
    # Check team members usage limit
    team_members_usage = user_subscription.team_members_usage or 0
    plan =user_subscription.price.plan
    plan_feature = PlanFeature.objects.filter(plan=plan, feature_name='team members').first()
    if plan_feature.feature_limit is not None and team_members_usage >= plan_feature.feature_limit:
        logger.info(f"User {request.user.email} has reached the maximum number of team members. Please upgrade to a paid plan.")
        return Response({
            'error': _('You have reached the maximum number of team members. Please upgrade to a paid plan.')
        }, status=status.HTTP_403_FORBIDDEN)

    # Enhanced validation
    if not team_name:
        return Response({
            'error': _('Team name is required.')
        }, status=status.HTTP_400_BAD_REQUEST)
    
    if len(team_name) < 2:
        return Response({
            'error': _('Team name must be at least 2 characters long.')
        }, status=status.HTTP_400_BAD_REQUEST)
    
    if len(team_name) > 255:
        return Response({
            'error': _('Team name must be less than 255 characters.')
        }, status=status.HTTP_400_BAD_REQUEST)
    
    if Team.objects.filter(team_name=team_name).exists():
        return Response({
            'error': _('Team with this name already exists.')
        }, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        team = Team.objects.create(
            team_name=team_name,
            user=request.user
        )
        logger.info(f"Team created: {team_name} by user: {request.user.email}")
        return Response({
            'message': _('Team created successfully!'),
            "team": {
                "team_id": team.team_id,
                "team_name": team.team_name,
                "team_created_at": team.created_at,
                "team_updated_at": team.updated_at,
            }
        })
    except Exception as e:
        logger.error(f"Team creation failed for user {request.user.email}: {str(e)}")
        return Response({
            'error': _('Failed to create team.')
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def add_team_member(request) -> Response:
    """Handle team member addition via API with improved validation"""
    team_id = request.data.get('team_id', '')
    team_member_name = request.data.get('team_member_name', '').strip()
    team_member_email = request.data.get('team_member_email', '').strip().lower()
    team_member_role = request.data.get('team_member_role', 'member')

    # Get user subscription with proper exception handling
    user_subscription = UserSubscription.objects.get(user=request.user, is_active=True, is_deleted=False)
    if not user_subscription:
        return Response({
            'error': _('No active subscription found. Please subscribe to a plan to add team members.')
        }, status=status.HTTP_403_FORBIDDEN)
    
    # Check team members usage limit
    team_members_usage = user_subscription.team_members_usage or 0
    plan =user_subscription.price.plan
    plan_feature = PlanFeature.objects.filter(plan=plan, feature_name='team members').first()
    if plan_feature.feature_limit is not None and team_members_usage >= plan_feature.feature_limit:
        logger.info(f"User {request.user.email} has reached the maximum number of team members. Please upgrade to a paid plan.")
        return Response({
            'error': _('You have reached the maximum number of team members. Please upgrade to a paid plan.')
        }, status=status.HTTP_403_FORBIDDEN)
    
    # Enhanced validation
    if not team_id:
        return Response({
            'error': _('Team ID is required.')
        }, status=status.HTTP_400_BAD_REQUEST)
    
    if not team_member_name:
        return Response({
            'error': _('Team member name is required.')
        }, status=status.HTTP_400_BAD_REQUEST)
    
    if not team_member_email:
        return Response({
            'error': _('Team member email is required.')
        }, status=status.HTTP_400_BAD_REQUEST)
    
    # Validate email format
    try:
        validate_email(team_member_email)
    except ValidationError:
        return Response({
            'error': _('Please provide a valid email address.')
        }, status=status.HTTP_400_BAD_REQUEST)
    
    # Validate role
    valid_roles = ['admin', 'member', 'viewer']
    if team_member_role not in valid_roles:
        return Response({
            'error': _('Invalid role. Must be one of: admin, member, viewer.')
        }, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        team = Team.objects.get(team_id=team_id)
        
        # Check if user owns the team
        if team.user != request.user:
            return Response({
                'error': _('You are not authorized to add members to this team.')
            }, status=status.HTTP_403_FORBIDDEN)
        
        # Check if team is active
        if not team.is_active or team.is_deleted:
            return Response({
                'error': _('Cannot add members to inactive or deleted team.')
            }, status=status.HTTP_400_BAD_REQUEST)
            
        # Check if member already exists
        if TeamMember.objects.filter(team=team, team_member_email=team_member_email).exists():
            return Response({
                'error': _('Team member with this email already exists in this team.')
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Create team member
        with transaction.atomic():
            team_member = TeamMember.objects.create(
                team=team,
                team_member_name=team_member_name,
                team_member_email=team_member_email,
                team_member_role=team_member_role
            )
            logger.info(f"Team member added: {team_member_name} ({team_member_email}) to team: {team.team_name}")
            user_subscription.team_members_usage += 1
            user_subscription.save()
            
        
        return Response({       
            "message": _('Team member added successfully!'),
            "team_member": {
                "team_member_id": team_member.team_member_id,
                "team_member_name": team_member.team_member_name,
                "team_member_email": team_member.team_member_email,
                "team_member_role": team_member.team_member_role,
                "team_member_created_at": team_member.created_at,
                "team_member_updated_at": team_member.updated_at,
            }
        })
    except Team.DoesNotExist:
        return Response({
            'error': _('Team not found.')
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        logger.error(f"Team member addition failed: {str(e)}")
        return Response({
            'error': _('Failed to add team member.')
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def assign_team_agent(request) -> Response:
    """Handle team agent assignment via API with improved validation and error handling"""
    team_id = request.data.get('team_id')
    agent_id = request.data.get('agent_id')
    
    # Enhanced validation with better error messages
    if not team_id:
        return Response({
            'error': _('Team ID is required.')
        }, status=status.HTTP_400_BAD_REQUEST)
    
    if not agent_id:
        return Response({
            'error': _('Agent ID is required.')
        }, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        # Validate team_id and agent_id are valid integers
        try:
            team_id = int(team_id)
            agent_id = int(agent_id)
        except ValueError:
            return Response({
                'error': _('Team ID and Agent ID must be valid numbers.')
            }, status=status.HTTP_400_BAD_REQUEST)
        
        team = Team.objects.get(team_id=team_id)
        agent = Agent.objects.get(agent_id=agent_id)
        
        # Check if user owns the team
        if team.user != request.user:
            logger.warning(f"Unauthorized team agent assignment attempt: user {request.user.email} tried to assign agent {agent_id} to team {team_id}")
            return Response({
                'error': _('You are not authorized to assign agents to this team.')
            }, status=status.HTTP_403_FORBIDDEN)
        
        # Check if team is active and not deleted
        if not team.is_active or team.is_deleted:   
            return Response({
                'error': _('Cannot assign agents to inactive or deleted team.')
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Check if agent is active and not deleted
        if not agent.is_active or agent.is_deleted:
            return Response({
                'error': _('Cannot assign inactive or deleted agent to team.')
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Check if agent is already assigned to team (optimized query)
        if TeamAgent.objects.filter(team=team, agent=agent).exists():
            return Response({
                'error': _('Agent is already assigned to this team.')
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Assign agent to team
        team_agent = TeamAgent.objects.create(
            team=team,
            agent=agent
        )
        
        # Log successful assignment
        logger.info(f"Agent {team_agent.agent.name} (ID: {team_agent.agent.agent_id}) assigned to team '{team.team_name}' (ID: {team_id}) by user: {request.user.email}")
        
        return Response({
            "message": _('Agent assigned to team successfully!'),
            "team": {
                "team_id": team.team_id,
                "team_name": team.team_name,
                "agent_count": team.get_team_agent_count(),
                "team_created_at": team.created_at,
                "team_updated_at": team.updated_at,
            },
            "agent": {
                "agent_id": agent.agent_id,
                "agent_name": agent.name,
            }
        })
        
    except Team.DoesNotExist:
        logger.warning(f"Team not found: team_id={team_id}")
        return Response({
            'error': _('Team not found.')
        }, status=status.HTTP_404_NOT_FOUND)
    except Agent.DoesNotExist:
        logger.warning(f"Agent not found: agent_id={agent_id}")
        return Response({
            'error': _('Agent not found.')
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        logger.error(f"Team agent assignment failed for user {request.user.email}: {str(e)}")
        return Response({
            'error': _('Failed to assign agent to team. Please try again.')
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def remove_team_agent(request) -> Response:
    """Handle team agent removal via API with improved validation and error handling"""
    team_id = request.data.get('team_id')
    agent_id = request.data.get('agent_id')
    
    # Enhanced validation with better error messages
    if not team_id:
        return Response({
            'error': _('Team ID is required.')
        }, status=status.HTTP_400_BAD_REQUEST)
    
    if not agent_id:
        return Response({
            'error': _('Agent ID is required.')
        }, status=status.HTTP_400_BAD_REQUEST)
        
    try:
        # Validate team_id and agent_id are valid integers
        try:
            team_id = int(team_id)
            agent_id = int(agent_id)
        except ValueError:
            return Response({
                'error': _('Team ID and Agent ID must be valid numbers.')
            }, status=status.HTTP_400_BAD_REQUEST)
        
        team = Team.objects.get(team_id=team_id)
        agent = Agent.objects.get(agent_id=agent_id)
        
        # Check if user owns the team
        if team.user != request.user:
            logger.warning(f"Unauthorized team agent removal attempt: user {request.user.email} tried to remove agent {agent_id} from team {team_id}")
            return Response({
                'error': _('You are not authorized to remove agents from this team.')
            }, status=status.HTTP_403_FORBIDDEN)
        
        # Check if team is active and not deleted
        if not team.is_active or team.is_deleted:
            return Response({
                'error': _('Cannot remove agents from inactive or deleted team.')
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Check if agent is active and not deleted  
        if not agent.is_active or agent.is_deleted:
            return Response({
                'error': _('Cannot remove inactive or deleted agent from team.')
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Check if agent is assigned to team
        if not TeamAgent.objects.filter(team=team, agent=agent).exists():
            return Response({
                'error': _('Agent is not assigned to this team.')
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Remove agent from team
        team_agent = TeamAgent.objects.get(team=team, agent=agent)
        team_agent.is_active = False
        team_agent.is_deleted = True
        team_agent.save(update_fields=['is_active', 'is_deleted', 'updated_at'])
        
        # Log successful removal
        logger.info(f"Agent {team_agent.agent.name} (ID: {team_agent.agent.agent_id}) removed from team '{team.team_name}' (ID: {team_id}) by user: {request.user.email}")
        
        return Response({
            "message": _('Agent removed from team successfully!'),
            "team": {
                "team_id": team.team_id,
                "team_name": team.team_name,
                "agent_count": team.get_team_agent_count(),
                "team_created_at": team.created_at,
                "team_updated_at": team.updated_at,
            },
            "agent": {
                "agent_id": agent.agent_id,
                "agent_name": agent.name,
            }
        })
    except Team.DoesNotExist:
        logger.warning(f"Team not found: team_id={team_id}")
        return Response({
            'error': _('Team not found.')
        }, status=status.HTTP_404_NOT_FOUND)
    
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def remove_team_member(request) -> Response:
    """Handle team member removal via API with improved validation and error handling"""
    team_id = request.data.get('team_id')
    team_member_id = request.data.get('team_member_id')
    
    # Enhanced validation with better error messages
    if not team_id:
        return Response({
            'error': _('Team ID is required.')
        }, status=status.HTTP_400_BAD_REQUEST)
    
    if not team_member_id:
        return Response({
            'error': _('Team member ID is required.')
        }, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        # Validate team_id and team_member_id are valid integers
        try:
            team_id = int(team_id)
            team_member_id = int(team_member_id)
        except ValueError:
            return Response({
                'error': _('Team ID and Team member ID must be valid numbers.')
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Get team and team member with single query optimization
        team = Team.objects.select_related('user').get(team_id=team_id)
        team_member = TeamMember.objects.get(team_member_id=team_member_id)
        
        # Check if user owns the team
        if team.user != request.user:
            logger.warning(f"Unauthorized team member removal attempt: user {request.user.email} tried to remove team member {team_member_id} from team {team_id}")
            return Response({
                'error': _('You are not authorized to remove team members from this team.')
            }, status=status.HTTP_403_FORBIDDEN)
        
        # Check if team is active and not deleted
        if not team.is_active or team.is_deleted:
            return Response({
                'error': _('Cannot remove team members from inactive or deleted team.')
            }, status=status.HTTP_400_BAD_REQUEST)

        # Check if team member is active and not deleted
        if not team_member.is_active or team_member.is_deleted:
            return Response({
                'error': _('Cannot remove inactive or deleted team member from team.')
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Check if team member belongs to the specified team
        if team_member.team != team:
            return Response({
                'error': _('Team member does not belong to this team.')
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Soft delete the team member instead of removing from relationship
        team_member.is_active = False
        team_member.is_deleted = True
        team_member.save(update_fields=['is_active', 'is_deleted', 'updated_at'])
        
        # Log successful removal
        logger.info(f"Team member {team_member.team_member_name} (ID: {team_member_id}) removed from team '{team.team_name}' (ID: {team_id}) by user: {request.user.email}")
        
        return Response({
            "message": _('Team member removed from team successfully!'),
            "team": {
                "team_id": team.team_id,
                "team_name": team.team_name,
                "team_member_count": team.get_team_member_count(),
                "team_created_at": team.created_at,
                "team_updated_at": team.updated_at,
            },
            "team_member": {
                "team_member_id": team_member.team_member_id,
                "team_member_name": team_member.team_member_name,
                "team_member_email": team_member.team_member_email,
                "team_member_role": team_member.team_member_role,
                "team_member_created_at": team_member.created_at,
                "team_member_updated_at": team_member.updated_at,
            }
        })
    except Team.DoesNotExist:
        logger.warning(f"Team not found: team_id={team_id}")
        return Response({
            'error': _('Team not found.')
        }, status=status.HTTP_404_NOT_FOUND)
    except TeamMember.DoesNotExist:
        logger.warning(f"Team member not found: team_member_id={team_member_id}")
        return Response({
            'error': _('Team member not found.')
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        logger.error(f"Team member removal failed: {str(e)}")
        return Response({
            'error': _('Failed to remove team member. Please try again.')
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def deactivate_team_member(request) -> Response:
    """Handle team member deactivation via API with improved validation and error handling"""
    team_id = request.data.get('team_id')
    team_member_id = request.data.get('team_member_id')
    
    # Enhanced validation with better error messages
    if not team_id:
        return Response({
            'error': _('Team ID is required.')
        }, status=status.HTTP_400_BAD_REQUEST)
    
    if not team_member_id:
        return Response({
            'error': _('Team member ID is required.')
        }, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        # Validate team_id and team_member_id are valid integers
        try:
            team_id = int(team_id)
            team_member_id = int(team_member_id)
        except ValueError:
            return Response({
                'error': _('Team ID and Team member ID must be valid numbers.')
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Get team and team member with single query optimization
        team = Team.objects.select_related('user').get(team_id=team_id)
        team_member = TeamMember.objects.select_related('team').get(team_member_id=team_member_id)
        
        # Check if user owns the team
        if team.user != request.user:
            logger.warning(f"Unauthorized team member deactivation attempt: user {request.user.email} tried to deactivate team member {team_member_id} from team {team_id}")
            return Response({
                'error': _('You are not authorized to deactivate team members in this team.')
            }, status=status.HTTP_403_FORBIDDEN)
        
        # Check if team is active and not deleted
        if not team.is_active or team.is_deleted:
            return Response({
                'error': _('Cannot deactivate team members in inactive or deleted team.')
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Check if team member is active and not deleted
        if not team_member.is_active or team_member.is_deleted:
            return Response({
                'error': _('Cannot deactivate inactive or deleted team member.')
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Check if team member belongs to the specified team
        if team_member.team != team:
            return Response({
                'error': _('Team member does not belong to this team.')
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Deactivate the team member
        team_member.is_active = False
        team_member.save(update_fields=['is_active', 'updated_at'])
        
        # Log successful deactivation
        logger.info(f"Team member {team_member.team_member_name} (ID: {team_member_id}) deactivated in team '{team.team_name}' (ID: {team_id}) by user: {request.user.email}")
        
        return Response({
            "message": _('Team member deactivated successfully!'),
            "team": {
                "team_id": team.team_id,
                "team_name": team.team_name,
                "team_member_count": team.get_team_member_count(),
                "team_created_at": team.created_at,
                "team_updated_at": team.updated_at,
            },
            "team_member": {
                "team_member_id": team_member.team_member_id,
                "team_member_name": team_member.team_member_name,
                "team_member_email": team_member.team_member_email,
                "team_member_role": team_member.team_member_role,
                "team_member_created_at": team_member.created_at,
                "team_member_updated_at": team_member.updated_at,
            }
        })
    except Team.DoesNotExist:
        logger.warning(f"Team not found: team_id={team_id}")
        return Response({
            'error': _('Team not found.')
        }, status=status.HTTP_404_NOT_FOUND)
    except TeamMember.DoesNotExist:
        logger.warning(f"Team member not found: team_member_id={team_member_id}")
        return Response({
            'error': _('Team member not found.')
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        logger.error(f"Team member deactivation failed: {str(e)}")
        return Response({
            'error': _('Failed to deactivate team member. Please try again.')
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def activate_team_member(request) -> Response:
    """Handle team member activation via API with improved validation and error handling"""
    team_id = request.data.get('team_id', '').strip()
    team_member_id = request.data.get('team_member_id', '').strip()
    
    # Enhanced validation with better error messages
    if not team_id: 
        return Response({
            'error': _('Team ID is required.')
        }, status=status.HTTP_400_BAD_REQUEST)
    
    if not team_member_id:
        return Response({
            'error': _('Team member ID is required.')
        }, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        # Validate team_id and team_member_id are valid integers
        try:
            team_id = int(team_id)
            team_member_id = int(team_member_id)
        except ValueError:
            return Response({
                'error': _('Team ID and Team member ID must be valid numbers.')
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Get team and team member with single query optimization
        team = Team.objects.select_related('user').get(team_id=team_id)
        team_member = TeamMember.objects.select_related('team').get(team_member_id=team_member_id)
        
        # Check if user owns the team
        if team.user != request.user:
            logger.warning(f"Unauthorized team member activation attempt: user {request.user.email} tried to activate team member {team_member_id} in team {team_id}")  
            return Response({
                'error': _('You are not authorized to activate team members in this team.')
            }, status=status.HTTP_403_FORBIDDEN)
        
        # Check if team is active and not deleted
        if not team.is_active or team.is_deleted:
            return Response({
                'error': _('Cannot activate team members in inactive or deleted team.')
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Check if team member belongs to the specified team
        if team_member.team != team:
            return Response({
                'error': _('Team member does not belong to this team.')
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Activate the team member
        team_member.is_active = True
        team_member.is_deleted = False
        team_member.save(update_fields=['is_active', 'is_deleted', 'updated_at'])
        
        # Log successful activation
        logger.info(f"Team member {team_member.team_member_name} (ID: {team_member_id}) activated in team '{team.team_name}' (ID: {team_id}) by user: {request.user.email}")
        
        return Response({
            "message": _('Team member activated successfully!'),
            "team": {
                "team_id": team.team_id,
                "team_name": team.team_name,
                "team_member_count": team.get_team_member_count(),
                "team_created_at": team.created_at,
                "team_updated_at": team.updated_at,
            },
            "team_member": {
                "team_member_id": team_member.team_member_id,
                "team_member_name": team_member.team_member_name,
                "team_member_email": team_member.team_member_email,
                "team_member_role": team_member.team_member_role,
                "team_member_created_at": team_member.created_at,
                "team_member_updated_at": team_member.updated_at,
            }
        })
    except Team.DoesNotExist:
        logger.warning(f"Team not found: team_id={team_id}")
        return Response({
            'error': _('Team not found.')
        }, status=status.HTTP_404_NOT_FOUND)
    except TeamMember.DoesNotExist:
        logger.warning(f"Team member not found: team_member_id={team_member_id}")
        return Response({
            'error': _('Team member not found.')
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        logger.error(f"Team member activation failed: {str(e)}")
        return Response({
            'error': _('Failed to activate team member. Please try again.')
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_team_members(request) -> Response:
    """Handle team members retrieval via API with improved validation and error handling"""
    team_id = request.GET.get('team_id', '').strip()
    
    # Enhanced validation with better error messages
    if not team_id:
        return Response({
            'error': _('Team ID is required.')
        }, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        # Validate team_id is a valid integer
        try:
            team_id = int(team_id)
        except ValueError:
            return Response({
                'error': _('Team ID must be a valid number.')
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Get team and team member with single query optimization
        team = Team.objects.select_related('user').get(team_id=team_id)
        
        # Check if user owns the team
        if team.user != request.user:
            logger.warning(f"Unauthorized team member retrieval attempt: user {request.user.email} tried to retrieve team members from team {team_id}")
            return Response({
                'error': _('You are not authorized to retrieve team members from this team.')
            }, status=status.HTTP_403_FORBIDDEN)
        
        # Check if team is active and not deleted
        if not team.is_active or team.is_deleted:
            return Response({
                'error': _('Cannot retrieve team members from inactive or deleted team.')
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Get all active team members for the team
        team_members = team.get_all_team_members(include_inactive=True)
        
        # Log successful retrieval
        logger.info(f"Retrieved {len(team_members)} team members from team '{team.team_name}' (ID: {team_id}) by user: {request.user.email}")
        
        return Response({
            "message": _('Team members retrieved successfully!'),
            "team": {
                "team_id": team.team_id,
                "team_name": team.team_name,
                "team_member_count": team.get_team_member_count(),
                "team_created_at": team.created_at,
                "team_updated_at": team.updated_at,
            },
            "team_members": [
                {
                    "team_member_id": member.team_member_id,
                    "team_member_name": member.team_member_name,
                    "team_member_email": member.team_member_email,
                    "team_member_role": member.team_member_role,
                    "team_member_created_at": member.created_at,
                    "team_member_updated_at": member.updated_at,
                }
                for member in team_members
            ]
        })
    except Team.DoesNotExist:
        logger.warning(f"Team not found: team_id={team_id}")
        return Response({
            'error': _('Team not found.')
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        logger.error(f"Team member retrieval failed: {str(e)}")
        return Response({
            'error': _('Failed to retrieve team members. Please try again.')
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_team_agents(request) -> Response:
    """Handle team agents retrieval via API with improved validation and error handling"""
    team_id = request.GET.get('team_id', '').strip()
    
    # Enhanced validation with better error messages
    if not team_id:
        return Response({
            'error': _('Failed to retrieve team agents. Please try again.')
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    try:
        # Validate team_id is a valid integer
        try:
            team_id = int(team_id)
        except ValueError:
            return Response({
                'error': _('Team ID must be a valid number.')
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Get team and agents with single query optimization
        team = Team.objects.select_related('user').get(team_id=team_id)
        
        # Check if user owns the team
        if team.user != request.user:
            logger.warning(f"Unauthorized team agent retrieval attempt: user {request.user.email} tried to retrieve agents from team {team_id}")
            return Response({
                'error': _('You are not authorized to retrieve agents from this team.')
            }, status=status.HTTP_403_FORBIDDEN)
        
        # Check if team is active and not deleted
        if not team.is_active or team.is_deleted:
            return Response({
                'error': _('Cannot retrieve agents from inactive or deleted team.')
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Get all active agents for the team
        agents = TeamAgent.objects.filter(team=team, is_active=True, is_deleted=False).select_related('agent').order_by('-created_at')
        
        # Log successful retrieval
        logger.info(f"Retrieved {len(agents)} agents from team '{team.team_name}' (ID: {team_id}) by user: {request.user.email}")
        
        return Response({
            "message": _('Team agents retrieved successfully!'),
            "team": {
                "team_id": team.team_id,
                "team_name": team.team_name,
                "team_agent_count": team.get_team_agent_count(),
                "team_created_at": team.created_at,
                "team_updated_at": team.updated_at,
            },
            "agents": [
                {
                    "agent_id": agent.agent_id,
                    "agent_name": agent.name,
                    "agent_created_at": agent.created_at,
                    "agent_updated_at": agent.updated_at,
                }
                for agent in agents
            ]
        })
    except Team.DoesNotExist:
        logger.warning(f"Team not found: team_id={team_id}")
        return Response({   
            'error': _('Failed to retrieve team members. Please try again.')
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    except Exception as e:
        logger.error(f"Team agent retrieval failed: {str(e)}")
        return Response({
            'error': _('Failed to retrieve team agents. Please try again.')
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_team_details(request) -> Response:
    """Handle team details retrieval via API with improved validation and error handling"""
    team_id = request.GET.get('team_id', '').strip()  # Changed from request.data to request.GET for GET requests
    
    # Enhanced validation with better error messages
    if not team_id:
        return Response({
            'error': _('Team ID is required.')
        }, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        # Validate team_id is a valid integer
        try:
            team_id = int(team_id)
        except ValueError:
            return Response({
                'error': _('Team ID must be a valid number.')
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Get team with optimized query using prefetch_related for better performance
        team = Team.objects.select_related('user').prefetch_related(
            'team_members', 'agents'
        ).get(team_id=team_id)
        
        # Check if user owns the team
        if team.user != request.user:
            logger.warning(f"Unauthorized team details retrieval attempt: user {request.user.email} tried to retrieve details of team {team_id}")
            return Response({
                'error': _('You are not authorized to retrieve details of this team.')
            }, status=status.HTTP_403_FORBIDDEN)
        
        # Check if team is active and not deleted
        if not team.is_active or team.is_deleted:
            return Response({
                'error': _('Cannot retrieve details of inactive or deleted team.')
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Get team members and agents using optimized methods
        team_members = team.get_team_members()  # Use the optimized method from model
        agents = TeamAgent.objects.filter(team=team, is_active=True, is_deleted=False).select_related('agent').order_by('-created_at')  # Direct query for better performance

        # Log successful retrieval
        logger.info(f"Retrieved team details for team '{team.team_name}' (ID: {team_id}) by user: {request.user.email}")
        
        return Response({
            "message": _('Team details retrieved successfully!'),
            "team": {
                "team_id": team.team_id,
                "team_name": team.team_name,
                "team_member_count": len(team_members),  # Use len() instead of database query
                "team_agent_count": len(agents),  # Use len() instead of database query
                "team_created_at": team.created_at,
                "team_updated_at": team.updated_at,
            },
            "team_members": [
                {
                    "team_member_id": member.team_member_id,
                    "team_member_name": member.team_member_name,
                    "team_member_email": member.team_member_email,  
                    "team_member_role": member.team_member_role,
                    "team_member_created_at": member.created_at,
                    "team_member_updated_at": member.updated_at,
                }
                for member in team_members
            ],
            "agents": [
                {
                    "agent_id": agent.agent_id,
                    "agent_name": agent.name,
                    "agent_created_at": agent.created_at,
                    "agent_updated_at": agent.updated_at,
                }
                for agent in agents
            ]
        })
    except Team.DoesNotExist:
        logger.warning(f"Team not found: team_id={team_id}")
        return Response({
            'error': _('Team not found.')  # Fixed error message to be more specific
        }, status=status.HTTP_404_NOT_FOUND)  # Changed to 404 for not found
    except Exception as e:
        logger.error(f"Team details retrieval failed for user {request.user.email}: {str(e)}")
        return Response({
            'error': _('Failed to retrieve team details. Please try again.')
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_team_list(request) -> Response:
    """Handle team list retrieval via API with improved validation and error handling"""
    user = request.user
    
    # Remove redundant authentication check since @permission_classes([IsAuthenticated]) handles this
    try:
        # Get all active teams for the user with optimized query
        teams = Team.objects.filter(
            user=user, 
            is_active=True, 
            is_deleted=False
        ).only('team_id', 'team_name', 'created_at', 'updated_at')  # Use only() for better performance
        
        # Log successful retrieval
        logger.info(f"Retrieved {len(teams)} active teams for user {user.email}")
        
        return Response({
            "message": _('Team list retrieved successfully!'),
            "teams": [
                {
                    "team_id": team.team_id,
                    "team_name": team.team_name,
                    "team_created_at": team.created_at,
                    "team_updated_at": team.updated_at,
                }
                for team in teams
            ]
        })
    except Exception as e:
        logger.error(f"Team list retrieval failed for user {user.email}: {str(e)}")
        return Response({
            'error': _('Failed to retrieve team list. Please try again.')
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def delete_team(request) -> Response:
    """Handle team deletion via API with improved validation and error handling"""
    team_id = request.data.get('team_id')
    
    # Enhanced validation with better error messages
    if not team_id:
        return Response({
            'error': _('Team ID is required.')
        }, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        # Validate team_id is a valid integer
        try:
            team_id = int(team_id)
        except ValueError:
            return Response({
                'error': _('Team ID must be a valid number.')
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Get team with optimized query using prefetch_related for better performance
        team = Team.objects.select_related('user').prefetch_related(
            'team_members', 'team_agents'
        ).get(team_id=team_id)
        
        # Check if user owns the team
        if team.user != request.user:
            logger.warning(f"Unauthorized team deletion attempt: user {request.user.email} tried to delete team {team_id}")
            return Response({
                'error': _('You are not authorized to delete this team.')
            }, status=status.HTTP_403_FORBIDDEN)
        
        # Check if team is active and not deleted
        if not team.is_active or team.is_deleted:
            return Response({
                'error': _('Cannot delete inactive or deleted team.')
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Soft delete the team instead of removing from database
        team.is_active = False
        team.is_deleted = True
        team.save(update_fields=['is_active', 'is_deleted', 'updated_at'])
        
        # Log successful deletion
        logger.info(f"Team '{team.team_name}' (ID: {team_id}) deleted by user: {request.user.email}")
        
        return Response({
            "message": _('Team deleted successfully!'),
            "team": {
                "team_id": team.team_id,
                "team_name": team.team_name,
                "team_created_at": team.created_at,
                "team_updated_at": team.updated_at,
            }
        })
    except Team.DoesNotExist:
        logger.warning(f"Team not found: team_id={team_id}")
        return Response({
            'error': _('Team not found.')
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        logger.error(f"Team deletion failed: {str(e)}")
        return Response({
            'error': _('Failed to delete team. Please try again.')
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_team_member(request) -> Response:
    """Handle team member deletion via API with improved validation and error handling"""
    team_id = request.data.get('team_id')
    team_member_id = request.data.get('team_member_id')
    
    # Enhanced validation with better error messages
    if not team_id:
        return Response({
            'error': _('Team ID is required.')
        }, status=status.HTTP_400_BAD_REQUEST)
    
    if not team_member_id:
        return Response({
            'error': _('Team member ID is required.')
        }, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        # Validate team_id and team_member_id are valid integers
        try:
            team_id = int(team_id)
            team_member_id = int(team_member_id)
        except ValueError:
            return Response({
                'error': _('Team ID and Team member ID must be valid numbers.')
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Get team and team member with single query optimization   
        team = Team.objects.select_related('user').prefetch_related(
            'team_members', 'team_agents'
        ).get(team_id=team_id)
        team_member = TeamMember.objects.get(team_member_id=team_member_id)
        
        # Check if user owns the team
        if team.user != request.user:
            logger.warning(f"Unauthorized team member deletion attempt: user {request.user.email} tried to delete team member {team_member_id} from team {team_id}")
            return Response({
                'error': _('You are not authorized to delete this team member.')
            }, status=status.HTTP_403_FORBIDDEN)
        
        # Check if team member is active and not deleted
        if not team_member.is_active or team_member.is_deleted:
            return Response({
                'error': _('Cannot delete inactive or deleted team member.')
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Soft delete the team member instead of removing from database
        team_member.is_active = False
        team_member.is_deleted = True
        team_member.save(update_fields=['is_active', 'is_deleted', 'updated_at'])
        
        # Log successful deletion
        logger.info(f"Team member {team_member.team_member_name} (ID: {team_member_id}) deleted from team '{team.team_name}' (ID: {team_id}) by user: {request.user.email}")
        
        return Response({
            "message": _('Team member deleted successfully!'),
            "team": {
                "team_id": team.team_id,
                "team_name": team.team_name,
                "team_created_at": team.created_at,
                "team_updated_at": team.updated_at,
            }
        })
    except Team.DoesNotExist:
        logger.warning(f"Team not found: team_id={team_id}")
        return Response({
            'error': _('Team not found.')
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        logger.error(f"Team deletion failed: {str(e)}")
        return Response({
            'error': _('Failed to delete team. Please try again.')
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def update_team_name(request) -> Response:
    """Handle team update via API with improved validation and error handling"""
    team_id = request.data.get('team_id')
    team_name = request.data.get('team_name')
    logger.info(f"Team ID: {team_id}")
    logger.info(f"Team Name: {team_name}")
    # Enhanced validation with better error messages
    if not team_id:
        return Response({
            'error': _('Team ID is required.')
        }, status=status.HTTP_400_BAD_REQUEST)
    
    if not team_name:
        return Response({
            'error': _('Team name is required.')
        }, status=status.HTTP_400_BAD_REQUEST)
        
    try:
        # Validate team_id is a valid integer
        try:
            team_id = int(team_id)
        except ValueError:
            return Response({
                'error': _('Team ID must be a valid number.')
            }, status=status.HTTP_400_BAD_REQUEST)
        logger.info(f"Team ID: {team_id}")
        # Get team with optimized query using prefetch_related for better performance   
        team = Team.objects.select_related('user').prefetch_related(
            'team_members', 'team_agents'
        ).get(team_id=team_id)
        logger.info(f"Team: {team}")
        
        # Check if user owns the team
        if team.user != request.user:
            logger.warning(f"Unauthorized team update attempt: user {request.user.email} tried to update team {team_id}")
            return Response({
                'error': _('You are not authorized to update this team.')
            }, status=status.HTTP_403_FORBIDDEN)
        
        # Check if team is active and not deleted
        if not team.is_active or team.is_deleted:
            return Response({
                'error': _('Cannot update inactive or deleted team.')
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Update team name
        team.team_name = team_name
        team.save(update_fields=['team_name', 'updated_at'])
        
        # Log successful update
        logger.info(f"Team '{team.team_name}' (ID: {team_id}) updated by user: {request.user.email}")
        
        return Response({
            "message": _('Team updated successfully!'),
            "team": {
                "team_id": team.team_id,
                "team_name": team.team_name,
                "team_created_at": team.created_at,
                "team_updated_at": team.updated_at,
            }
        })
    except Team.DoesNotExist:
        logger.warning(f"Team not found: team_id={team_id}")
        return Response({
            'error': _('Failed to delete team. Please try again.')
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def update_team_member_role(request) -> Response:
    """Handle team member role update via API with improved validation and error handling"""
    team_id = request.data.get('team_id')
    team_member_id = request.data.get('team_member_id')
    team_member_role = request.data.get('team_member_role')
    
    # Enhanced validation with better error messages
    if not team_id:
        return Response({
            'error': _('Team ID is required.')
        }, status=status.HTTP_400_BAD_REQUEST)
    
    if not team_member_id:
        return Response({
            'error': _('Team member ID is required.')
        }, status=status.HTTP_400_BAD_REQUEST)

    if not team_member_role:
        return Response({
            'error': _('Team member role is required.')
        }, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        # Validate team_id and team_member_id are valid integers
        try:
            team_id = int(team_id)
            team_member_id = int(team_member_id)
        except ValueError:
            return Response({
                'error': _('Team ID and Team member ID must be valid numbers.')
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Get team and team member with single query optimization
        team = Team.objects.select_related('user').prefetch_related(
            'team_members', 'team_agents'
        ).get(team_id=team_id)
        team_member = TeamMember.objects.get(team_member_id=team_member_id)
        
        # Check if user owns the team
        if team.user != request.user:
            logger.warning(f"Unauthorized team member role update attempt: user {request.user.email} tried to update role of team member {team_member_id} in team {team_id}")
            return Response({
                'error': _('You are not authorized to update this team member.')
            }, status=status.HTTP_403_FORBIDDEN)
        
        # Check if team member is active and not deleted
        if not team_member.is_active or team_member.is_deleted:
            return Response({
                'error': _('Cannot update inactive or deleted team member.')
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Check if team member belongs to the specified team
        if team_member.team != team:
            return Response({
                'error': _('Team member does not belong to this team.')
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Update team member role
        team_member.team_member_role = team_member_role
        team_member.save(update_fields=['team_member_role', 'updated_at'])
        
        # Log successful update
        logger.info(f"Team member {team_member.team_member_name} (ID: {team_member_id}) role updated to '{team_member_role}' in team '{team.team_name}' (ID: {team_id}) by user: {request.user.email}")
        
        return Response({
            "message": _('Team member role updated successfully!'),
            "team": {
                "team_id": team.team_id,
                "team_name": team.team_name,
                "team_created_at": team.created_at,
                "team_updated_at": team.updated_at,
            },
            "team_member": {
                "team_member_id": team_member.team_member_id,
                "team_member_name": team_member.team_member_name,
                "team_member_email": team_member.team_member_email,
                "team_member_role": team_member.team_member_role,
                "team_member_created_at": team_member.created_at,
                "team_member_updated_at": team_member.updated_at,
            }
        })
    except Team.DoesNotExist:
        logger.warning(f"Team not found: team_id={team_id}")
        return Response({
            'error': _('Failed to delete team. Please try again.')
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    except TeamMember.DoesNotExist:
        logger.warning(f"Team member not found: team_member_id={team_member_id}")
        return Response({
            'error': _('Failed to delete team member. Please try again.')
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    except Exception as e:
        logger.error(f"Team member role update failed: {str(e)}")
        return Response({
            'error': _('Failed to update team member role. Please try again.')
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def update_team_member_name(request) -> Response:
    """Handle team member name update via API with improved validation and error handling"""
    team_id = request.data.get('team_id')
    team_member_id = request.data.get('team_member_id')
    team_member_name = request.data.get('team_member_name')

    # Enhanced validation with better error messages
    if not team_id:
        return Response({
            'error': _('Team ID is required.')
        }, status=status.HTTP_400_BAD_REQUEST)
    
    if not team_member_id:
        return Response({
            'error': _('Team member ID is required.')
        }, status=status.HTTP_400_BAD_REQUEST)
    
    if not team_member_name:
        return Response({
            'error': _('Team member name is required.')
        }, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        # Validate team_id and team_member_id are valid integers
        try:
            team_id = int(team_id)
            team_member_id = int(team_member_id)
        except ValueError:
            return Response({
                'error': _('Team ID and Team member ID must be valid numbers.')
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Get team and team member with single query optimization
        team = Team.objects.select_related('user').prefetch_related(
            'team_members', 'team_agents'
        ).get(team_id=team_id)
        team_member = TeamMember.objects.get(team_member_id=team_member_id)
        
        # Check if user owns the team
        if team.user != request.user:
            logger.warning(f"Unauthorized team member name update attempt: user {request.user.email} tried to update name of team member {team_member_id} in team {team_id}")
            return Response({
                'error': _('You are not authorized to update this team member.')
            }, status=status.HTTP_403_FORBIDDEN)

        # Check if team member is active and not deleted
        if not team_member.is_active or team_member.is_deleted:
            return Response({
                'error': _('Cannot update inactive or deleted team member.')
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Check if team member belongs to the specified team
        if team_member.team != team:
            return Response({
                'error': _('Team member does not belong to this team.')
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Update team member name
        team_member.team_member_name = team_member_name
        team_member.save(update_fields=['team_member_name', 'updated_at'])
        
        # Log successful update
        logger.info(f"Team member {team_member.team_member_name} (ID: {team_member_id}) name updated to '{team_member_name}' in team '{team.team_name}' (ID: {team_id}) by user: {request.user.email}") 
        
        return Response({
            "message": _('Team member name updated successfully!'),
            "team": {
                "team_id": team.team_id,
                "team_name": team.team_name,
                "team_created_at": team.created_at,
                "team_updated_at": team.updated_at,
            },
            "team_member": {
                "team_member_id": team_member.team_member_id,
                "team_member_name": team_member.team_member_name,
                "team_member_email": team_member.team_member_email,
                "team_member_role": team_member.team_member_role,
                "team_member_created_at": team_member.created_at,
                "team_member_updated_at": team_member.updated_at,
            }
        })
    except Team.DoesNotExist:
        logger.warning(f"Team not found: team_id={team_id}")
        return Response({
            'error': _('Failed to delete team. Please try again.')
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    except TeamMember.DoesNotExist:
        logger.warning(f"Team member not found: team_member_id={team_member_id}")
        return Response({
            'error': _('Failed to delete team member. Please try again.')
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    except Exception as e:
        logger.error(f"Team member name update failed: {str(e)}")
        return Response({
            'error': _('Failed to update team member name. Please try again.')
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    