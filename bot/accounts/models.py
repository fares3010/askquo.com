from django.db import models
from django.contrib.auth.models import AbstractUser, PermissionsMixin
from django.utils.translation import gettext_lazy as _
from django.core.mail import send_mail
from django.conf import settings
from django.utils import timezone
import logging
from create_agent.models import Agent

logger = logging.getLogger(__name__)

class CustomUser(AbstractUser, PermissionsMixin):
    """
    Custom user model that extends the default Django user model.
    Uses email as the unique identifier instead of username.
    """
    username = None
    email = models.EmailField(_('email address'), unique=True)
    full_name = models.CharField(_('full name'), max_length=255, blank=True)
    profile_image = models.ImageField(
        _('profile image'),
        upload_to='profile_images/',
        null=True,
        blank=True
    )
    profile_updated_at = models.DateTimeField(
        _('profile updated at'),
        default=timezone.now
    )

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['full_name']

    class Meta:
        verbose_name = _('user')
        verbose_name_plural = _('users')
        ordering = ['-date_joined']
        
    def __str__(self):
        return self.email
    
    def get_full_name(self):
        """Return the full name of the user."""
        return self.full_name or super().get_full_name()
    
    def get_short_name(self):
        """Return the short name of the user."""
        return self.first_name or self.email.split('@')[0]
    
    def email_user(self, subject, message, from_email=None, **kwargs):
        """Send an email to the user with proper error handling."""
        if not from_email:
            from_email = settings.DEFAULT_FROM_EMAIL
        try:
            send_mail(subject, message, from_email, [self.email], **kwargs)
        except Exception as e:
            logger.error(f"Failed to send email to {self.email}: {str(e)}")
    
    def get_profile_image_url(self):
        """Return the profile image URL or default image URL."""
        if self.profile_image and hasattr(self.profile_image, 'url'):
            return self.profile_image.url
        return f'{settings.STATIC_URL}images/default_profile.png'
    
    def update_profile_image(self, image_file):
        """Update the user's profile image and timestamp."""
        self.profile_image = image_file
        self.profile_updated_at = timezone.now()
        self.save(update_fields=['profile_image', 'profile_updated_at'])

class Team(models.Model):
    """
    Model representing a team.
    A team is a group of users who share the same subscription.
    """
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='teams',
        help_text="The user this team belongs to."
    )
    team_id = models.AutoField(
        primary_key=True,
        help_text="Primary key for the team."
    )
    team_name = models.CharField(
        max_length=255,
        unique=True,
        help_text="Name of the team."
    )
    is_active = models.BooleanField(
        default=True,
        help_text="Whether the team is active."
    )
    is_deleted = models.BooleanField(
        default=False,
        help_text="Whether the team is deleted."
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="Timestamp when the team was created."
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        help_text="Timestamp when the team was updated."
    )
    meta_data = models.JSONField(
        blank=True,
        null=True,
        help_text="Additional metadata for the team."
    )

    class Meta:
        verbose_name = "Team"
        verbose_name_plural = "Teams"
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', 'is_active']),
            models.Index(fields=['team_name']),
            models.Index(fields=['created_at']),
        ]

    def __str__(self):
        return self.team_name
    
    def get_team_members(self):
        """Return all active team members of the team."""
        return self.team_members.filter(is_active=True, is_deleted=False)
    
    def get_team_member_count(self):
        """Return the number of active team members in the team."""
        return self.team_members.filter(is_active=True, is_deleted=False).count()
    
    def get_team_member_by_email(self, email):
        """Return the active team member with the given email."""
        return self.team_members.get(
            team_member_email=email, 
            is_active=True, 
            is_deleted=False
        )
    
    def get_team_member_by_id(self, member_id):
        """Return the active team member with the given id."""
        return self.team_members.get(
            team_member_id=member_id, 
            is_active=True, 
            is_deleted=False
        )
    
    def get_team_member_by_name(self, name):
        """Return the active team member with the given name."""
        return self.team_members.get(
            team_member_name=name, 
            is_active=True, 
            is_deleted=False
        )
    
    def get_team_member_by_role(self, role):    
        """Return the active team members with the given role."""
        return self.team_members.filter(
            team_member_role=role, 
            is_active=True, 
            is_deleted=False
        )
    
    def get_all_team_members(self, include_inactive=False):
        """Return all team members, optionally including inactive ones."""
        queryset = self.team_members.filter(is_deleted=False)
        if not include_inactive:
            queryset = queryset.filter(is_active=True)
        return queryset
    
    def add_team_member(self, name, email, role, **kwargs):
        """Add a new team member to the team."""
        return self.team_members.create(
            team_member_name=name,
            team_member_email=email,
            team_member_role=role,
            **kwargs
        )
    
    def remove_team_member(self, member_id):
        """Soft delete a team member by setting is_deleted=True."""
        try:
            member = self.team_members.get(team_member_id=member_id)
            member.is_deleted = True
            member.is_active = False
            member.save(update_fields=['is_deleted', 'is_active', 'updated_at'])
            return True
        except TeamMember.DoesNotExist:
            return False
    
    def deactivate_team_member(self, member_id):
        """Deactivate a team member by setting is_active=False."""
        try:
            member = self.team_members.get(team_member_id=member_id)
            member.is_active = False
            member.save(update_fields=['is_active', 'updated_at'])
            return True
        except TeamMember.DoesNotExist:
            return False
        
    def get_team_agent_count(self):
        """Return the number of agents in the team."""
        return self.team_agents.filter(is_active=True, is_deleted=False).count()
    
    def get_team_agent_by_id(self, agent_id):
        """Return the agent with the given id."""
        return self.team_agents.get(agent_id=agent_id)
    
    def get_team_agent_by_name(self, agent_name):
        """Return the agent with the given name."""
        return self.team_agents.get(agent_name=agent_name)
    


class TeamMember(models.Model):
    """
    Model representing a team member.
    """
    team = models.ForeignKey(
        Team,
        on_delete=models.CASCADE,
        related_name='team_members',
        help_text="The team this team member belongs to."
    )
    team_member_id = models.AutoField(
        primary_key=True,
        help_text="Primary key for the team member."
    )
    team_member_name = models.CharField(
        max_length=255,
        help_text="Name of the team member."
    )
    team_member_email = models.EmailField(
        max_length=255,
        help_text="Email of the team member."
    )
    team_member_role = models.CharField(
        max_length=100,
        choices=[
            ('admin', 'Admin'),
            ('member', 'Member'),
            ('viewer', 'Viewer'),
        ],
        default='member',
        help_text="Role of the team member."
    )
    is_active = models.BooleanField(
        default=True,
        help_text="Whether the team member is active."
    )
    is_deleted = models.BooleanField(
        default=False,
        help_text="Whether the team member is deleted."
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="Timestamp when the team member was created."
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        help_text="Timestamp when the team member was updated."
    )
    meta_data = models.JSONField(
        blank=True,
        null=True,
        help_text="Additional metadata for the team member."
    )
    
    class Meta:
        verbose_name = "Team Member"
        verbose_name_plural = "Team Members"
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['team', 'is_active']),
            models.Index(fields=['team_member_email']),
            models.Index(fields=['team_member_role']),
            models.Index(fields=['created_at']),
        ]
        unique_together = ['team', 'team_member_email']
    
    def __str__(self):
        return f"{self.team_member_name} ({self.team.team_name})"
    
    
    @property
    def is_admin(self):
        """Check if the team member has admin role."""
        return self.team_member_role == 'admin'
    
    @property
    def is_viewer(self):
        """Check if the team member has viewer role."""
        return self.team_member_role == 'viewer'
    
    def promote_to_admin(self):
        """Promote team member to admin role."""
        self.team_member_role = 'admin'
        self.save(update_fields=['team_member_role', 'updated_at'])
    
    def demote_to_member(self):
        """Demote team member to member role."""
        self.team_member_role = 'member'
        self.save(update_fields=['team_member_role', 'updated_at'])


class TeamAgent(models.Model):
    """
    Model representing a team agent.
    """
    team_agent_id = models.AutoField(
        primary_key=True,
        help_text="Primary key for the team agent."
    )
    team = models.ForeignKey(
        Team,
        on_delete=models.CASCADE,
        related_name='team_agents',
        help_text="The team this team agent belongs to."
    )
    agent = models.ForeignKey(
        Agent,
        on_delete=models.CASCADE,
        related_name='team_agents',
        help_text="The agent this team agent belongs to."
    )
    is_active = models.BooleanField(
        default=True,
        help_text="Whether the team agent is active."
    )
    is_deleted = models.BooleanField(
        default=False,
        help_text="Whether the team agent is deleted."
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="Timestamp when the team agent was created."
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        help_text="Timestamp when the team agent was updated."
    )
    meta_data = models.JSONField(
        blank=True,
        null=True,
        help_text="Additional metadata for the team agent."
    )
    
    class Meta:
        verbose_name = "Team Agent"
        verbose_name_plural = "Team Agents"
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['team', 'agent', 'is_active']),
            models.Index(fields=['created_at']),
        ]
        unique_together = ['team', 'agent']
    
    def __str__(self):
        return f"{self.team.team_name} - {self.agent.name}"
    
    
    