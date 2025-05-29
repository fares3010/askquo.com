from django.db import models
from django.contrib.auth.models import AbstractUser, PermissionsMixin
from django.utils.translation import gettext_lazy as _
from django.core.mail import send_mail
from django.conf import settings
from django.utils import timezone
import logging

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