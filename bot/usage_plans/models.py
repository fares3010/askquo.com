from django.db import models
from django.contrib.auth.models import User
from django.forms import ValidationError
from django.utils import timezone
from datetime import timedelta
from django.conf import settings

class SubscriptionPlan(models.Model):
    """
    Model representing a subscription plan with pricing and feature details.
    """
    PERIOD_CHOICES = [
        ("daily", "Daily"),
        ("weekly", "Weekly"), 
        ("monthly", "Monthly"),
        ("yearly", "Yearly")
    ]

    plan_id = models.AutoField(primary_key=True)
    plan_name = models.CharField(max_length=255, help_text="Name of the subscription plan")
    plan_description = models.TextField(blank=True, null=True, help_text="Detailed description of the plan")
    plan_period = models.CharField(max_length=50, choices=PERIOD_CHOICES, help_text="Billing period of the plan")
    plan_tier = models.CharField(max_length=50, blank=True, null=True, help_text="Tier level of the plan")
    plan_price = models.DecimalField(max_digits=10, decimal_places=2, default=0, help_text="Price of the plan")
    plan_currency = models.CharField(max_length=10, default='USD', help_text="Currency for the plan price")
    plan_duration_days = models.IntegerField(blank=True, null=True, help_text="Duration of the plan in days")
    is_trial = models.BooleanField(default=False, help_text="Whether this is a trial plan")
    is_active = models.BooleanField(default=True, help_text="Whether the plan is currently active")
    meta_data = models.JSONField(blank=True, null=True, help_text="Additional metadata for the plan")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Subscription Plan"
        verbose_name_plural = "Subscription Plans"
        ordering = ['plan_price']
        indexes = [
            models.Index(fields=['is_active']),
            models.Index(fields=['plan_period']),
        ]

    def __str__(self):
        return self.plan_name or "Unknown Plan"

    @property
    def plan_features(self):
        """Returns active features for the plan."""
        return self.features.filter(is_active=True).values(
            "feature_name", "feature_type", "feature_description", "is_active"
        )

    def get_plan_details(self):
        """Returns basic plan details as a dictionary."""
        return {
            "plan_id": self.plan_id,
            "plan_name": self.plan_name,
            "plan_description": self.plan_description,
            "plan_tier": self.plan_tier,
            "plan_period": self.plan_period,
            "plan_price": self.plan_price,
            "plan_currency": self.plan_currency,
            "is_active": self.is_active,
            "is_trial": self.is_trial,
        }

    def is_valid(self):
        """Validates if the plan has all required fields."""
        return all([
            self.plan_name,
            self.plan_price is not None,
            self.plan_period in dict(self.PERIOD_CHOICES),
        ])

    def get_price_display(self):
        """Returns formatted price display string."""
        return f"{self.plan_price} {self.plan_currency}" if self.plan_price else "Free"

    def to_dict(self, include_features=False):
        """Converts plan to dictionary representation."""
        data = {
            "plan_id": self.plan_id,
            "plan_name": self.plan_name,
            "plan_description": self.plan_description,
            "plan_tier": self.plan_tier,
            "plan_period": self.plan_period,
            "plan_price": float(self.plan_price) if self.plan_price else 0,
            "plan_currency": self.plan_currency,
            "is_active": self.is_active,
            "is_trial": self.is_trial,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if include_features:
            data["features"] = list(self.plan_features)
        return data

    def get_plan_duration_days(self):
        """Returns plan duration in days based on period."""
        duration_map = {
            'daily': 1,
            'weekly': 7,
            'monthly': 30,
            'yearly': 365
        }
        return duration_map.get(self.plan_period, 0)

    def get_expiry_date(self, from_date=None):
        """Calculates expiry date from given date."""
        from_date = from_date or timezone.now()
        duration_days = self.get_plan_duration_days()
        return from_date + timedelta(days=duration_days) if duration_days else None

    def feature_count(self):
        """Returns count of active features."""
        return self.features.filter(is_active=True).count()

    def get_feature(self, feature_name):
        """Returns a specific feature by name."""
        return self.features.filter(is_active=True, feature_name=feature_name).first()

    def feature_limit(self, feature_name):
        """Returns limit for a specific feature."""
        feature = self.get_feature(feature_name)
        return feature.feature_limit if feature else None

    def feature_accessible(self, feature_name):
        """Checks if a feature is accessible."""
        feature = self.get_feature(feature_name)
        return bool(feature and feature.is_active)


class PlanFeature(models.Model):
    """
    Model representing features available in subscription plans.
    """
    plan = models.ForeignKey(SubscriptionPlan, on_delete=models.CASCADE, related_name='features')
    feature_name = models.CharField(max_length=255, help_text="Name of the feature")
    feature_type = models.CharField(max_length=50, blank=True, null=True, help_text="Type of the feature")
    feature_description = models.TextField(blank=True, null=True, help_text="Description of the feature")
    feature_limit = models.IntegerField(blank=True, null=True, help_text="Usage limit for the feature")
    is_active = models.BooleanField(default=True, help_text="Whether the feature is active")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ['plan', 'feature_name']
        ordering = ['feature_name']
        indexes = [
            models.Index(fields=['is_active']),
            models.Index(fields=['feature_name']),
        ]

    def __str__(self):
        return f"{self.feature_name} ({self.plan.plan_name})"

    def to_dict(self):
        """Converts feature to dictionary representation."""
        return {
            "feature_name": self.feature_name,
            "feature_type": self.feature_type,
            "feature_description": self.feature_description,
            "feature_limit": self.feature_limit,
            "is_active": self.is_active,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class UserSubscription(models.Model):
    """
    Model representing user subscriptions to plans.
    """
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='subscriptions')
    plan = models.ForeignKey(SubscriptionPlan, on_delete=models.CASCADE, related_name='subscriptions')
    subscription_id = models.AutoField(primary_key=True)
    stripe_subscription_id = models.CharField(max_length=255, blank=True, null=True, help_text="Stripe subscription ID")
    usage_start_date = models.DateTimeField(default=timezone.now, help_text="Start date of subscription")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True, help_text="Whether the subscription is active")
    is_deleted = models.BooleanField(default=False, help_text="Whether the subscription is deleted")
    meta_data = models.JSONField(blank=True, null=True, help_text="Additional metadata for the subscription")

    class Meta:
        verbose_name = "User Subscription"
        verbose_name_plural = "User Subscriptions"
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['is_active']),
            models.Index(fields=['user', 'is_active']),
        ]

    @property
    def usage_end_date(self):
        """Calculates subscription end date."""
        if self.plan and self.usage_start_date:
            return self.usage_start_date + timedelta(days=self.plan.get_plan_duration_days())
        return None

    def is_valid_subscription(self):
        """Validates if subscription is currently valid."""
        now = timezone.now()
        return (
            self.is_active 
            and not self.is_deleted 
            and self.usage_end_date 
            and self.usage_end_date > now
        )

    def __str__(self):
        return f"Subscription #{self.subscription_id} - {self.user.username} ({self.plan.plan_name})"

    def to_dict(self):
        """Converts subscription to dictionary representation."""
        return {
            "subscription_id": self.subscription_id,
            "user_id": self.user.id,
            "username": self.user.username,
            "plan_id": self.plan.plan_id,
            "plan_name": self.plan.plan_name,
            "usage_start_date": self.usage_start_date,
            "usage_end_date": self.usage_end_date,
            "is_active": self.is_active,
            "is_deleted": self.is_deleted,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def check_if_plan_expired(self):
        return bool(self.usage_end_date and timezone.now() > self.usage_end_date)
