from django.db import models
from django.contrib.auth.models import User
from django.forms import ValidationError
from django.utils import timezone
from datetime import timedelta
from django.conf import settings
from django.core.exceptions import ValidationError as DjangoValidationError
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)


class StripeCustomer(models.Model):
    """
    Model representing a Stripe customer.
    """
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, 
        on_delete=models.CASCADE, 
        related_name='stripe_customers',
        help_text="User who owns this Stripe customer"
    )
    stripe_customer_id = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        unique=True,
        help_text="Stripe customer ID for this customer"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Stripe Customer"
        verbose_name_plural = "Stripe Customers"
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['stripe_customer_id']),
        ]

    def __str__(self):  
        return f"Stripe Customer #{self.stripe_customer_id} - {self.user.username}"
    
    def to_dict(self):
        """Converts customer to dictionary representation."""
        return {
            "stripe_customer_id": self.stripe_customer_id,
            "user_id": self.user.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    
    def is_valid(self):
        """Validates if the customer has all required fields."""
        try:
            return all([
                self.stripe_customer_id and self.stripe_customer_id.strip(),
                self.user and self.user.id,
            ])
        except Exception as e:  
            logger.error(f"Error validating customer {self.stripe_customer_id}: {e}")
            return False
        
    def save(self, *args, **kwargs):
        """Override save to automatically set stripe_customer_id."""
        if not self.stripe_customer_id:
            self.stripe_customer_id = self.user.id
        super().save(*args, **kwargs)

    def clean(self):
        """Custom validation for the model."""  
        super().clean()
        if not self.stripe_customer_id:
            raise DjangoValidationError("Stripe customer ID is required.")      

class SubscriptionPlan(models.Model):
    """
    Model representing a subscription plan with pricing and feature details.
    """
    plan_id = models.BigAutoField(primary_key=True)
    stripe_product_id = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        unique=True,
        help_text="Stripe product ID for this plan"
    )
    plan_name = models.CharField(
        max_length=255, 
        help_text="Name of the subscription plan",
        db_index=True
    )
    plan_description = models.TextField(
        blank=True, 
        null=True, 
        help_text="Detailed description of the plan"
    )
    plan_tier = models.CharField(
        max_length=50, 
        blank=True, 
        null=True, 
        help_text="Tier level of the plan"
    )
    is_trial = models.BooleanField(
        default=False, 
        help_text="Whether this is a trial plan"
    )
    is_active = models.BooleanField(
        default=True, 
        help_text="Whether the plan is currently active"
    )
    meta_data = models.JSONField(
        blank=True, 
        null=True, 
        help_text="Additional metadata for the plan"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Subscription Plan"
        verbose_name_plural = "Subscription Plans"
        ordering = ['plan_name']
        indexes = [
            models.Index(fields=['is_active']),
            models.Index(fields=['plan_tier']),
            models.Index(fields=['is_trial']),
        ]

    def __str__(self):
        return self.plan_name or "Unknown Plan"

    def clean(self):
        """Custom validation for the model."""
        super().clean()


    def save(self, *args, **kwargs):
        """Override save to include validation."""
        self.clean()
        super().save(*args, **kwargs)

    @property
    def plan_features(self):
        """Returns active features for the plan."""
        return self.features.filter(is_active=True)

    def get_plan_details(self):
        """Returns basic plan details as a dictionary."""
        return {
            "plan_id": self.plan_id,
            "plan_name": self.plan_name,
            "plan_description": self.plan_description,
            "is_active": self.is_active,
            "is_trial": self.is_trial,
            "features":{
                    feature.feature_name: {
                    "feature_description": feature.feature_description,
                    "is_active": feature.is_active,
                }
                for feature in self.features.filter(is_active=True)
            }
        }

    def is_valid(self):
        """Validates if the plan has all required fields."""
        try:
            return all([
                self.plan_name and self.plan_name.strip(),
            ])
        except Exception as e:
            logger.error(f"Error validating plan {self.plan_id}: {e}")
            return False

    def get_price_display(self):
        """Returns formatted price display string."""
        try:
            if self.prices.first().price_amount and self.prices.first().price_amount > 0:
                return f"{self.prices.first().price_amount} {self.prices.first().price_currency}"
            return "Free"
        except Exception as e:
            logger.error(f"Error formatting price for plan {self.plan_id}: {e}")
            return "Price unavailable"

    def to_dict(self, include_features=False):
        """Converts plan to dictionary representation."""
        try:
            data = {
                "plan_id": self.plan_id,
                "plan_name": self.plan_name,
                "plan_description": self.plan_description,
                "plan_tier": self.plan_tier,
                "is_active": self.is_active,
                "is_trial": self.is_trial,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
            }
            if include_features:
                data["features"] = list(self.plan_features)
            return data
        except Exception as e:
            logger.error(f"Error converting plan {self.plan_id} to dict: {e}")
            return {}


    def get_expiry_date(self, from_date=None):
        """Calculates expiry date from given date."""
        try:
            from_date = from_date or timezone.now()
            duration_days = self.prices.first().price_duration_days
            return from_date + timedelta(days=duration_days) if duration_days else None
        except Exception as e:
            logger.error(f"Error calculating expiry date for plan {self.plan_id}: {e}")
            return None

    def feature_count(self):
        """Returns count of active features."""
        try:
            return self.features.filter(is_active=True).count()
        except Exception as e:
            logger.error(f"Error counting features for plan {self.plan_id}: {e}")
            return 0

    def get_feature(self, feature_name):
        """Returns a specific feature by name."""
        try:
            logger.info(f"Getting feature {feature_name} for plan {self.plan_id}")
            return self.features.filter(
                is_active=True, 
                feature_name=feature_name
            ).first()
        except Exception as e:
            logger.error(f"Error getting feature {feature_name} for plan {self.plan_id}: {e}")
            return None

    def feature_limit(self, feature_name):
        """Returns limit for a specific feature.""" 
        try:
            feature = self.get_feature(feature_name)
            if feature:
                logger.info(f"Feature limit for {feature_name} in plan {self.plan_id}: {feature.feature_limit}")
                return feature.feature_limit
            logger.warning(f"Feature {feature_name} not found for plan {self.plan_id}")
            return None
        except Exception as e:
            logger.error(f"Error getting feature limit for {feature_name} in plan {self.plan_id}: {e}")
            return None

    def feature_accessible(self, feature_name):
        """Checks if a feature is accessible."""
        try:
            feature = self.get_feature(feature_name)
            return bool(feature and feature.is_active)
        except Exception as e:
            logger.error(f"Error checking feature accessibility for {feature_name} in plan {self.plan_id}: {e}")
            return False

class PlanPrice(models.Model):
    """
    Model representing a price for a subscription plan.
    """
    PERIOD_CHOICES = [
        ("monthly", "Monthly"),
        ("yearly", "Yearly")
    ]
    PRICE_TYPE_CHOICES = [
        ("one_time", "One Time"),
        ("recurring", "Recurring")
    ]
    DURATION_DAYS_CHOICES = [
        (30, "30 days"),
        (90, "90 days"),
        (180, "180 days"),
        (365, "365 days")
    ]
    plan = models.ForeignKey(
        SubscriptionPlan, 
        on_delete=models.CASCADE, 
        related_name='prices'
    )
    price_id = models.BigAutoField(primary_key=True)
    stripe_price_id = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        unique=True,    
        help_text="Stripe price ID for this plan"
    )
    price_amount = models.DecimalField(
        max_digits=10, 
        decimal_places=2, 
        default=Decimal('0.00'), 
        help_text="Price amount for the plan"
    )
    price_currency = models.CharField(
        max_length=10, 
        default='USD', 
        help_text="Currency for the plan price"
    )
    price_period = models.CharField(
        max_length=50, 
        blank=True, 
        null=True, 
        choices=PERIOD_CHOICES,
        help_text="Period of the price"
    )
    price_duration_days = models.IntegerField(
        default=30,
        choices=DURATION_DAYS_CHOICES,
        help_text="Duration of the price in days"
    )
    price_type = models.CharField(
        max_length=50, 
        blank=True, 
        null=True, 
        choices=PRICE_TYPE_CHOICES,
        help_text="Type of the price"
    )
    is_active = models.BooleanField(
        default=True, 
        help_text="Whether the price is active"
    )
    is_deleted = models.BooleanField(
        default=False, 
        help_text="Whether the price is deleted"
    )
    meta_data = models.JSONField(
        blank=True, 
        null=True, 
        help_text="Additional metadata for the price"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ['plan', 'stripe_price_id']
        ordering = ['price_amount']
        indexes = [
            models.Index(fields=['price_amount']),
        ]

    def __str__(self):
        return f"Price #{self.stripe_price_id} - {self.plan.plan_name}"

    def to_dict(self):
        """Converts price to dictionary representation."""
        return { 
            "price_id": self.stripe_price_id,
            "price_amount": self.price_amount,
            "price_currency": self.price_currency,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    
    def is_valid(self):
        """Validates if the price has all required fields."""
        try:
            return all([
                self.price_amount is not None and self.price_amount >= 0,
                self.price_currency and self.price_currency.strip(),
                self.plan and self.plan.is_valid(),
            ])
        except Exception as e:
            logger.error(f"Error validating price {self.stripe_price_id}: {e}")
            return False
        
    def clean(self):
        """Custom validation for the model."""
        super().clean()
        if self.price_amount < 0:
            raise DjangoValidationError("Price amount cannot be negative.")
        
        if not self.price_currency or not self.price_currency.strip():
            raise DjangoValidationError("Price currency is required.")

    def save(self, *args, **kwargs):
        """Override save to include validation."""
        self.clean()
        super().save(*args, **kwargs)

    def get_price_display(self):
        """Returns formatted price display string."""
        try:
            if self.price_amount and self.price_amount > 0:
                return f"{self.price_amount} {self.price_currency}"
            return "Free"
        except Exception as e:
            logger.error(f"Error formatting price for price {self.stripe_price_id}: {e}")
            return "Price unavailable"
        
    def get_price_amount(self):
        """Returns price amount as a float."""
        try:
            return float(self.price_amount) if self.price_amount else 0
        except Exception as e:
            logger.error(f"Error getting price amount for price {self.stripe_price_id}: {e}")
            return 0
        
    def get_price_currency(self):
        """Returns price currency."""
        try:
            return self.price_currency if self.price_currency else "USD"
        except Exception as e:  
            logger.error(f"Error getting price currency for price {self.stripe_price_id}: {e}")
            return "USD"
        
    def get_price_id(self):
        """Returns price ID."""
        try:
            return self.stripe_price_id if self.stripe_price_id else None
        except Exception as e:
            logger.error(f"Error getting price ID for price {self.stripe_price_id}: {e}")
            return None
        
    def get_plan_id(self):
        """Returns plan ID."""
        try:
            return self.plan.plan_id if self.plan else None
        except Exception as e:
            logger.error(f"Error getting plan ID for price {self.stripe_price_id}: {e}")
            return None
        
    def get_plan_name(self):
        """Returns plan name."""
        try:
            return self.plan.plan_name if self.plan else None
        except Exception as e:
            logger.error(f"Error getting plan name for price {self.stripe_price_id}: {e}")
            return None
        
    def get_plan_description(self):
        """Returns plan description."""
        try:
            return self.plan.plan_description if self.plan else None
        except Exception as e:  
            logger.error(f"Error getting plan description for price {self.stripe_price_id}: {e}")
            return None
        
    def get_plan_period(self):
        """Returns plan period."""
        try:
            return self.plan.plan_period if self.plan else None
        except Exception as e:
            logger.error(f"Error getting plan period for price {self.stripe_price_id}: {e}")
            return None
        


class PlanFeature(models.Model):
    """
    Model representing features available in subscription plans.
    """
    plan = models.ForeignKey(
        SubscriptionPlan, 
        on_delete=models.CASCADE, 
        related_name='features'
    )
    feature_name = models.CharField(
        max_length=255, 
        help_text="Name of the feature",
        db_index=True
    )
    feature_type = models.CharField(
        max_length=50, 
        blank=True, 
        null=True, 
        help_text="Type of the feature"
    )
    feature_description = models.TextField(
        blank=True, 
        null=True, 
        help_text="Description of the feature"
    )
    feature_limit = models.IntegerField(
        blank=True, 
        null=True, 
        help_text="Usage limit for the feature"
    )
    is_active = models.BooleanField(
        default=True, 
        help_text="Whether the feature is active"
    )
    is_deleted = models.BooleanField(
        default=False, 
        help_text="Whether the feature is deleted"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ['plan', 'feature_name']
        ordering = ['feature_name']
        indexes = [
            models.Index(fields=['is_active']),
            models.Index(fields=['feature_name']),
            models.Index(fields=['feature_type']),
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
            "is_deleted": self.is_deleted,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class UserSubscription(models.Model):
    """
    Model representing user subscriptions to plans.
    """
    subscription_id = models.BigAutoField(primary_key=True)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, 
        on_delete=models.CASCADE, 
        related_name='subscriptions',
        help_text="User who owns this subscription"
    )

    price = models.ForeignKey(
        PlanPrice,
        on_delete=models.CASCADE,
        related_name='subscriptions',
        help_text="Price associated with this subscription"
    )

    plan_name = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        help_text="Name of the plan"
    )

    stripe_subscription_id = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        unique=True,
        help_text="Stripe subscription ID for this subscription"
    )
    usage_start_date = models.DateTimeField(
        default=timezone.now, 
        help_text="Start date of subscription"
    )
    usage_end_date = models.DateTimeField(
        blank=True, 
        null=True, 
        help_text="End date of subscription (calculated automatically)"
    )
    tokens_usage = models.BigIntegerField(
        default=0,
        help_text="Total tokens used in this subscription period"
    )
    conversations_usage = models.BigIntegerField(
        default=0,
        help_text="Total conversations used in this subscription period"
    )
    agents_usage = models.BigIntegerField(
        default=0,
        help_text="Total agents used in this subscription period"
    )
    widgets_usage = models.BigIntegerField(
        default=0,
        help_text="Total widgets used in this subscription period"
    )
    team_members_usage = models.BigIntegerField(
        default=0,
        help_text="Total team members used in this subscription period"
    )
    is_trial = models.BooleanField(
        default=False, 
        help_text="Whether the subscription is a trial"
    )
    is_trial_expired = models.BooleanField(
        default=False, 
        help_text="Whether the trial has expired"
    )
    is_active = models.BooleanField(
        default=True, 
        help_text="Whether the subscription is active"
    )
    is_deleted = models.BooleanField(
        default=False, 
        help_text="Whether the subscription is deleted"
    )
    meta_data = models.JSONField(
        blank=True, 
        null=True, 
        help_text="Additional metadata for the subscription"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "User Subscription"
        verbose_name_plural = "User Subscriptions"
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['is_active']),
            models.Index(fields=['user', 'is_active']),
            models.Index(fields=['stripe_subscription_id']),
            models.Index(fields=['usage_start_date']),
            models.Index(fields=['usage_end_date']),
        ]


    def save(self, *args, **kwargs):
        """Override save to automatically calculate usage_end_date."""
        if not self.usage_end_date and self.price and self.usage_start_date:
            duration_days = self.price.price_duration_days or 30
            self.usage_end_date = self.usage_start_date + timedelta(days=duration_days)
        super().save(*args, **kwargs)

    def clean(self):
        """Custom validation for the model."""
        super().clean()
        if self.usage_start_date and self.usage_end_date:
            if self.usage_start_date >= self.usage_end_date:
                raise DjangoValidationError("End date must be after start date.")
                    
    def is_valid_subscription(self):
        """Validates if subscription is currently valid."""
        try:
            now = timezone.now()
            return (
                self.is_active 
                and not self.is_deleted 
                and self.usage_end_date 
                and self.usage_end_date > now
            )
        except Exception as e:
            logger.error(f"Error validating subscription {self.subscription_id}: {e}")
            return False

    def is_expired(self):
        """Check if the subscription has expired."""
        try:
            return bool(self.usage_end_date and timezone.now() > self.usage_end_date)
        except Exception as e:
            logger.error(f"Error checking expiration for subscription {self.subscription_id}: {e}")
            return False

    def get_remaining_days(self):
        """Returns remaining days in subscription."""
        try:
            if self.usage_end_date:
                remaining = self.usage_end_date - timezone.now()
                return max(0, remaining.days)
            return 0
        except Exception as e:
            logger.error(f"Error calculating remaining days for subscription {self.subscription_id}: {e}")
            return 0

    def __str__(self):
        plan_name = self.price.plan.plan_name if self.price and self.price.plan else "Unknown Plan"
        return f"Subscription #{self.subscription_id} - {self.user.email} ({plan_name})"

    def to_dict(self, include_metadata=False):
        """Converts subscription to dictionary representation."""
        try:
            data = {
                "subscription_id": self.subscription_id,
                "user_id": self.user.id,
                "stripe_subscription_id": self.stripe_subscription_id,
                "usage_start_date": self.usage_start_date,
                "usage_end_date": self.usage_end_date,
                "is_active": self.is_active,
                "is_deleted": self.is_deleted,
                "is_expired": self.is_expired(),
                "remaining_days": self.get_remaining_days(),
                "created_at": self.created_at,
                "updated_at": self.updated_at,
            }
            if include_metadata and self.meta_data:
                data["meta_data"] = self.meta_data
            return data
        except Exception as e:
            logger.error(f"Error converting subscription {self.subscription_id} to dict: {e}")
            return {}
    def get_plan_id(self):
        """Returns plan ID."""
        try:
            return self.price.plan.plan_id if self.price and self.price.plan else None
        except Exception as e:
            logger.error(f"Error getting plan ID for subscription {self.subscription_id}: {e}")
            return None
class StripeTransaction(models.Model):
    """
    Model representing a transaction in Stripe.
    """
    transaction_id = models.BigAutoField(primary_key=True)
    stripe_transaction_id = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        unique=True,
        help_text="Stripe transaction ID for this transaction"
    )
    user_email = models.EmailField(
        max_length=255,
        blank=True,
        null=True,
        help_text="Email of the user who owns this transaction"
    )
    stripe_customer_id = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        help_text="Stripe customer ID for this transaction"
    )
    transaction_amount = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        default=Decimal('0.00'),
        help_text="Amount of the transaction"
    )
    transaction_currency = models.CharField(
        max_length=10,
        default='USD',
        help_text="Currency of the transaction"
    )
    transaction_status = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        help_text="Status of the transaction"
    )
    transaction_receipt_url = models.URLField(
        blank=True,
        null=True,
        help_text="Receipt URL for the transaction"
    )
    transaction_address_country = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        help_text="Country of the transaction"
    )
    transaction_metadata = models.JSONField(
        blank=True,
        null=True,
        default=dict,
        help_text="Additional metadata for the transaction"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Transaction #{self.transaction_id} - {self.user_email}"

    def to_dict(self):
        """Converts transaction to dictionary representation."""
        return {
            "transaction_id": self.transaction_id,
            "transaction_amount": self.transaction_amount,
            "transaction_currency": self.transaction_currency,
            "transaction_status": self.transaction_status,
            "transaction_receipt_url": self.transaction_receipt_url,
            "created_at": self.created_at,
        }