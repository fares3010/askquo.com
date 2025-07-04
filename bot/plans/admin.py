from django.contrib import admin
from .models import SubscriptionPlan, PlanFeature, UserSubscription, PlanPrice, StripeCustomer
from django.contrib.auth.models import User
from django.utils import timezone


class PlanFeatureInline(admin.TabularInline):
    model = PlanFeature
    extra = 1
    readonly_fields = ('created_at', 'updated_at')
    fields = ('feature_name', 'feature_type', 'feature_description', 'feature_limit', 'is_active', 'created_at', 'updated_at')


class PlanPriceInline(admin.TabularInline):
    model = PlanPrice
    extra = 1
    readonly_fields = ('created_at', 'updated_at')
    fields = ('price_amount', 'price_currency', 'price_period', 'price_duration_days', 'price_type', 'is_active', 'created_at', 'updated_at')


class SubscriptionPlanAdmin(admin.ModelAdmin):
    list_display = ('plan_name', 'get_plan_price', 'get_plan_period', 'is_active', 'created_at')
    list_filter = ('is_active', 'is_trial', 'plan_tier')
    search_fields = ('plan_name', 'plan_description')
    readonly_fields = ('created_at', 'updated_at')
    inlines = [PlanFeatureInline, PlanPriceInline]
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('plan_name', 'plan_description', 'plan_tier')
        }),
        ('Status', {
            'fields': ('is_active', 'is_trial')
        }),
        ('Additional Data', {
            'fields': ('meta_data',),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )

    def get_plan_price(self, obj):
        """Display the first active price for the plan"""
        first_price = obj.prices.filter(is_active=True).first()
        if first_price:
            return f"{first_price.price_amount} {first_price.price_currency}"
        return "No price set"
    get_plan_price.short_description = 'Price'

    def get_plan_period(self, obj):
        """Display the period for the plan"""
        first_price = obj.prices.filter(is_active=True).first()
        if first_price and first_price.price_period:
            return first_price.price_period
        return "No period set"
    get_plan_period.short_description = 'Period'

    def has_add_permission(self, request, obj=None):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False

    def has_view_permission(self, request, obj=None):
        return True


class PlanPriceAdmin(admin.ModelAdmin):
    list_display = ('plan', 'price_amount', 'price_currency', 'price_period', 'price_type', 'is_active', 'created_at')
    list_filter = ('is_active', 'price_period', 'price_type', 'price_currency')
    search_fields = ('plan__plan_name', 'stripe_price_id')
    readonly_fields = ('created_at', 'updated_at')
    
    fieldsets = (
        ('Plan Information', {
            'fields': ('plan',)
        }),
        ('Price Information', {
            'fields': ('stripe_price_id', 'price_amount', 'price_currency', 'price_period', 'price_duration_days', 'price_type')
        }),
        ('Status', {
            'fields': ('is_active', 'is_deleted')
        }),
        ('Additional Data', {
            'fields': ('meta_data',),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )


class PlanFeatureAdmin(admin.ModelAdmin):
    list_display = ('feature_name', 'plan', 'feature_type', 'feature_limit', 'is_active', 'created_at')
    list_filter = ('is_active', 'feature_type')
    search_fields = ('feature_name', 'feature_description', 'plan__plan_name')
    readonly_fields = ('created_at', 'updated_at')
    
    fieldsets = (
        ('Feature Information', {
            'fields': ('plan', 'feature_name', 'feature_type', 'feature_description', 'feature_limit')
        }),
        ('Status', {
            'fields': ('is_active',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )

    def has_add_permission(self, request, obj=None):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False

    def has_view_permission(self, request, obj=None):
        return True


class StripeCustomerAdmin(admin.ModelAdmin):
    list_display = ('user', 'stripe_customer_id', 'created_at', 'updated_at')
    list_filter = ('created_at',)
    search_fields = ('user__username', 'user__email', 'stripe_customer_id')
    readonly_fields = ('created_at', 'updated_at')
    
    fieldsets = (
        ('Customer Information', {
            'fields': ('user', 'stripe_customer_id')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )

    def has_add_permission(self, request, obj=None):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False

    def has_view_permission(self, request, obj=None):
        return True


class UserSubscriptionAdmin(admin.ModelAdmin):
    list_display = ('user', 'get_plan_name', 'get_plan_price', 'is_active', 'usage_start_date', 'usage_end_date', 'created_at')
    list_filter = ('is_active', 'is_deleted')
    search_fields = ('user__username', 'user__email', 'stripe_subscription_id')
    readonly_fields = ('created_at', 'updated_at', 'usage_start_date', 'usage_end_date')
    
    fieldsets = (
        ('Subscription Information', {
            'fields': ('user', 'price_id', 'plan_name', 'subscription_id', 'stripe_subscription_id', 'usage_start_date', 'usage_end_date')
        }),
        ('Usage Tracking', {
            'fields': ('tokens_usage', 'conversations_usage', 'agents_usage', 'integrations_usage'),
            'classes': ('collapse',)
        }),
        ('Status', {
            'fields': ('is_active', 'is_deleted')
        }),
        ('Additional Information', {
            'fields': ('meta_data',),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )

    def get_plan_name(self, obj):
        """Display the plan name from the related price"""
        if obj.price_id and obj.price_id.plan:
            return obj.price_id.plan.plan_name
        return "No plan"
    get_plan_name.short_description = 'Plan'

    def get_plan_price(self, obj):
        """Display the plan price"""
        if obj.price_id:
            return f"{obj.price_id.price_amount} {obj.price_id.price_currency}"
        return "No price"
    get_plan_price.short_description = 'Price'

    def has_add_permission(self, request, obj=None):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False

    def has_view_permission(self, request, obj=None):
        return True


# Register models with their admin classes
admin.site.register(SubscriptionPlan, SubscriptionPlanAdmin)
admin.site.register(PlanFeature, PlanFeatureAdmin)
admin.site.register(UserSubscription, UserSubscriptionAdmin)
admin.site.register(PlanPrice, PlanPriceAdmin)
admin.site.register(StripeCustomer, StripeCustomerAdmin)