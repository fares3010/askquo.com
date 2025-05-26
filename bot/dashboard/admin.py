from django.contrib import admin
from .models import DashboardStats
@admin.register(DashboardStats)
class DashboardStatsAdmin(admin.ModelAdmin):
    list_display = ('dashboard_id', 'user', 'dashboard_type', 'created_at', 'is_active', 'is_deleted')
    list_filter = ('dashboard_type', 'is_active', 'is_deleted', 'created_at')
    search_fields = ('user__username', 'dashboard_type')
    ordering = ('-created_at',)
    readonly_fields = ('dashboard_id', 'created_at', 'updated_at')
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('dashboard_id', 'user', 'dashboard_type')
        }),
        ('Status', {
            'fields': ('is_active', 'is_deleted')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
        ('Configuration', {
            'fields': ('dashboard_config', 'dashboard_layout'),
            'classes': ('collapse',)
        })
    )

    def has_add_permission(self, request):
        return False

    def has_delete_permission(self, request, obj=None):
        return False

    def has_change_permission(self, request, obj=None):
        return False
