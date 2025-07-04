from django.urls import path
from . import views

app_name = 'accounts'

urlpatterns = [
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('csrf/', views.auth_csrf_token, name='csrf'),
    path('logout/', views.logout_view, name='logout'),
    path('profile/', views.profile_view, name='profile'),
    path('profile/image/', views.update_profile_image, name='update_profile_image'),
    path('profile/delete/', views.delete_account, name='delete_account'),
    path('team/create/', views.create_team, name='create_team'),
    path('team/add-member/', views.add_team_member, name='add_team_member'),
    path('team/assign-agent/', views.assign_team_agent, name='assign_team_agent'),
    path('team/remove-agent/', views.remove_team_agent, name='remove_team_agent'),
    path('team/remove-member/', views.remove_team_member, name='remove_team_member'),
    path('team/deactivate-member/', views.deactivate_team_member, name='deactivate_team_member'),
    path('team/activate-member/', views.activate_team_member, name='activate_team_member'),
    path('team/members/', views.get_team_members, name='get_team_members'),
    path('team/agents/', views.get_team_agents, name='get_team_agents'),
    path('team/details/', views.get_team_details, name='get_team_details'),
    path('team/list/', views.get_team_list, name='get_team_list'),
    path('team/delete/', views.delete_team, name='delete_team'),
    path('team/delete-member/', views.delete_team_member, name='delete_team_member'),
    path('team/update-name/', views.update_team_name, name='update_team_name'),
    path('team/update-member-role/', views.update_team_member_role, name='update_team_member_role'),
    path('team/update-member-name/', views.update_team_member_name, name='update_team_member_name'),
]