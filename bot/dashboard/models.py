from django.core.cache import cache
from django.db import models
from django.conf import settings
from django.contrib.auth.models import User
from conversations.models import Conversation
from create_agent.models import Agent
from django.utils import timezone

class DashboardStats(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='dashboard_stats')
    dashboard_id = models.AutoField(primary_key=True)
    dashboard_type = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    is_deleted = models.BooleanField(default=False)
    
    def total_conversations(self):
        cache_key = 'total_conversations_monthly'
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        now = timezone.now()
        conversations = Conversation.objects.filter(
            created_at__year=now.year,
            created_at__month=now.month
        ).count()

        cache.set(cache_key, conversations, timeout= 60 * 10)  # Cache for 10 minutes (optional)

        return conversations
    
    def last_week_conversations(self):
        cache_key = f'last_week_conversations_{self.user.id}'
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        last_week_conversations = []
        day_name=[]
        now = timezone.now()
        for i in range(7):
            day_datetime = now - timezone.timedelta(days=i)
            day_name.append(day_datetime.strftime("%A"))  # Get the full name of the day
            # Get the number of conversations for that day
            no_conversations = Conversation.objects.filter( created_at__year=day_datetime.year,
                created_at__month=day_datetime.month,
                created_at__day=day_datetime.day,
            ).count()
            last_week_conversations.append(no_conversations)

        return dict(zip(day_name,last_week_conversations))   

    def last_week_responses(self):
        cache_key = f'last_week_responses_{self.user.id}'
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        responses = []
        now = timezone.now()
        for i in range(7):
            day_datetime = now - timezone.timedelta(days=i)
            # Get the number of responses for that day
            no_responses = Conversation.objects.filter( created_at__year=day_datetime.year,
                created_at__month=day_datetime.month,
                created_at__day=day_datetime.day,
            ).aggregate(daily_total_responses=sum("number_of_agent_messages"))["daily_total_responses"] or 0
            responses.append(no_responses) 
            
        return responses
            
    def conversations_change_rate(self):
        cache_key = 'conversations_change_rate'
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        now = timezone.now()
        last_month = now - timezone.timedelta(days=30)
        current_month_conversations = Conversation.objects.filter(
            created_at__year=now.year,
            created_at__month=now.month
        ).count()

        last_month_conversations = Conversation.objects.filter(
            created_at__year=last_month.year,
            created_at__month=last_month.month
        ).count()

        if last_month_conversations == 0:
            rate = 100.0 if current_month_conversations > 0 else 0.0
        elif current_month_conversations == 0:
            rate = -100.0 if last_month_conversations > 0 else 0.0
        else:
            rate = (current_month_conversations - last_month_conversations) / last_month_conversations * 100

        cache.set(cache_key, rate, timeout=60 * 10)
        return rate
    
    def check_conversations_rate_ispositive(self): 
        rate = self.conversations_change_rate()
        if rate > 0:
            return True
        else:
            return False

    def get_conversations(self):
        Conversations = Conversation.objects.all()
        if not Conversations:
            return 0
        return Conversations

    def active_conversations(self):
        return Conversation.objects.filter(is_active=True).count()
    
    def active_conversations_change_rate(self):
        cache_key = 'active_conversations_change_rate'
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        now = timezone.now()
        last_month = now - timezone.timedelta(days=30)
        current_month_conversations = Conversation.objects.filter(
            created_at__year=now.year,
            created_at__month=now.month,
            is_active=True
        ).count()

        last_month_conversations = Conversation.objects.filter(
            created_at__year=last_month.year,
            created_at__month=last_month.month,
            is_active=True
        ).count()

        if last_month_conversations == 0:
            rate = 100.0 if current_month_conversations > 0 else 0.0
        elif current_month_conversations == 0:
            rate = -100.0 if last_month_conversations > 0 else 0.0
        else:
            rate = (current_month_conversations - last_month_conversations) / last_month_conversations * 100

        cache.set(cache_key, rate, timeout=60 * 10)
        return rate
    
    def check_active_conversations_rate_ispositive(self):
        rate = self.active_conversations_change_rate()
        if rate > 0:
            return True
        else:
            return False

    def inactive_conversations(self):
        return Conversation.objects.filter(is_active=False).count()

    def total_of_agents(self):
        return Agent.objects.count()

    def active_agents(self):
        return Agent.objects.filter(is_active=True).count()

    def inactive_agents(self):
        return Agent.objects.filter(is_active=False).count()

    def avg_response_time(self):
        cache_key = 'avg_response_time'
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        conversations = self.get_conversations()
        if not conversations:
            return 0

        total_response_time = sum(conv.user_response_time() for conv in conversations)
        avg = total_response_time / len(conversations)

        cache.set(cache_key, avg, timeout=300)  # cache for 5 minutes
        return avg
    
    def avg_response_time_change_rate(self):
        cache_key = 'avg_response_time_change_rate'
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        now = timezone.now()
        last_month = now - timezone.timedelta(days=30)
        current_month_avg_response_time = self.avg_response_time()

        last_month_avg_response_time = Conversation.objects.filter(
            created_at__year=last_month.year,
            created_at__month=last_month.month
        ).aggregate(models.Avg('response_time'))['response_time__avg']

        if last_month_avg_response_time == 0:
            rate = 100.0 if current_month_avg_response_time > 0 else 0.0
        elif current_month_avg_response_time == 0:
            rate = -100.0 if last_month_avg_response_time > 0 else 0.0
        else:
            rate = (current_month_avg_response_time - last_month_avg_response_time) / last_month_avg_response_time * 100

        cache.set(cache_key, rate, timeout=60 * 10)
        return rate
    
    def check_rate_ispositive_avg_response_time(self):
        rate = self.avg_response_time_change_rate()
        if rate > 0:
            return True
        else:
            return False

    def user_satisfaction_rate(self):
        cache_key = 'user_satisfaction_rate'
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        conversations = self.get_conversations()
        if not conversations:
            return 0

        total_satisfaction = sum(conv.feedback_rate() for conv in conversations)
        rate = total_satisfaction / len(conversations)

        cache.set(cache_key, rate, timeout=300)
        return rate

    def __str__(self):
        return f"DashboardStats(User Name:{self.user.get_full_name()})"
    
    def user_satisfaction_change_rate(self):
        cache_key = 'user_satisfaction_change_rate'
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        now = timezone.now()
        last_month = now - timezone.timedelta(days=30)
        current_month_satisfaction_rate = self.user_satisfaction_rate()

        last_month_satisfaction_rate = Conversation.objects.filter(
            created_at__year=last_month.year,
            created_at__month=last_month.month
        ).aggregate(models.Avg('satisfaction'))['satisfaction__avg']

        if last_month_satisfaction_rate == 0:
            rate = 100.0 if current_month_satisfaction_rate > 0 else 0.0
        elif current_month_satisfaction_rate == 0:
            rate = -100.0 if last_month_satisfaction_rate > 0 else 0.0
        else:
            rate = (current_month_satisfaction_rate - last_month_satisfaction_rate) / last_month_satisfaction_rate * 100

        cache.set(cache_key, rate, timeout=60 * 10)
        return rate
    
    def check_rate_ispositive_user_satisfaction(self):
        rate = self.user_satisfaction_change_rate()
        if rate > 0:
            return True
        else:
            return False
         
