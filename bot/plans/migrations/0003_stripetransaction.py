# Generated by Django 5.2 on 2025-06-28 11:49

from decimal import Decimal
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('plans', '0002_rename_price_id_usersubscription_price'),
    ]

    operations = [
        migrations.CreateModel(
            name='StripeTransaction',
            fields=[
                ('transaction_id', models.BigAutoField(primary_key=True, serialize=False)),
                ('stripe_transaction_id', models.CharField(blank=True, help_text='Stripe transaction ID for this transaction', max_length=255, null=True, unique=True)),
                ('user_email', models.EmailField(blank=True, help_text='Email of the user who owns this transaction', max_length=255, null=True)),
                ('stripe_customer_id', models.CharField(blank=True, help_text='Stripe customer ID for this transaction', max_length=255, null=True)),
                ('transaction_amount', models.DecimalField(decimal_places=2, default=Decimal('0.00'), help_text='Amount of the transaction', max_digits=10)),
                ('transaction_currency', models.CharField(default='USD', help_text='Currency of the transaction', max_length=10)),
                ('transaction_status', models.CharField(blank=True, help_text='Status of the transaction', max_length=255, null=True)),
                ('transaction_receipt_url', models.URLField(blank=True, help_text='Receipt URL for the transaction', null=True)),
                ('transaction_address_country', models.CharField(blank=True, help_text='Country of the transaction', max_length=255, null=True)),
                ('transaction_metadata', models.JSONField(blank=True, default=dict, help_text='Additional metadata for the transaction', null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]
