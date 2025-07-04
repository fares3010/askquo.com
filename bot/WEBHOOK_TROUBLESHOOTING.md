# Stripe Webhook Troubleshooting Guide

## Current Issue
Webhook signature verification failing with error: "No signatures found matching the expected signature for payload"

## Step-by-Step Troubleshooting

### 1. **Verify Environment Variables**

Check your `.env` file contains:
```bash
STRIPE_SECRET_KEY=sk_test_...  # or sk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...
```

**Verify in Python:**
```python
import os
from dotenv import load_dotenv
load_dotenv()

print("STRIPE_SECRET_KEY:", "✓ Present" if os.getenv('STRIPE_SECRET_KEY') else "✗ Missing")
print("STRIPE_WEBHOOK_SECRET:", "✓ Present" if os.getenv('STRIPE_WEBHOOK_SECRET') else "✗ Missing")

webhook_secret = os.getenv('STRIPE_WEBHOOK_SECRET')
if webhook_secret:
    print(f"Webhook secret format: {'✓ Valid' if webhook_secret.startswith('whsec_') else '✗ Invalid'}")
    print(f"Webhook secret length: {len(webhook_secret)}")
```

### 2. **Check System Time Synchronization**

The logs show timestamp `1751028551` (January 2025) which is in the future.

**Windows:**
```cmd
# Run as Administrator
w32tm /query /status
w32tm /resync
```

**Linux/Mac:**
```bash
sudo ntpdate -s time.nist.gov
# or
sudo chronyd sources -v
```

**Verify current time:**
```python
import time
print(f"Current timestamp: {int(time.time())}")
print(f"Your webhook timestamp: 1751028551")
print(f"Difference: {int(time.time()) - 1751028551} seconds")
```

### 3. **Test Webhook Secret Manually**

Create `test_signature.py`:
```python
import os
import time
import hmac
import hashlib
import stripe
from dotenv import load_dotenv

load_dotenv()

def test_signature():
    webhook_secret = os.getenv('STRIPE_WEBHOOK_SECRET')
    if not webhook_secret:
        print("❌ STRIPE_WEBHOOK_SECRET not found")
        return
    
    # Test payload
    payload = '{"id":"evt_test","object":"event","type":"test"}'
    timestamp = int(time.time())
    
    # Create signature manually
    signed_payload = f"{timestamp}.{payload}"
    signature = hmac.new(
        webhook_secret.encode('utf-8'),
        signed_payload.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    sig_header = f"t={timestamp},v1={signature}"
    
    # Test with Stripe
    stripe.api_key = os.getenv('STRIPE_SECRET_KEY')
    
    try:
        event = stripe.Webhook.construct_event(
            payload.encode('utf-8'),
            sig_header,
            webhook_secret
        )
        print("✅ Signature verification works!")
        return True
    except Exception as e:
        print(f"❌ Signature verification failed: {e}")
        return False

if __name__ == "__main__":
    test_signature()
```

### 4. **Update Stripe Webhook Endpoint**

In your Stripe Dashboard:

1. Go to **Developers > Webhooks**
2. Update your webhook endpoint URL to use `/webhook-raw/` instead of `/webhook/`
3. Or create a new endpoint: `https://yourdomain.com/plans/webhook-raw/`

### 5. **Test with Stripe CLI**

Install Stripe CLI and test locally:

```bash
# Install Stripe CLI (Windows)
# Download from: https://github.com/stripe/stripe-cli/releases

# Login
stripe login

# Forward webhooks to your local server
stripe listen --forward-to localhost:8000/plans/webhook/

# In another terminal, trigger a test event
stripe trigger customer.created
```

### 6. **Debugging Steps in Order**

1. **Environment Check:**
   ```bash
   python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('Webhook secret exists:', bool(os.getenv('STRIPE_WEBHOOK_SECRET')))"
   ```

2. **Time Check:**
   ```bash
   python -c "import time; print('Current timestamp:', int(time.time()))"
   ```

3. **Manual Signature Test:**
   ```bash
   python test_signature.py
   ```

4. **Live Webhook Test:**
   - Use ngrok or similar to expose localhost
   - Update Stripe webhook URL
   - Test with real Stripe events

### 7. **Common Issues and Solutions**

| Issue | Solution |
|-------|----------|
| Future timestamp | Sync system clock |
| Invalid webhook secret | Regenerate in Stripe Dashboard |
| Payload modification | Use `webhook_raw` endpoint |
| Middleware interference | Add middleware bypass |
| Wrong signature format | Check v1 vs v0 signatures |

### 8. **Emergency Bypass (Development Only)**

For immediate testing, temporarily disable signature verification:

```python
# In stripe_gateway.py, modify _verify_webhook_signature method
def _verify_webhook_signature(self, payload, signature) -> Tuple[bool, str, Any]:
    """TEMPORARY: Bypass signature verification for debugging"""
    import json
    
    # WARNING: Only for development/debugging
    if os.getenv('STRIPE_BYPASS_SIGNATURE_VERIFICATION') == 'true':
        logger.warning("🚨 BYPASSING SIGNATURE VERIFICATION - DEVELOPMENT ONLY")
        try:
            event = json.loads(payload)
            return True, None, event
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON payload: {str(e)}", None
    
    # Continue with normal verification...
```

Add to your `.env`:
```
STRIPE_BYPASS_SIGNATURE_VERIFICATION=true
```

**⚠️ IMPORTANT: Remove this bypass before production!**

### 9. **Monitor Logs**

After implementing fixes, monitor your logs for:
```
✅ "Webhook signature verified successfully"
❌ "Signature verification failed"
⚠️  "Time difference warnings"
```

### 10. **Final Verification**

1. ✅ System time is synchronized
2. ✅ Webhook secret is correct format (whsec_...)
3. ✅ Using webhook_raw endpoint
4. ✅ Stripe CLI test passes
5. ✅ Live webhook test passes

## Next Steps

Once signature verification works:
1. Revert to normal webhook endpoint (`/webhook/`)
2. Remove any bypass mechanisms
3. Test with real subscription events
4. Monitor production webhooks

## Need Help?

If issues persist:
1. Check Stripe Dashboard webhook logs
2. Enable verbose logging in Django
3. Use Stripe CLI for real-time debugging
4. Contact Stripe support with webhook endpoint details 