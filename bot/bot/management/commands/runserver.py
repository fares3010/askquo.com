import os
import logging
from typing import Any
from django.core.management.commands.runserver import Command as RunserverCommand
from pyngrok import ngrok

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_ngrok() -> None:
    """
    Configure and start ngrok tunnel with error handling and logging.
    Updates ALLOWED_HOSTS dynamically with ngrok domain.
    """
    try:
        auth_token = os.getenv('NGROK_AUTH_TOKEN')
        if not auth_token:
            logger.warning("NGROK_AUTH_TOKEN not found in environment variables")
            return

        ngrok.set_auth_token(auth_token)
        tunnel = ngrok.connect(8000)
        
        logger.info("=== Ngrok tunnel established ===")
        logger.info(f"Public URL: {tunnel.public_url}")
        logger.info("=== End Ngrok tunnel info ===")
        
        from django.conf import settings
        ngrok_domain = tunnel.public_url.replace('https://', '').replace('http://', '')
        if ngrok_domain not in settings.ALLOWED_HOSTS:
            settings.ALLOWED_HOSTS.append(ngrok_domain)
            logger.info(f"Added {ngrok_domain} to ALLOWED_HOSTS")
            
    except Exception as e:
        logger.error(f"Failed to setup ngrok: {str(e)}", exc_info=True)

class Command(RunserverCommand):
    """Enhanced runserver command with ngrok integration."""
    def handle(self, *args: Any, **options: Any) -> None:
        if not os.getenv('DISABLE_NGROK'):
            setup_ngrok()
        super().handle(*args, **options) 