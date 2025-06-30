from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from pydantic import EmailStr
from typing import List, Dict
from app_config import settings
import logging

logger = logging.getLogger(__name__)

conf = ConnectionConfig(
    MAIL_USERNAME=settings.MAIL_USERNAME.get_secret_value(),
    MAIL_PASSWORD=settings.MAIL_PASSWORD.get_secret_value(),
    MAIL_FROM=settings.MAIL_FROM.get_secret_value(),
    MAIL_PORT=settings.MAIL_PORT,
    MAIL_SERVER=settings.MAIL_SERVER,
    MAIL_STARTTLS=True,
    MAIL_SSL_TLS=False,
    USE_CREDENTIALS=True,
    VALIDATE_CERTS=True
)

async def send_enquiry_email(enquiry_details: Dict, conversation_summary: str):
    try:
        subject = f"New Inquiry from {enquiry_details.get('name', 'N/A')} ({enquiry_details.get('email', 'N/A')})"
        
        body = f"""New customer inquiry received:

Customer Name: {enquiry_details.get('name', 'N/A')}
Customer Email: {enquiry_details.get('email', 'N/A')}

----------------------------------------

Conversation Summary:
{conversation_summary}"""
        
        message = MessageSchema(
            subject=subject,
            recipients=[settings.MAIL_ENQUIRY_RECIPIENT.get_secret_value()],
            body=body,
            subtype="plain"
        )
        
        fm = FastMail(conf)
        await fm.send_message(message)
        
        logger.info(f"Inquiry email sent successfully to {settings.MAIL_ENQUIRY_RECIPIENT.get_secret_value()}")
        
    except Exception as e:
        logger.error("Failed to send inquiry email", exc_info=True)