#!/usr/bin/env python3
from dotenv import load_dotenv
load_dotenv()
import logging
from app.processor import DocumentProcessor
from app.chatbot import RAGChatbot
from app_config import settings
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("core_initializer")

try:
    processor_instance = DocumentProcessor(
        qdrant_host=settings.QDRANT_HOST,
        qdrant_port=settings.QDRANT_PORT,
        mongodb_uri=settings.MONGODB_URL,
        mongodb_db=settings.DB_NAME,
        collection=settings.QDRANT_COLLECTION,
        embedding_model=settings.EMBEDDING_MODEL
    )
    logger.info("DocumentProcessor instance initialized.")
    
    chatbot_instance = RAGChatbot(
        qdrant_host=settings.QDRANT_HOST,
        qdrant_port=settings.QDRANT_PORT,
        collection=settings.QDRANT_COLLECTION,
        embedding_model=settings.EMBEDDING_MODEL,
        score_threshold=0.1
    )
    logger.info("RAGChatbot (Groq-enabled) instance initialized.")
    
except Exception as e:
    logger.critical(f"CRITICAL: Core RAG component initialization failed: {e}", exc_info=True)
    raise