import redis
import json
import logging
from typing import Optional
from app_config import settings

logger = logging.getLogger(__name__)

try:
    r = redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=0,
        decode_responses=True
    )
    r.ping()
    logger.info("Successfully connected to Redis")
except redis.exceptions.ConnectionError as e:
    logger.critical("Failed to connect to Redis", exc_info=True)
    raise ConnectionError("Unable to connect to Redis server. Please check your Redis configuration.")
except Exception as e:
    logger.critical("Unexpected error during Redis initialization", exc_info=True)
    raise RuntimeError("Failed to initialize Redis connection")


def set_session_state(session_id: str, state_data: dict, expiry_seconds: int = 3600):
    """
    Store session state data in Redis with expiration.
    
    Args:
        session_id: Unique session identifier
        state_data: Dictionary containing session state
        expiry_seconds: Expiration time in seconds (default: 3600)
    """
    try:
        json_data = json.dumps(state_data)
        r.set(f"session:{session_id}", json_data, ex=expiry_seconds)
        logger.debug(f"Updated session state for session_id: {session_id}")
    except Exception as e:
        logger.error(f"Failed to set session state for session_id: {session_id}", exc_info=True)


def get_session_state(session_id: str) -> Optional[dict]:
    """
    Retrieve session state data from Redis.
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        Dictionary containing session state or None if not found
    """
    try:
        result = r.get(f"session:{session_id}")
        if result is not None:
            logger.debug(f"Retrieved session state for session_id: {session_id}")
            return json.loads(result)
        else:
            logger.debug(f"No session state found for session_id: {session_id}")
            return None
    except Exception as e:
        logger.error(f"Failed to get session state for session_id: {session_id}", exc_info=True)
        return None


def clear_session_state(session_id: str):
    """
    Delete session state data from Redis.
    
    Args:
        session_id: Unique session identifier
    """
    try:
        r.delete(f"session:{session_id}")
        logger.debug(f"Cleared session state for session_id: {session_id}")
    except Exception as e:
        logger.error(f"Failed to clear session state for session_id: {session_id}", exc_info=True)