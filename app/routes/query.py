from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field
from typing import List, Annotated, Optional, Dict, Tuple
import asyncio
from functools import partial
import logging
import uuid
from datetime import datetime, timezone
from app.api_server import chatbot_instance
from app.database import get_db
from pymongo.database import Database
from app_config import settings
from qdrant_client.models import PointStruct

logger = logging.getLogger("query_router")
router = APIRouter()

async def _save_chat_interaction(
    db: Database,
    session_id: str,
    user_question: str,
    bot_answer: str,
    retrieved_contexts: Optional[List[Dict]] = None,
    llm_params: Optional[Dict] = None,
    interaction_type: str = "query"
):
    interaction_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc)

    mongo_collection = db[settings.MONGODB_CHAT_HISTORY_COLLECTION]
    chat_log_mongo = {
        "_id": interaction_id,
        "session_id": session_id,
        "user_question": user_question,
        "bot_answer": bot_answer,
        "retrieved_contexts": retrieved_contexts or [],
        "timestamp": timestamp,
        "llm_params_used": llm_params or {},
        "endpoint": "query",
        "interaction_type": interaction_type
    }
    
    try:
        mongo_collection.insert_one(chat_log_mongo)
        logger.info(f"Query interaction {interaction_id} saved to MongoDB.")
    except Exception as e:
        logger.error(f"Failed to save query interaction {interaction_id} to MongoDB: {e}", exc_info=True)

    try:
        text_to_embed = f"User: {user_question}\nBot: {bot_answer}"
        vector = chatbot_instance.encoder.encode(text_to_embed).tolist()
        
        qdrant_payload = {
            "mongo_id": interaction_id,
            "session_id": session_id,
            "user_question_preview": user_question[:200],
            "full_interaction_text": text_to_embed,
            "timestamp_iso": timestamp.isoformat(),
            "endpoint": "query",
            "interaction_type": interaction_type
        }
        
        point = PointStruct(
            id=interaction_id,
            vector=vector,
            payload=qdrant_payload
        )
        
        chatbot_instance.qdrant.upsert(
            collection_name=chatbot_instance.qdrant_chat_collection,
            points=[point],
            wait=False
        )
        logger.info(f"Query interaction {interaction_id} embedding saved to Qdrant.")
    except Exception as e:
        logger.error(f"Failed to save query interaction {interaction_id} to Qdrant: {e}", exc_info=True)
    
    return interaction_id

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500,
                         example="What does this document explain?")
    session_id: Optional[str] = Field(None, description="Optional session ID to group conversations.")
    history: Optional[List[Tuple[Optional[str], str]]] = Field(None, description="Optional conversation history.")
    top_k: Annotated[int, Field(ge=1, le=10)] = Field(
        default=5,
        description="Number of context chunks to retrieve (1-10)"
    )
    temperature: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.3,
        description="LLM creativity control (0.0-1.0)"
    )
    max_tokens: Annotated[int, Field(ge=100, le=2000)] = Field(
        default=512,
        description="Maximum tokens in the response (100-2000)"
    )

class QueryResponse(BaseModel):
    question: str
    answer: str
    contexts: List[dict]
    metadata: dict
    session_id: str
    interaction_id: str

async def _execute_query_async(
    question: str,
    session_id: str,
    history: Optional[List[Tuple[Optional[str], str]]],
    top_k: int,
    temperature: float,
    max_tokens: int
):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        partial(
            chatbot_instance.query,
            question=question,
            session_id=session_id,
            history=history,
            top_k=top_k,
            temperature=temperature,
            max_tokens=max_tokens
        )
    )

@router.post("/", response_model=QueryResponse, tags=["Query"])
async def handle_query(request: QueryRequest, db: Database = Depends(get_db)):
    try:
        logger.info(f"New query received: '{request.question[:100]}' (Session: {request.session_id})...")
        current_session_id = request.session_id or str(uuid.uuid4())
        
        query_result = await _execute_query_async(
            question=request.question,
            session_id=current_session_id,
            history=request.history,
            top_k=request.top_k,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        bot_response_text = query_result.get("answer", "Error: No answer found.")
        retrieved_contexts_for_log = query_result.get("contexts", [])
        
        llm_params = {
            "top_k": request.top_k,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens
        }

        interaction_id = await _save_chat_interaction(
            db,
            current_session_id,
            request.question,
            bot_response_text,
            retrieved_contexts_for_log,
            llm_params,
            interaction_type="query"
        )
        
        response_payload = {
            "question": query_result["metadata"]["question"],
            "answer": bot_response_text,
            "contexts": retrieved_contexts_for_log,
            "metadata": {
                "top_k": query_result["metadata"]["top_k"],
                "temperature": query_result["metadata"]["temperature"],
                "max_tokens": query_result["metadata"].get("max_tokens", request.max_tokens),
                "context_count": len(retrieved_contexts_for_log),
                "total_context_length": sum(len(ctx.get("text", "")) for ctx in retrieved_contexts_for_log),
                "retrieved_chat_history_count": query_result["metadata"].get("retrieved_chat_history_count", 0)
            },
            "session_id": current_session_id,
            "interaction_id": interaction_id
        }
        
        logger.info(f"Query processed successfully for: '{request.question[:50]}' - "
                   f"Retrieved {len(retrieved_contexts_for_log)} contexts")
        return QueryResponse(**response_payload)
        
    except Exception as e:
        logger.error(f"Query failed for '{request.question[:100]}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Question processing error: {str(e)}"
        )