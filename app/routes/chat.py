import asyncio
import json
from functools import partial
from fastapi import APIRouter, HTTPException, status, Depends, BackgroundTasks
from pydantic import BaseModel, Field, EmailStr, TypeAdapter
from typing import Optional, List, Dict, Tuple, Any
import logging
import uuid
from datetime import datetime, timezone
from app.api_server import chatbot_instance
from app.database import get_db
from pymongo.database import Database
from app_config import settings
from qdrant_client.models import PointStruct
from app.models.inquiry_model import Inquiry
import app.email_service as email_service
from app.state_manager import get_session_state, set_session_state, clear_session_state
from pymongo.errors import PyMongoError
import groq
from bson import ObjectId

logger = logging.getLogger("chat_router")
router = APIRouter()

LLM_INTENT_ROUTER_SYSTEM_PROMPT = """You are an intent classification system. Your goal is to differentiate between information-seeking intent and contact-seeking intent.

Return ONLY a JSON object with a single boolean field:
{"is_inquiry": true} - if the user wants to contact someone or make an inquiry
{"is_inquiry": false} - if the user is asking for information or knowledge

Information-seeking examples (return false):
- "Who is the CEO?"
- "What are your services?"
- "Explain how machine learning works"
- "Tell me about your company"
- "I want to know about pricing"
- "How does this product work?"

Contact-seeking examples (return true):
- "I need to speak to a person"
- "Can I contact sales?"
- "Put me in touch with support"
- "I want to talk to someone"
- "Connect me with your team"

Rule: If the user's query is structured as a question to gain knowledge (who, what, where, when, why, how), it is ALWAYS information-seeking (false), regardless of the topic.

Return only the JSON, no other text."""

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = Field(None, description="Session identifier for conversation continuity")
    history: Optional[List[Tuple[Optional[str], str]]] = Field(None, description="Previous conversation history")
    top_k: int = 5
    temperature: float = 0.2
    max_tokens: int = 512

class ChatResponse(BaseModel):
    answer: str
    session_id: str
    interaction_id: str

async def route_intent_for_enquiry(question: str) -> bool:
    try:
        response = chatbot_instance.llm_client.chat.completions.create(
            model=settings.GROQ_MODEL_NAME,
            messages=[
                {"role": "system", "content": LLM_INTENT_ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": question}
            ],
            temperature=0.1,
            max_tokens=50
        )
        
        raw_response = response.choices[0].message.content.strip()
        logger.debug(f"Raw LLM intent response: {raw_response}")
        
        try:
            parsed_response = json.loads(raw_response)
            return parsed_response.get("is_inquiry", False)
        except json.JSONDecodeError as json_err:
            logger.error(f"Failed to parse LLM intent response as JSON: '{raw_response}'. Error: {json_err}", exc_info=True)
            return False
            
    except Exception as e:
        logger.error(f"Error in intent routing: {e}", exc_info=True)
        return False

async def handle_enquiry_flow(
    session_id: str,
    user_message: str,
    current_session_state: Dict,
    db: Database,
    background_tasks: BackgroundTasks,
    full_conversation_history: Optional[List[Tuple[Optional[str], str]]] = None
) -> Tuple[str, str]:    
    try:
        inquiries_collection = db[settings.MONGODB_INQUIRIES_COLLECTION]
        inquiry_record_id = current_session_state.get("inquiry_record_id")
        
        inquiry_doc = None
        inquiry: Optional[Inquiry] = None

        if inquiry_record_id:
            logger.debug(f"DEBUG: Attempting to find inquiry record with ID: {inquiry_record_id[:8]}...")
            inquiry_doc = await asyncio.to_thread(
                inquiries_collection.find_one, 
                {"_id": inquiry_record_id}
            )

            if not inquiry_doc:
                logger.warning(f"Inquiry record {inquiry_record_id[:8]}... not found in DB but referenced in session state. Clearing session and restarting flow.")
                clear_session_state(session_id)
                return "I apologize, but there was an issue with your inquiry session. Please start over.", "inquiry_error"
            
            try:
                inquiry = Inquiry(**inquiry_doc)
            except Exception as model_err:
                logger.error(f"Failed to create Inquiry model from retrieved document for {inquiry_record_id[:8]}...: {model_err}", exc_info=True)
                clear_session_state(session_id)
                return "There was an error processing your inquiry. Please start over.", "inquiry_error"
        
        if not inquiry_doc:
            inquiry = Inquiry(
                session_id=session_id,
                original_question=current_session_state.get("original_question", user_message),
                status="awaiting_name",
                conversation_history_snapshot=full_conversation_history or []
            )
            
            try:
                doc_to_insert = inquiry.model_dump()
                doc_to_insert['_id'] = doc_to_insert.pop('id')

                result = await asyncio.to_thread(
                    inquiries_collection.insert_one,
                    doc_to_insert
                )
                
                if result.inserted_id != inquiry.id:
                    logger.error(
                        f"CRITICAL: ID mismatch during inquiry creation for session {session_id[:8]}. "
                        f"Pydantic ID: {inquiry.id}, DB Inserted ID: {result.inserted_id}. "
                        "This indicates a problem with the insertion logic."
                    )
                    clear_session_state(session_id)
                    return "I'm sorry, a critical error occurred while saving your inquiry. Please start over.", "inquiry_critical_error"

                logger.debug(f"DEBUG: MongoDB insert_one for inquiry record {inquiry.id[:8]}... completed successfully.")
                
                current_session_state.update({
                    "inquiry_record_id": inquiry.id,
                    "inquiry_active": True,
                    "inquiry_status": "awaiting_name"
                })
                set_session_state(session_id, current_session_state)
                
                logger.info(f"New inquiry flow started and record {inquiry.id[:8]}... created for session {session_id[:8]}...")
                return "Thank you for your inquiry. To help you better, I'll need some information. What is your name?", "inquiry_name_request"
                
            except PyMongoError as db_err:
                logger.error(f"Database error creating initial inquiry record: {db_err}", exc_info=True)
                clear_session_state(session_id)
                return "I'm sorry, there was a technical issue saving your inquiry. Please try again later.", "inquiry_db_error"
                
        else:
            inquiry.updated_at = datetime.now(timezone.utc)
            response_message = ""
            interaction_type = "inquiry_progress"
            
            if inquiry.status == "awaiting_name":
                inquiry.user_name = user_message.strip()
                inquiry.status = "awaiting_email"
                
                update_fields_for_db = {
                    "user_name": inquiry.user_name,
                    "status": inquiry.status,
                    "updated_at": inquiry.updated_at
                }
                
                response_message = "Thank you! What is your email address?"
                interaction_type = "inquiry_email_request"
                
            elif inquiry.status == "awaiting_email":
                try:
                    validated_email = TypeAdapter(EmailStr).validate_python(user_message.strip())
                    inquiry.user_email = str(validated_email)
                    inquiry.status = "completed"
                    inquiry.updated_at = datetime.now(timezone.utc)
                    
                    email_details_for_template = {
                        "name": inquiry.user_name,
                        "email": inquiry.user_email,
                        "original_question": inquiry.original_question,
                        "session_id": session_id
                    }
                    
                    email_conversation_summary = f"""
New Inquiry Received:

Customer Name: {inquiry.user_name or 'Not Provided'}
Customer Email: {inquiry.user_email or 'Not Provided'}
Original Question/Trigger: {inquiry.original_question or 'N/A'}
Session ID: {session_id}

--- Conversation History Snapshot ---
"""
                    if full_conversation_history:
                        for u_msg, b_msg in full_conversation_history:
                            if u_msg is not None: 
                                email_conversation_summary += f"User: {u_msg}\n"
                            if b_msg is not None: 
                                email_conversation_summary += f"Bot: {b_msg}\n"
                    else:
                        email_conversation_summary += "No conversation snapshot available.\n"
                    
                    background_tasks.add_task(
                        email_service.send_enquiry_email,
                        email_details_for_template,
                        email_conversation_summary
                    )
                    
                    current_session_state["inquiry_done_in_session"] = True
                    clear_session_state(session_id) 
                    
                    logger.info(f"Inquiry completed for session {session_id[:8]}... by {inquiry.user_email}")
                    response_message = f"Thank you, {inquiry.user_name}! Your inquiry has been submitted and someone will get back to you at {inquiry.user_email} soon. Is there anything else I can help with?"
                    interaction_type = "inquiry_completed"
                    
                except ValueError as email_err:
                    logger.warning(f"Invalid email provided for inquiry {inquiry.id[:8]}...: {user_message}. Error: {email_err}", exc_info=True)
                    inquiry.status = "email_validation_failed"
                    inquiry.updated_at = datetime.now(timezone.utc)
                    update_fields_for_db = {"status": inquiry.status, "updated_at": inquiry.updated_at}
                    
                    try:
                        await asyncio.to_thread(
                            inquiries_collection.update_one,
                            {"_id": inquiry.id},
                            {"$set": update_fields_for_db}
                        )
                        logger.debug(f"Updated inquiry record {inquiry.id[:8]}... status to {inquiry.status}.")
                    except PyMongoError as update_err:
                        logger.error(f"Failed to update inquiry record status to email_validation_failed for {inquiry.id[:8]}...: {update_err}", exc_info=True)
                    
                    response_message = "That doesn't look like a valid email address. Please provide a valid email address:"
                    interaction_type = "inquiry_email_invalid"
                
            else:
                logger.warning(f"Unexpected inquiry status: {inquiry.status} for session {session_id[:8]}... Clearing state.")
                clear_session_state(session_id)
                response_message = "I'm sorry, something went wrong with the inquiry process. Please try starting over."
                interaction_type = "inquiry_error"
            
            if inquiry.status != "completed": 
                try:
                    update_fields_final = {
                        "status": inquiry.status,
                        "updated_at": inquiry.updated_at,
                    }
                    if inquiry.user_name is not None:
                        update_fields_final["user_name"] = inquiry.user_name
                    if inquiry.user_email is not None:
                        update_fields_final["user_email"] = inquiry.user_email

                    update_result = await asyncio.to_thread(
                        inquiries_collection.update_one,
                        {"_id": inquiry.id},
                        {"$set": update_fields_final}
                    )
                    logger.debug(f"Updated inquiry record {inquiry.id[:8]}...: {update_result.modified_count} documents modified. Status: {inquiry.status}.")
                        
                except PyMongoError as update_err:
                    logger.error(f"MongoDB error updating inquiry record {inquiry.id[:8]}...: {update_err}", exc_info=True)
                    response_message = "I'm sorry, there was a database error processing your inquiry. Please try again later."
                    clear_session_state(session_id)
                    interaction_type = "inquiry_db_error"

            return response_message, interaction_type
        
    except PyMongoError as db_err:
        logger.error(f"Database error in inquiry flow: {db_err}", exc_info=True)
        clear_session_state(session_id)
        return "I'm experiencing technical difficulties. Please try again later.", "inquiry_db_error"
        
    except Exception as e:
        logger.error(f"Unexpected error in inquiry flow: {e}", exc_info=True)
        clear_session_state(session_id)
        return "Something went wrong. Please try again.", "inquiry_error"

async def _save_chat_interaction(
    db: Database,
    session_id: str,
    user_question: str,
    bot_answer: str,
    retrieved_contexts: Optional[List[Dict]] = None,
    llm_params: Optional[Dict] = None,
    interaction_type: str = "rag_chat"
) -> str:
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
        "interaction_type": interaction_type
    }
    
    try:
        await asyncio.to_thread(mongo_collection.insert_one, chat_log_mongo)
        logger.info(f"Chat interaction {interaction_id[:8]}... saved to MongoDB.")
    except Exception as e:
        logger.error(f"Failed to save chat interaction {interaction_id[:8]}... to MongoDB: {e}", exc_info=True)
    
    try:
        text_to_embed = f"User: {user_question}\nBot: {bot_answer}"
        
        vector = chatbot_instance.encoder.encode(text_to_embed).tolist()
        
        qdrant_payload = {
            "mongo_id": interaction_id,
            "session_id": session_id,
            "user_question_preview": user_question[:200],
            "full_interaction_text": text_to_embed,
            "timestamp_iso": timestamp.isoformat(),
            "interaction_type": interaction_type
        }
        
        point = PointStruct(
            id=interaction_id,
            vector=vector,
            payload=qdrant_payload
        )
        
        chatbot_instance.qdrant.upsert(
            collection_name=settings.QDRANT_CHAT_HISTORY_COLLECTION,
            points=[point],
            wait=False
        )
        logger.info(f"Chat interaction {interaction_id[:8]}... embedding saved to Qdrant.")
    except Exception as e:
        logger.error(f"Failed to save chat interaction {interaction_id[:8]}... to Qdrant: {e}", exc_info=True)
    
    return interaction_id

@router.post("/ask", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks, db: Database = Depends(get_db)):
    try:
        logger.info(f"Received chat request for question: '{request.question[:50]}' (Session: {request.session_id[:8]}...)")
        session_id = request.session_id or str(uuid.uuid4())
        user_question = request.question.strip()
        
        inquiries_collection = db[settings.MONGODB_INQUIRIES_COLLECTION]
        try:
            await asyncio.to_thread(inquiries_collection.create_index, "session_id", unique=False) 
            await asyncio.to_thread(inquiries_collection.create_index, "status", unique=False)
            logger.debug(f"Ensured indexes for inquiry collection: {settings.MONGODB_INQUIRIES_COLLECTION}")
        except Exception as index_err:
            logger.warning(f"Error creating indexes for inquiry collection: {index_err}. This may occur if indexes already exist. Continuing.")
        
        retrieved_state = get_session_state(session_id)
        current_session_state: Dict[str, Any] = {}
        if retrieved_state is not None and isinstance(retrieved_state, dict):
            current_session_state = retrieved_state
        else:
            if retrieved_state is not None:
                logger.warning(f"Unexpected non-dict session state retrieved for {session_id[:8]}...: {type(retrieved_state)}. Initializing as empty and clearing corrupted state.")
            else:
                logger.debug(f"No existing session state found for {session_id[:8]}... Initializing as empty.")
            clear_session_state(session_id)
        
        logger.debug(f"DEBUG: current_session_state after explicit check: {current_session_state}")
        
        if current_session_state.get("inquiry_active"):
            logger.info(f"Processing inquiry flow for session {session_id[:8]}...")
            bot_response, interaction_type = await handle_enquiry_flow(
                session_id, user_question, current_session_state, db, background_tasks, request.history
            )
            interaction_id = await _save_chat_interaction(
                db, session_id, user_question, bot_response, retrieved_contexts=[], llm_params={}, interaction_type=interaction_type
            )
            return ChatResponse(answer=bot_response, session_id=session_id, interaction_id=interaction_id)
        
        else:
            if current_session_state.get("offered_inquiry_fallback"):
                user_response_lower = user_question.lower().strip()
                if any(word in user_response_lower for word in ['yes', 'sure', 'ok', 'okay', 'please', 'help']):
                    logger.info(f"User accepted inquiry fallback offer for session {session_id[:8]}...")
                    
                    new_inquiry_state = current_session_state.copy() 
                    new_inquiry_state.update({
                        "inquiry_active": True,
                        "original_question": user_question
                    })
                    del new_inquiry_state["offered_inquiry_fallback"]
                    set_session_state(session_id, new_inquiry_state)
                    
                    bot_response, interaction_type = await handle_enquiry_flow(
                        session_id, user_question, new_inquiry_state, db, background_tasks, request.history
                    )
                    interaction_id = await _save_chat_interaction(
                        db, session_id, user_question, bot_response, [], {}, interaction_type
                    )
                    return ChatResponse(answer=bot_response, session_id=session_id, interaction_id=interaction_id)
                else:
                    logger.info(f"User declined or ignored inquiry fallback. Processing new question: '{user_question[:50]}'")
                    del current_session_state["offered_inquiry_fallback"]
                    set_session_state(session_id, current_session_state)
            
            if current_session_state.get("inquiry_done_in_session"):
                logger.info(f"Session {session_id[:8]}... previously completed inquiry, proceeding with RAG.")
            
            loop = asyncio.get_event_loop()
            
            try:
                query_result = await loop.run_in_executor(
                    None, 
                    partial(
                        chatbot_instance.query,
                        question=user_question,
                        session_id=session_id,
                        history=request.history,
                        top_k=request.top_k,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens
                    )
                )
            except Exception as query_err:
                logger.error(f"Error in RAG query execution for session {session_id[:8]}...: {query_err}", exc_info=True)
                query_result = {"answer": "I'm sorry, I encountered an error processing your question. Please try again.", "contexts": [], "metadata": {}} 
            
            bot_response = query_result.get("answer", "I'm sorry, I couldn't generate a response.")
            retrieved_contexts = query_result.get("contexts", [])
            llm_params = query_result.get("metadata", {})
            
            if (not retrieved_contexts and                 
                not current_session_state.get("offered_inquiry_fallback") and 
                not current_session_state.get("inquiry_done_in_session")):
                
                is_inquiry_intent = await route_intent_for_enquiry(user_question)
                
                if is_inquiry_intent:
                    logger.info(f"RAG failed but user intent is inquiry. Starting inquiry flow for session {session_id[:8]}...")
                    
                    new_inquiry_state = current_session_state.copy()
                    new_inquiry_state.update({
                        "inquiry_active": True,
                        "original_question": user_question,
                        "inquiry_status": "initiated"
                    })
                    set_session_state(session_id, new_inquiry_state)
                    
                    bot_response, interaction_type = await handle_enquiry_flow(
                        session_id, user_question, new_inquiry_state, db, background_tasks, request.history
                    )
                    interaction_id = await _save_chat_interaction(
                        db, session_id, user_question, bot_response, [], {}, interaction_type
                    )
                    return ChatResponse(answer=bot_response, session_id=session_id, interaction_id=interaction_id)
                else:
                    logger.info(f"No relevant contexts found, offering inquiry fallback for session {session_id[:8]}...")
                    bot_response = "I wasn't able to find specific information to answer your question. Would you like to submit an inquiry so someone can help you personally? Just say 'yes' if you'd like to proceed."
                    
                    current_session_state["offered_inquiry_fallback"] = True
                    set_session_state(session_id, current_session_state)
                    interaction_type = "rag_fallback_offer"
            else:
                interaction_type = "rag_chat"
            
            interaction_id = await _save_chat_interaction(
                db, session_id, user_question, bot_response, retrieved_contexts, llm_params, interaction_type
            )
            
            return ChatResponse(
                answer=bot_response,
                session_id=session_id,
                interaction_id=interaction_id
            )
            
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        if isinstance(e, groq.AuthenticationError):
            detail = "I apologize, but I'm currently experiencing an authentication error with the AI service. Please check your API key."
        elif isinstance(e, groq.RateLimitError):
            detail = "The AI service is temporarily unavailable due to high demand. Please try again in a moment."
        elif isinstance(e, groq.APIConnectionError):
            detail = "I'm having trouble connecting to the AI service. Please check your internet connection and try again."
        elif isinstance(e, groq.APIStatusError):
            detail = f"The AI service returned an unexpected status. Details: {e.status_code}."
        elif isinstance(e, PyMongoError):
            detail = "The database service is temporarily unavailable. Please try again later."
        else:
            detail = "An internal server error occurred. Please try again or contact support if the issue persists."
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail
        )