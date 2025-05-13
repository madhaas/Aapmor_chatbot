from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.chatbot import RAGChatbot

router = APIRouter()

class ChatRequest(BaseModel):
    question: str
    top_k: int = 3
    temperature: float = 0.3
    max_tokens: int = 512

@router.post("/ask")
def chat_endpoint(request: ChatRequest):
    try:
        bot = RAGChatbot()
        answer = bot.query(
            request.question,
            top_k=request.top_k,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(500, str(e))