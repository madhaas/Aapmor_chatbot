#!/usr/bin/env python3
from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List
import os

from app.database import SessionLocal
from app.models import Document
from app.utils import process_document, delete_document
from app.chatbot import RAGChatbot

app = FastAPI()

# Serve static assets from ./static
app.mount("/static", StaticFiles(directory="static"), name="static")
# Load Jinja2 templates from ./templates
templates = Jinja2Templates(directory="templates")

# Initialize RAG chatbot
bot = RAGChatbot(
    qdrant_host="localhost",
    qdrant_port=6333,
    model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    n_threads=8
)

# Pydantic model for chat requests
class ChatRequest(BaseModel):
    question: str
    top_k: int = 3

@app.post("/api/documents/")
async def upload_document(file: UploadFile = File(...)):
    """Upload a file and ingest it into Qdrant + database."""
    os.makedirs("data/uploads", exist_ok=True)
    file_path = os.path.join("data/uploads", file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    doc = process_document(file_path)
    return {"message": f"Document '{file.filename}' processed", "id": doc.id}

@app.delete("/api/documents/{doc_id}")
async def delete_doc(doc_id: int):
    """Delete a document and its vectors by ID."""
    try:
        delete_document(doc_id)
        return {"message": "Document deleted"}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """Handle chat queries using the RAG pipeline."""
    answer = bot.query(request.question, request.top_k)
    return {"answer": answer}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the homepage with a list of ingested documents."""
    db = SessionLocal()
    documents: List[Document] = db.query(Document).all()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "documents": documents
    })
