#!/usr/bin/env python3
"""
FastAPI server exposing endpoints for document ingestion and JSON-based question querying,
leveraging DocumentProcessor and RAGChatbot from app.processor and app.chatbot.
Allows PDFs of any size (no file-size limit). Logs responses to console.
"""
import os
import tempfile
from typing import List, Dict, Any
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from app.processor import DocumentProcessor
from app.chatbot import RAGChatbot

# Configuration via environment or defaults
QDRANT_HOST       = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT       = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")
MONGODB_URI       = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB        = os.getenv("MONGODB_DB", "rag_system")
# LLM model path and settings
MODEL_PATH        = os.getenv("MODEL_PATH", "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf")
THREADS           = int(os.getenv("THREADS", 8))
GPU_LAYERS        = int(os.getenv("GPU_LAYERS", 35))

# Create FastAPI app and configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
api_logger = logging.getLogger("api_server")
app = FastAPI(debug=True)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate processor and chatbot once
try:
    processor = DocumentProcessor(
        qdrant_host=QDRANT_HOST,
        qdrant_port=QDRANT_PORT,
        mongodb_uri=MONGODB_URI,
        mongodb_db=MONGODB_DB,
        collection=QDRANT_COLLECTION
    )
    chatbot = RAGChatbot(
        qdrant_host=QDRANT_HOST,
        qdrant_port=QDRANT_PORT,
        collection=QDRANT_COLLECTION,
        model_path=MODEL_PATH,
        n_threads=THREADS,
        gpu_layers=GPU_LAYERS
    )
    api_logger.info("Initialized DocumentProcessor and RAGChatbot")
except Exception as e:
    api_logger.critical("Failed to initialize backend components: %s", e)
    raise

@app.on_event("startup")
def startup_event():
    """Ensure Qdrant collection exists before accepting requests"""
    if not processor.ensure_collection_exists(recreate=False):
        api_logger.error("Failed to initialize Qdrant collection '%s'", QDRANT_COLLECTION)
        raise RuntimeError(f"Could not initialize collection {QDRANT_COLLECTION}")
    api_logger.info("Qdrant collection '%s' ready", QDRANT_COLLECTION)

@app.post("/ingest")
async def ingest_documents(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    """
    Upload and ingest multiple documents (PDF/DOCX/TXT) in one request.
    Returns detailed processing summary or error and logs it.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            for f in files:
                contents = await f.read()
                out_path = os.path.join(tmpdir, f.filename)
                with open(out_path, "wb") as out_file:
                    out_file.write(contents)

            result = processor.process_document_folder(
                folder=tmpdir,
                recreate=False
            )

        # Convert datetime fields to ISO strings
        for key in ("start_time", "end_time"):
            if key in result and isinstance(result[key], datetime):
                result[key] = result[key].isoformat()

        # Log to console
        api_logger.info("/ingest result: %s", result)
        print("[ingest] Response:\n", result)

        status_code = 201 if result.get("success") else 500
        return JSONResponse(status_code=status_code, content=result)

    except HTTPException:
        raise
    except Exception as e:
        api_logger.exception("Unexpected error during ingest")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_endpoint(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accepts JSON `{'question': str, 'top_k': int, 'max_tokens': int, 'temperature': float}`
    Returns: { 'answer': str, 'context': [...], 'question': str } and logs it.
    """
    question = payload.get("question", "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="'question' field is required")

    top_k = int(payload.get("top_k", 3))
    max_tokens = int(payload.get("max_tokens", 512))
    temperature = float(payload.get("temperature", 0.3))

    # Retrieve context
    context = chatbot.retrieve_context(question, top_k)
    if not context:
        answer = "No relevant information found."
    else:
        answer = chatbot.generate_response(question, context, max_tokens, temperature)

    response = {
        "question": question,
        "context": context,
        "answer": answer
    }   

    # Log to console
    api_logger.info("/query question='%s' response=%s", question, response)
    print("[query] Response:\n", response)

    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug"
    )
