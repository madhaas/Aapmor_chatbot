#!/usr/bin/env python3
"""
Production-Ready RAG API Server with Full Validation
"""
from dotenv import load_dotenv
load_dotenv()      # reads your .env
import os
import json
import tempfile
from datetime import datetime
from typing import List, Annotated, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, UploadFile, File, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import asyncio
from functools import partial
import time
from app.processor import DocumentProcessor
from app.chatbot import RAGChatbot



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("api_[[2025_05_20]]rver")

# Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")
MODEL_PATH = os.getenv("MODEL_PATH", "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf")
THREADS = max(1, int(os.getenv("THREADS", 8))) #need to optmizise as much as possible here

app = FastAPI(
    title="RAG API Server",
    version="1.0.1",    
    docs_url="/docs",
    redoc_url=None
)

# Fixed CORS configuration: Either specify origins or disable credentials
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],  # Replace with your actual domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500, 
                        example="What does this document explain?")
    top_k: Annotated[int, Field(ge=1, le=10)] = Field(
        default=5,
        description="Number of context chunks to retrieve (1-10)"
    )
    temperature: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.3,
        description="LLM creativity control (0.0-1.0)"
    )

class QueryResponse(BaseModel):
    question: str
    answer: str
    contexts: List[dict]
    metadata: dict

class IngestResponse(BaseModel):
    success: bool
    documents_loaded: int
    chunks_created: int
    vectors_uploaded: int
    errors: List[str]
    duration: float
    start_time: Optional[str] = Field(None, example="2025-05-16T10:49:30.123456")
    end_time: Optional[str] = Field(None, example="2025-05-16T10:49:41.987654")

# Initialize components
try:
    processor = DocumentProcessor(
        qdrant_host=QDRANT_HOST,
        qdrant_port=QDRANT_PORT,
        collection=QDRANT_COLLECTION
    )
    
    chatbot = RAGChatbot(
        qdrant_host=QDRANT_HOST,
        qdrant_port=QDRANT_PORT,
        collection=QDRANT_COLLECTION,
        model_path=MODEL_PATH,
        n_threads=THREADS
    )
    logger.info("Backend components initialized")
except Exception as e:
    logger.critical("Initialization failed: %s", e)
    raise

@app.on_event("startup")
async def verify_collection():
    """Validate Qdrant collection exists"""
    try:
        # Run blocking operation in a threadpool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, 
                                           lambda: processor.ensure_collection_exists(recreate=False))
        if not result:
            logger.error("Qdrant collection verification failed")
            raise RuntimeError("Qdrant collection unavailable")
        logger.info("Qdrant collection '%s' ready", QDRANT_COLLECTION)
    except Exception as e:
        logger.critical("Collection verification error: %s", e)
        raise

def _sanitize_dates(data: dict) -> dict:
    """Recursively convert datetime objects to ISO format strings"""
    def date_converter(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return obj

    return json.loads(json.dumps(data, default=date_converter))

async def _process_documents_async(dir_path: str):
    """Run document processing in a thread pool"""
    loop = asyncio.get_event_loop()
    
    # Execute blocking operation in the default thread pool
    result = await loop.run_in_executor(
        None, 
        processor.process_document_folder,
        dir_path
    )
    
    # Ensure the result has all expected fields for IngestResponse
    result = _sanitize_dates(result)
    
    # Add any missing fields to match the response model
    result.setdefault("success", False)
    result.setdefault("documents_loaded", 0)
    result.setdefault("chunks_created", 0)
    result.setdefault("vectors_uploaded", 0)
    result.setdefault("errors", [])
    result.setdefault("start_time", None)
    result.setdefault("end_time", None)
    
    return result

@app.post("/ingest", response_model=IngestResponse, tags=["Data Management"])
async def ingest_documents(files: List[UploadFile] = File(...)):
    """Process and store documents in vector database"""
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Minimum one file required"
        )

    try:
        with tempfile.TemporaryDirectory() as tmpdir:  # Proper tempdir handling
            # Save files
            filenames = []
            for file in files:
                if not file.filename:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Files must have valid names"
                    )
                file_path = os.path.join(tmpdir, file.filename)
                with open(file_path, "wb") as f:
                    f.write(await file.read())
                filenames.append(file_path)

            # Process documents asynchronously 
            start = datetime.now()
            result = await _process_documents_async(tmpdir)
            
            # Calculate and add duration
            result["duration"] = (datetime.now() - start).total_seconds()

        # Return a properly validated response that matches the Pydantic model
        response = IngestResponse(**result)
        
        return JSONResponse(
            status_code=status.HTTP_201_CREATED if response.success else status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=response.dict()
        )

    except Exception as e:
        logger.error("Ingestion error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {str(e)}"
        )

async def _execute_query_async(question: str, top_k: int, temperature: float):
    """Run query in a thread pool"""
    loop = asyncio.get_event_loop()
    
    # Execute blocking operation in the default thread pool
    return await loop.run_in_executor(
        None,
        partial(chatbot.query, question=question, top_k=top_k, temperature=temperature)
    )

@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def handle_query(request: QueryRequest):
    """Handle natural language queries"""
    try:
        logger.info("New query: '%s'", request.question[:50])
        
        # Run blocking chatbot query in a thread pool
        result = await _execute_query_async(
            question=request.question,
            top_k=request.top_k,
            temperature=request.temperature
        )

        # Format response to match Pydantic model
        response = {
            "question": result["metadata"]["question"],
            "answer": result["answer"],
            "contexts": result["contexts"],
            "metadata": {
                "top_k": result["metadata"]["top_k"],
                "temperature": result["metadata"]["temperature"]
            }
        }

        return response

    except Exception as e:
        logger.error("Query failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Question processing error"
        )

@app.exception_handler(HTTPException)
async def custom_exception_handler(request, exc):
    """Standard error response format"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "success": False
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )