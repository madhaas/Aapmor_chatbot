from fastapi import APIRouter, UploadFile, File, HTTPException, status, BackgroundTasks
from datetime import datetime, timezone
import uuid
import os
import logging
import asyncio
from functools import partial
import shutil
from typing import List, Optional, Dict, Any
from app.api_server import processor_instance
from app_config import settings

logger = logging.getLogger("documents_router")
router = APIRouter()

async def process_document_with_error_handling(temp_dir: str, doc_id: str):
    try:
        logger.info(f"Background processing started for document {doc_id} in {temp_dir}")
        processing_result = await asyncio.get_event_loop().run_in_executor(
            None, 
            partial(processor_instance.process_document_folder, folder=temp_dir, recreate=False)
        )
        
        if processing_result["success"]:
            logger.info(f"Successfully processed document {doc_id}.")
        else:
            raise Exception(f"Processing failed for {doc_id}: {processing_result.get('errors', ['Unknown processing error'])}")
            
    except Exception as e:
        logger.error(f"Processing failed for document {doc_id}: {str(e)}", exc_info=True)
        processor_instance.docs_collection.update_one(
            {"_id": doc_id},
            {"$set": {
                "status": "failed",
                "error": str(e),
                "updated_at": datetime.now(timezone.utc)
            }}
        )
    finally:
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except OSError as clean_e:
            logger.error(f"Failed to cleanup temp directory {temp_dir}: {clean_e}", exc_info=True)

@router.post("/upload", response_model=Dict[str, Any], status_code=status.HTTP_202_ACCEPTED, tags=["Document Management"])
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Document file to upload (PDF, DOCX, TXT).")
):
    
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File must have a valid name.")
    
    doc_id = str(uuid.uuid4())
    temp_dir = os.path.join("temp_uploads", doc_id)
    file_path = os.path.join(temp_dir, file.filename)
    
    try:
        os.makedirs(temp_dir, exist_ok=True)
        
        processor_instance.docs_collection.insert_one({
            "_id": doc_id,
            "filename": file.filename,
            "content_type": file.content_type,
            "upload_date": datetime.now(timezone.utc),
            "status": "uploading"
        })
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"File '{file.filename}' saved temporarily to: {file_path}")
        
        background_tasks.add_task(process_document_with_error_handling, temp_dir, doc_id)
        
        return {
            "id": doc_id,
            "status": "processing",
            "message": f"Document '{file.filename}' accepted for background processing. Check logs for details."
        }
        
    except Exception as e:
        logger.error(f"Upload endpoint failed for '{file.filename}': {str(e)}", exc_info=True)
        
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temp directory {temp_dir} due to upload error.")
        except OSError as clean_e:
            logger.error(f"Failed to cleanup temp directory {temp_dir}: {clean_e}", exc_info=True)
        
        processor_instance.docs_collection.update_one(
            {"_id": doc_id},
            {"$set": {
                "status": "upload_failed",
                "error": str(e),
                "updated_at": datetime.now(timezone.utc)
            }}
        )
        
        detail_message = f"Upload failed: {str(e)}" if settings.DEBUG else "Upload failed due to an internal server error."
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail_message)