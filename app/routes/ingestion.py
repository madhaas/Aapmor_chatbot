from fastapi import APIRouter, UploadFile, File, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import json
import tempfile
from datetime import datetime, timezone
import asyncio
from functools import partial
import logging
import shutil
from app.api_server import processor_instance  

logger = logging.getLogger("ingestion_router")

router = APIRouter()

class IngestResponse(BaseModel):
    success: bool
    documents_loaded: int
    chunks_created: int
    vectors_uploaded: int
    errors: List[str]
    duration: float
    start_time: Optional[str] = Field(None, example="2025-05-16T10:49:30.123456")
    end_time: Optional[str] = Field(None, example="2025-05-16T10:49:41.987654")

class IngestAcceptedResponse(BaseModel):
    status: str
    message: str

def _sanitize_dates(data: dict) -> dict:
    def date_converter(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return obj


    return json.loads(json.dumps(data, default=date_converter))

async def _process_documents_async(dir_path: str):
    loop = asyncio.get_event_loop()
    

    result = await loop.run_in_executor(
        None, 
        partial(processor_instance.process_document_folder, recreate=False),  
        dir_path
    )
    
    
    result = _sanitize_dates(result)
    result.setdefault("success", False)
    result.setdefault("documents_loaded", 0)
    result.setdefault("chunks_created", 0)
    result.setdefault("vectors_uploaded", 0)
    result.setdefault("errors", [])
    result.setdefault("start_time", None)
    result.setdefault("end_time", None)
    
    return result

@router.post("/", response_model=IngestAcceptedResponse, status_code=status.HTTP_202_ACCEPTED, tags=["Data Management"])
async def ingest_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="List of files to ingest. At least one file is required.")
):
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one file required for ingestion."
        )

    try:
        temp_dir_suffix = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        tmpdir = tempfile.mkdtemp(prefix=f"rag_ingest_{temp_dir_suffix}_")
        
        logger.info(f"Created temporary directory for ingestion: {tmpdir}")

        # Saving files to the temporary directory
        for file in files:
            if not file.filename:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Files must have valid names."
                )
            file_path = os.path.join(tmpdir, file.filename)
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            logger.info(f"Saved uploaded file: {file.filename} to {file_path}")


        background_tasks.add_task(_ingestion_background_wrapper, tmpdir)

        logger.info(f"Accepted {len(files)} files for background ingestion.")
        return IngestAcceptedResponse(
            status="processing",
            message=f"Ingestion of {len(files)} files initiated in the background. Check logs for details."
        )

    except Exception as e:
        logger.error(f"Error accepting ingestion request: {e}", exc_info=True)
        if 'tmpdir' in locals() and os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)
            logger.info(f"Cleaned up temp directory {tmpdir} due to error.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initiate ingestion: {str(e)}"
        )

async def _ingestion_background_wrapper(tmpdir: str):
    #wrapper to handle bg
    start_time_wrapper = datetime.now()
    try:
        logger.info(f"Background ingestion task started for directory: {tmpdir}")
        ingestion_stats = await _process_documents_async(tmpdir)
        
        if ingestion_stats["success"]:
            logger.info(f"Background ingestion for {tmpdir} completed successfully. Stats: {ingestion_stats}")
        else:
            logger.error(f"Background ingestion for {tmpdir} failed. Errors: {ingestion_stats['errors']}. Stats: {ingestion_stats}")
            
    except Exception as e:
        logger.critical(f"Unhandled critical error in background ingestion task for {tmpdir}: {e}", exc_info=True)
    finally:
        # Clean up the temporary directory regardless of success or failure
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)
            logger.info(f"Cleaned up temporary directory: {tmpdir}")
