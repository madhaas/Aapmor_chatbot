from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, logger
from app.database import get_db
from app.processor import process_document
from datetime import datetime
import uuid
import os

router = APIRouter()

@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    db = get_db()
    
    try:
        # Generate document ID and metadata
        doc_id = str(uuid.uuid4())
        doc_data = {
            "_id": doc_id,
            "filename": file.filename,
            "content_type": file.content_type,
            "upload_date": datetime.utcnow(),
            "status": "processing"
        }

        # Save initial metadata to MongoDB
        db.documents.insert_one(doc_data)
        
        # Create temp directory if needed
        file_path = f"temp/{doc_id}_{file.filename}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Async file write
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Add background processing task
        background_tasks.add_task(
            process_document_with_error_handling,
            file_path,
            doc_id,
            db
        )

        return {"id": doc_id, "status": "processing"}
    
    except Exception as e:
        # Cleanup partially uploaded file if it exists
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
            
        raise HTTPException(500, f"Upload failed: {str(e)}")

def process_document_with_error_handling(file_path: str, doc_id: str, db):
    """Wrapper function to handle processing errors"""
    try:
        process_document(file_path, doc_id)
        db.documents.update_one(
            {"_id": doc_id},
            {"$set": {"status": "completed"}}
        )
    except Exception as e:
        db.documents.update_one(
            {"_id": doc_id},
            {"$set": {
                "status": "failed",
                "error": str(e)
            }}
        )
        logger.error(f"Processing failed for {doc_id}: {str(e)}")
    finally:
        # Cleanup temporary file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError as e:
                logger.error(f"Failed to cleanup {file_path}: {str(e)}")