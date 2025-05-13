from fastapi import APIRouter, UploadFile, File, HTTPException
from app.database import get_db
from app.processor import process_document
from datetime import datetime
import uuid
import os

router = APIRouter()

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    db = get_db()
    
    try:
        # Save to MongoDB first
        doc_id = str(uuid.uuid4())
        doc_data = {
            "_id": doc_id,
            "filename": file.filename,
            "content_type": file.content_type,
            "upload_date": datetime.utcnow(),
            "status": "processing"
        }
        
        # Save metadata to MongoDB
        db.documents.insert_one(doc_data)
        
        # Save file temporarily
        file_path = f"temp/{doc_id}_{file.filename}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Process document in background
        process_document(file_path, doc_id)
        
        return {"id": doc_id, "status": "processing"}
    
    except Exception as e:
        db.documents.update_one(
            {"_id": doc_id},
            {"$set": {"status": "failed", "error": str(e)}}
        )
        raise HTTPException(500, str(e))