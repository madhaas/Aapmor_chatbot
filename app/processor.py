import argparse
import os
import uuid
import json
import traceback
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import torch
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType, FieldCondition
from qdrant_client.http import models as rest
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from app_config import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("document_processor.log")
    ]
)
logger = logging.getLogger("document_processor")


class DocumentProcessor:
    def __init__(
        self,
        qdrant_host: str = settings.QDRANT_HOST,
        qdrant_port: int = settings.QDRANT_PORT,
        mongodb_uri: str = settings.MONGODB_URL,
        mongodb_db: str = settings.DB_NAME,
        collection: str = settings.QDRANT_COLLECTION,
        chunk_size: int = int(os.getenv("CHUNK_SIZE", 1000)),
        chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 200)),
        embedding_model: str = settings.EMBEDDING_MODEL,
        batch_size: int = int(os.getenv("BATCH_SIZE", 100))
    ):
        self.collection = collection
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        
        logger.info(f"Initializing embedding model: {embedding_model}")
        self.model = SentenceTransformer(embedding_model, device="cuda" if torch.cuda.is_available() else "cpu")
        self.dim = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Connecting to Qdrant at {qdrant_host}:{qdrant_port}")
        self.qdrant = QdrantClient(url=f"http://{qdrant_host}:{qdrant_port}")
        
        logger.info(f"Connecting to MongoDB at {mongodb_uri}")
        try:
            self.mongo_client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
            self.mongo_client.admin.command('ping')
            self.db = self.mongo_client[mongodb_db]
            self.docs_collection = self.db["documents"]
            self.chunks_collection = self.db["chunks"]
            self.processing_collection = self.db["processing_logs"]
            logger.info("Successfully connected to MongoDB in DocumentProcessor.")
        except Exception as e:
            logger.critical(f"Error connecting to MongoDB at {mongodb_uri}: {e}", exc_info=True)
            raise

    def load_documents(self, folder: str) -> List[Document]:
        docs: List[Document] = []
        supported_extensions = ['.pdf', '.docx', '.txt']
        
        process_id = str(uuid.uuid4())
        self.processing_collection.insert_one({
            "process_id": process_id,
            "start_time": datetime.now(timezone.utc),
            "status": "loading",
            "folder": folder,
            "files_total": 0,
            "files_processed": 0,
            "files_failed": 0
        })
        
        total_files = sum(1 for _, _, files in os.walk(folder) 
                         for f in files if Path(f).suffix.lower() in supported_extensions)
        
        self.processing_collection.update_one(
            {"process_id": process_id},
            {"$set": {"files_total": total_files}}
        )
        
        processed = 0
        failed = 0
        
        for root, _, files in os.walk(folder):
            for fname in files:
                path = Path(root) / fname
                ext = path.suffix.lower()
                
                if ext not in supported_extensions:
                    logger.info(f"Skipping unsupported file: {fname} (extension: {ext})")
                    continue

                doc_id = str(uuid.uuid4())
                
                try:
                    self.docs_collection.insert_one({
                        "_id": doc_id,
                        "filename": fname,
                        "filepath": str(path),
                        "status": "loading",
                        "created_at": datetime.now(timezone.utc),
                        "size": path.stat().st_size,
                        "extension": ext
                    })
                    
                    loader = None
                    if ext == '.pdf':
                        loader = PyMuPDFLoader(str(path))
                    elif ext == '.docx':
                        loader = Docx2txtLoader(str(path))
                    elif ext == '.txt':
                        loader = TextLoader(str(path))

                    if loader:
                        documents = loader.load()
                        
                        for doc in documents:
                            doc.metadata["doc_id"] = doc_id
                            doc.metadata["filename"] = fname
                            doc.metadata["source"] = str(path)
                            
                        logger.info(f"Loaded {len(documents)} pages from {fname}")
                        docs.extend(documents)
                        processed += 1
                        
                        self.docs_collection.update_one(
                            {"_id": doc_id},
                            {"$set": {
                                "status": "loaded",
                                "pages": len(documents),
                                "updated_at": datetime.now(timezone.utc)
                            }}
                        )
                        
                except Exception as e:
                    error_msg = f"Failed to load {fname}: {str(e)[:200]}..."
                    logger.error(error_msg, exc_info=True)
                    failed += 1
                    
                    self.docs_collection.update_one(
                        {"_id": doc_id},
                        {"$set": {
                            "status": "failed",
                            "error": error_msg,
                            "updated_at": datetime.now(timezone.utc)
                        }}
                    )
                
                self.processing_collection.update_one(
                    {"process_id": process_id},
                    {"$set": {
                        "files_processed": processed,
                        "files_failed": failed
                    }}
                )

        self.processing_collection.update_one(
            {"process_id": process_id},
            {"$set": {
                "status": "loading_complete" if docs else "loading_failed",
                "end_time": datetime.now(timezone.utc)
            }}
        )
            
        return docs

    def chunk_documents(
        self,
        docs: List[Document],
        min_chunk_size: int = 200,
        min_chunk_overlap: int = 50
    ) -> List[Document]:
        process_id = str(uuid.uuid4())
        self.processing_collection.insert_one({
            "process_id": process_id,
            "start_time": datetime.now(timezone.utc),
            "status": "chunking",
            "documents": len(docs),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        })
        
        current_size = self.chunk_size
        current_overlap = self.chunk_overlap
        
        while True:
            try:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=current_size,
                    chunk_overlap=current_overlap,
                    length_function=len,
                    add_start_index=True
                )
                chunks = splitter.split_documents(docs)
                
                if chunks and all(len(c.page_content) > 50 for c in chunks):
                    logger.info(f"Chunking successful: {len(chunks)} chunks using size {current_size}, overlap {current_overlap}")
                    logger.info(f"Sample chunk ({len(chunks[0].page_content)} chars): {chunks[0].page_content[:100]}...")
                    
                    chunk_info = []
                    for i, chunk in enumerate(chunks):
                        chunk_id = str(uuid.uuid4())
                        chunk.metadata["chunk_id"] = chunk_id
                        
                        chunk_info.append({
                            "_id": chunk_id,
                            "doc_id": chunk.metadata.get("doc_id", "unknown"),
                            "filename": chunk.metadata.get("filename", "unknown"),
                            "page": chunk.metadata.get("page", 0),
                            "start_index": chunk.metadata.get("start_index", 0),
                            "length": len(chunk.page_content),
                            "content": chunk.page_content,
                            "created_at": datetime.now(timezone.utc)
                        })
                    
                    if chunk_info:
                        try:
                            self.chunks_collection.insert_many(chunk_info)
                        except PyMongoError as pe:
                            logger.error(f"MongoDB error inserting chunks: {pe}", exc_info=True)
                            raise
                    
                    self.processing_collection.update_one(
                        {"process_id": process_id},
                        {"$set": {
                            "status": "chunking_complete",
                            "chunks_created": len(chunks),
                            "end_time": datetime.now(timezone.utc),
                            "final_chunk_size": current_size,
                            "final_chunk_overlap": current_overlap
                        }}
                    )
                    
                    return chunks
                
                current_size = max(current_size // 2, min_chunk_size)
                current_overlap = max(current_overlap // 2, min_chunk_overlap)
                logger.info(f"Chunking adjustment: Reducing to chunk_size={current_size}, overlap={current_overlap} as initial chunks were too small or absent.")
                
                if current_size <= min_chunk_size and current_overlap <= min_chunk_overlap:
                    raise ValueError(f"Failed to produce valid chunks after multiple adjustments. Minimum chunk size: {min_chunk_size}, minimum overlap: {min_chunk_overlap}.")

            except Exception as e:
                error_msg = f"Critical chunking error: {str(e)}"
                logger.error(error_msg, exc_info=True)
                
                self.processing_collection.update_one(
                    {"process_id": process_id},
                    {"$set": {
                        "status": "chunking_failed",
                        "error": error_msg,
                        "end_time": datetime.now(timezone.utc)
                    }}
                )
                
                return []

    def ensure_collection_exists(self, recreate: bool = False) -> bool:
        self.qdrant = QdrantClient(url=f"http://{self.qdrant.host}:{self.qdrant.port}")

        if recreate:
            logger.info(f"Attempting to delete collection '{self.collection}' (recreate requested)...")
            try:
                self.qdrant.delete_collection(self.collection)
                logger.info(f"Collection '{self.collection}' deleted successfully.")
            except Exception as e:
                logger.warning(f"Collection deletion skipped or failed for '{self.collection}': {str(e)[:100]}...", exc_info=True)

        try:
            self.qdrant.get_collection(self.collection)
            logger.info(f"Collection '{self.collection}' already exists and is ready.")
            return True
        except Exception:
            logger.info(f"Collection '{self.collection}' not found. Creating it with dimension {self.dim} and full schema...")
            try:
                self.qdrant.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(
                        size=self.dim,
                        distance=Distance.COSINE
                    ),
                    optimizers_config=rest.OptimizersConfigDiff(
                        indexing_threshold=20000,
                        memmap_threshold=100000
                    ),
                    payload_schema={
                        "text": PayloadSchemaType.TEXT,
                        "doc_id": PayloadSchemaType.KEYWORD,
                        "filename": PayloadSchemaType.KEYWORD,
                        "page": PayloadSchemaType.INTEGER,
                        "source": PayloadSchemaType.KEYWORD,
                        "chunk_id": PayloadSchemaType.KEYWORD
                    }
                )
                
                self.qdrant.create_payload_index(
                    collection_name=self.collection,
                    field_name="doc_id",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                self.qdrant.create_payload_index(
                    collection_name=self.collection,
                    field_name="filename",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                self.qdrant.create_payload_index(
                    collection_name=self.collection,
                    field_name="chunk_id",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                self.qdrant.create_payload_index(
                    collection_name=self.collection,
                    field_name="page",
                    field_schema=PayloadSchemaType.INTEGER
                )
                
                logger.info(f"Collection '{self.collection}' created successfully with indexes.")
                return True
            except Exception as e:
                logger.critical(f"Failed to create Qdrant collection '{self.collection}': {str(e)}", exc_info=True)
                return False

    def batch_upsert(self, points: List[PointStruct]) -> int:
        total = len(points)
        successful = 0
        
        process_id = str(uuid.uuid4())
        self.processing_collection.insert_one({
            "process_id": process_id,
            "start_time": datetime.now(timezone.utc),
            "status": "uploading",
            "total_points": total,
            "uploaded": 0
        })
        
        for i in range(0, total, self.batch_size):
            batch = points[i:i+self.batch_size]
            try:
                self.qdrant.upsert(
                    collection_name=self.collection,
                    points=batch,
                    wait=True
                )
                
                chunk_ids = [point.payload.get("chunk_id") for point in batch if point.payload.get("chunk_id")]
                
                if chunk_ids:
                    self.chunks_collection.update_many(
                        {"_id": {"$in": chunk_ids}},
                        {"$set": {"indexed": True, "indexed_at": datetime.now(timezone.utc)}}
                    )
                
                progress = min(i + self.batch_size, total)
                successful += len(batch)
                
                self.processing_collection.update_one(
                    {"process_id": process_id},
                    {"$set": {"uploaded": progress, "updated_at": datetime.now(timezone.utc)}}
                )
                
                logger.info(f"Uploaded {progress}/{total} ({progress/total:.1%})")
                
            except Exception as e:
                error_msg = f"Batch upload failed: {str(e)[:200]}..."
                logger.error(error_msg, exc_info=True)
                
                chunk_ids = [point.payload.get("chunk_id") for point in batch if point.payload.get("chunk_id")]
                if chunk_ids:
                    self.chunks_collection.update_many(
                        {"_id": {"$in": chunk_ids}},
                        {"$set": {"indexed": False, "error": error_msg, "updated_at": datetime.now(timezone.utc)}}
                    )

        self.processing_collection.update_one(
            {"process_id": process_id},
            {"$set": {
                "status": "upload_complete" if successful == total else "upload_partial",
                "uploaded": successful,
                "end_time": datetime.now(timezone.utc)
            }}
        )
        
        logger.info(f"Upload completed: {successful}/{total} points successful")
        return successful

    def prepare_vectors(self, chunks: List[Document]) -> List[PointStruct]:
        logger.info("Preparing embedding points...")
        points = []
        
        process_id = str(uuid.uuid4())
        self.processing_collection.insert_one({
            "process_id": process_id,
            "start_time": datetime.now(timezone.utc),
            "status": "embedding",
            "total_chunks": len(chunks),
            "processed": 0
        })
        
        for i, chunk in enumerate(chunks):
            try:
                vector = self.model.encode(chunk.page_content).tolist()
                
                payload = {
                    "text": chunk.page_content,
                    "doc_id": chunk.metadata.get("doc_id"),
                    "filename": chunk.metadata.get("filename"),
                    "page": chunk.metadata.get("page", 0),
                    "source": chunk.metadata.get("source"),
                    "chunk_id": chunk.metadata.get("chunk_id")
                }
                payload = {k: v for k, v in payload.items() if v is not None}

                if "page" in payload:
                    try:
                        payload["page"] = int(payload["page"])
                    except (ValueError, TypeError):
                        logger.warning(f"Could not convert page '{payload['page']}' to integer for chunk_id {payload.get('chunk_id')}. Storing as is.")
                        pass
                
                point_id = payload.get("chunk_id")
                if not point_id:
                    point_id = str(uuid.uuid4())
                    logger.warning(f"Chunk_id missing for chunk {i}, generated new ID: {point_id}")
                
                points.append(PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                ))
                
                if (i + 1) % 10 == 0:
                    self.processing_collection.update_one(
                        {"process_id": process_id},
                        {"$set": {"processed": i + 1, "updated_at": datetime.now(timezone.utc)}}
                    )
                    
            except Exception as e:
                error_msg = f"Chunk embedding failed for chunk {i}: {str(e)[:200]}..."
                logger.error(error_msg, exc_info=True)
                
                chunk_id = chunk.metadata.get("chunk_id")
                if chunk_id:
                    self.chunks_collection.update_one(
                        {"_id": chunk_id},
                        {"$set": {
                            "embedding_error": error_msg,
                            "embedding_failed": True,
                            "updated_at": datetime.now(timezone.utc)
                        }}
                    )
        
        self.processing_collection.update_one(
            {"process_id": process_id},
            {"$set": {
                "status": "embedding_complete",
                "processed": len(points),
                "end_time": datetime.now(timezone.utc)
            }}
        )
                
        return points
    
    def update_document_status(self, doc_id: str, status: str, error: Optional[str] = None) -> None:
        update = {"status": status, "updated_at": datetime.now(timezone.utc)}
        if error:
            update["error"] = error
            
        self.docs_collection.update_one(
            {"_id": doc_id},
            {"$set": update}
        )
    
    def process_document_folder(self, folder: str, recreate: bool = False) -> Dict[str, Any]:
        result = {
            "start_time": datetime.now(timezone.utc),
            "folder": folder,
            "success": False,
            "documents_loaded": 0,
            "chunks_created": 0,
            "vectors_uploaded": 0,
            "errors": []
        }
        
        try:
            logger.info(f"Loading documents from {folder}...")
            docs = self.load_documents(folder)
            if not docs:
                result["errors"].append("No loadable documents found or an error occurred during loading.")
                return result
                
            result["documents_loaded"] = len(docs)

            logger.info("Chunking documents...")
            chunks = self.chunk_documents(docs)
            if not chunks:
                result["errors"].append("Failed to generate valid chunks.")
                doc_ids_in_folder = set(doc.metadata.get("doc_id") for doc in docs if "doc_id" in doc.metadata)
                for doc_id in doc_ids_in_folder:
                    self.update_document_status(doc_id, "chunking_failed", "No valid chunks generated.")
                return result
                
            result["chunks_created"] = len(chunks)

            if not self.ensure_collection_exists(recreate):
                result["errors"].append("Failed to create or access vector collection.")
                doc_ids_in_folder = set(doc.metadata.get("doc_id") for doc in docs if "doc_id" in doc.metadata)
                for doc_id in doc_ids_in_folder:
                    self.update_document_status(doc_id, "collection_error", "Failed to ensure Qdrant collection.")
                return result

            logger.info("Preparing vectors (embedding chunks)...")
            vectors = self.prepare_vectors(chunks)
            if not vectors:
                result["errors"].append("Failed to generate vector embeddings for any chunks.")
                for chunk in chunks:
                    if not chunk.metadata.get("embedding_failed"):
                         self.chunks_collection.update_one({"_id": chunk.metadata.get("chunk_id")}, {"$set": {"embedding_failed": True, "error": "No vector generated"}})
                return result

            logger.info(f"Uploading {len(vectors)} vectors to Qdrant...")
            uploaded = self.batch_upsert(vectors)
            result["vectors_uploaded"] = uploaded
            
            result["success"] = (uploaded == len(vectors))
            
            doc_ids_in_folder = set(doc.metadata.get("doc_id") for doc in docs if "doc_id" in doc.metadata)
            for doc_id in doc_ids_in_folder:
                if result["success"]:
                    self.update_document_status(doc_id, "processed")
                else:
                    self.update_document_status(doc_id, "processing_partial_failure", "Not all vectors uploaded.")
                
        except Exception as e:
            error_msg = f"Overall document processing error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            result["errors"].append(error_msg)
            doc_ids_in_folder = set(doc.metadata.get("doc_id") for doc in docs if "doc_id" in doc.metadata)
            for doc_id in doc_ids_in_folder:
                self.update_document_status(doc_id, "processing_failed", error_msg)
            
        finally:
            result["end_time"] = datetime.now(timezone.utc)
            result["duration"] = (result["end_time"] - result["start_time"]).total_seconds()
            
            result["start_time"] = result["start_time"].isoformat()
            result["end_time"] = result["end_time"].isoformat()
            return result


def main():
    parser = argparse.ArgumentParser(description="Document Processor for RAG System")
    parser.add_argument("--input-folder", required=True, help="Path to documents folder")
    parser.add_argument("--qdrant-host", default=os.getenv("QDRANT_HOST", "localhost"), help="Qdrant host")
    parser.add_argument("--qdrant-port", type=int, default=int(os.getenv("QDRANT_PORT", 6333)), help="Qdrant port")
    parser.add_argument("--mongodb-uri", default=os.getenv("MONGODB_URI", "mongodb://localhost:27017"), help="MongoDB URI")
    parser.add_argument("--mongodb-db", default=os.getenv("MONGODB_DB", "rag_system"), help="MongoDB database name")
    parser.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", "documents"), help="Qdrant collection name")
    parser.add_argument("--chunk-size", type=int, default=int(os.getenv("CHUNK_SIZE", 1000)), help="Initial chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=int(os.getenv("CHUNK_OVERLAP", 200)), help="Initial chunk overlap")
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", 100)), help="Upsert batch size")
    parser.add_argument("--recreate", action="store_true", help="Recreate Qdrant collection")
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"), help="Embedding model name")
    parser.add_argument("--log-level", default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                        help="Set logging level")
    
    args = parser.parse_args()
    
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    processor = DocumentProcessor(
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        mongodb_uri=args.mongodb_uri,
        mongodb_db=args.mongodb_db,
        collection=args.collection,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.model,
        batch_size=args.batch_size
    )
    
    result = processor.process_document_folder(args.input_folder, args.recreate)
    
    print("\n--- Processing Summary ---")
    print(f"Documents loaded: {result['documents_loaded']}")
    print(f"Chunks created: {result['chunks_created']}")
    print(f"Vectors uploaded: {result['vectors_uploaded']}")
    print(f"Status: {'Success' if result['success'] else 'Failed'}")
    print(f"Duration: {result['duration']:.1f} seconds")
    
    if result["errors"]:
        print("\nErrors encountered:")
        for error in result["errors"]:
            print(f"- {error}")


if __name__ == "__main__":
    main()