"""
Enhanced document processor with Qdrant 1.7 compatibility and MongoDB integration
for RAG system according to project flow.
"""

import argparse
import os
import uuid
import json
import traceback
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType, FieldCondition
from qdrant_client.http import models as rest
from pymongo import MongoClient
from pymongo.errors import PyMongoError

# Configure logging
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
    """Document processor for RAG system with MongoDB tracking and Qdrant storage"""
    
    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        mongodb_uri: str = "mongodb://localhost:27017",
        mongodb_db: str = "rag_system",
        collection: str = "documents",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "all-MiniLM-L6-v2",
        batch_size: int = 100
    ):
        self.collection = collection
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        
        # Initialize embedding model
        logger.info(f"Initializing embedding model: {embedding_model}")
        self.model = SentenceTransformer(embedding_model)
        self.dim = self.model.get_sentence_embedding_dimension()
        
        # Initialize Qdrant client (version 1.7+ compatible)
        logger.info(f"Connecting to Qdrant at {qdrant_host}:{qdrant_port}")
        self.qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
        
        # Initialize MongoDB client
        logger.info(f"Connecting to MongoDB at {mongodb_uri}")
        self.mongo_client = MongoClient(mongodb_uri)
        self.db = self.mongo_client[mongodb_db]
        self.docs_collection = self.db["documents"]
        self.chunks_collection = self.db["chunks"]
        self.processing_collection = self.db["processing_logs"]

    def load_documents(self, folder: str) -> List[Document]:
        """Load documents with enhanced error recovery and status tracking"""
        docs: List[Document] = []
        supported_extensions = ['.pdf', '.docx', '.txt']
        
        # Create processing log entry
        process_id = str(uuid.uuid4())
        self.processing_collection.insert_one({
            "process_id": process_id,
            "start_time": datetime.now(),
            "status": "loading",
            "folder": folder,
            "files_total": 0,
            "files_processed": 0,
            "files_failed": 0
        })
        
        # Count total files first
        total_files = sum(1 for _, _, files in os.walk(folder) 
                         for f in files if Path(f).suffix.lower() in supported_extensions)
        
        self.processing_collection.update_one(
            {"process_id": process_id},
            {"$set": {"files_total": total_files}}
        )
        
        # Process files
        processed = 0
        failed = 0
        
        for root, _, files in os.walk(folder):
            for fname in files:
                path = Path(root) / fname
                ext = path.suffix.lower()
                
                if ext not in supported_extensions:
                    continue

                # Create document metadata record
                doc_id = str(uuid.uuid4())
                
                try:
                    # Track document in MongoDB first
                    self.docs_collection.insert_one({
                        "doc_id": doc_id,
                        "filename": fname,
                        "filepath": str(path),
                        "status": "loading",
                        "created_at": datetime.now(),
                        "size": path.stat().st_size,
                        "extension": ext
                    })
                    
                    # Load the document
                    loader = None
                    if ext == '.pdf':
                        loader = PyMuPDFLoader(str(path))
                    elif ext == '.docx':
                        loader = Docx2txtLoader(str(path))
                    elif ext == '.txt':
                        loader = TextLoader(str(path))

                    if loader:
                        documents = loader.load()
                        
                        # Enhance documents with our doc_id
                        for doc in documents:
                            doc.metadata["doc_id"] = doc_id
                            doc.metadata["filename"] = fname
                            
                        logger.info(f"Loaded {len(documents)} pages from {fname}")
                        docs.extend(documents)
                        processed += 1
                        
                        # Update MongoDB status
                        self.docs_collection.update_one(
                            {"doc_id": doc_id},
                            {"$set": {
                                "status": "loaded",
                                "pages": len(documents),
                                "updated_at": datetime.now()
                            }}
                        )
                        
                except Exception as e:
                    error_msg = f"Failed to load {fname}: {str(e)[:100]}..."
                    logger.error(error_msg)
                    logger.debug(traceback.format_exc())
                    failed += 1
                    
                    # Update MongoDB with failure
                    self.docs_collection.update_one(
                        {"doc_id": doc_id},
                        {"$set": {
                            "status": "failed",
                            "error": error_msg,
                            "updated_at": datetime.now()
                        }}
                    )
                
                # Update processing log
                self.processing_collection.update_one(
                    {"process_id": process_id},
                    {"$set": {
                        "files_processed": processed,
                        "files_failed": failed
                    }}
                )

        # Update final processing status
        self.processing_collection.update_one(
            {"process_id": process_id},
            {"$set": {
                "status": "loading_complete" if docs else "loading_failed",
                "end_time": datetime.now()
            }}
        )
            
        return docs

    def chunk_documents(
        self,
        docs: List[Document],
        min_chunk_size: int = 200,
        min_chunk_overlap: int = 50
    ) -> List[Document]:
        """Adaptive chunking with validation and MongoDB tracking"""
        # Create processing log entry
        process_id = str(uuid.uuid4())
        self.processing_collection.insert_one({
            "process_id": process_id,
            "start_time": datetime.now(),
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
                
                if chunks and len(chunks[0].page_content) > 50:
                    logger.info(f"Chunking successful: {len(chunks)} chunks")
                    logger.info(f"Sample chunk ({len(chunks[0].page_content)} chars): {chunks[0].page_content[:100]}...")
                    
                    # Store chunk metadata in MongoDB
                    chunk_info = []
                    for i, chunk in enumerate(chunks):
                        chunk_id = str(uuid.uuid4())
                        # Assign chunk ID to document for later reference
                        chunk.metadata["chunk_id"] = chunk_id
                        
                        chunk_info.append({
                            "chunk_id": chunk_id,
                            "doc_id": chunk.metadata.get("doc_id", "unknown"),
                            "filename": chunk.metadata.get("filename", "unknown"),
                            "page": chunk.metadata.get("page", 0),
                            "start_index": chunk.metadata.get("start_index", 0),
                            "length": len(chunk.page_content),
                            "created_at": datetime.now()
                        })
                    
                    # Batch insert chunk metadata
                    if chunk_info:
                        self.chunks_collection.insert_many(chunk_info)
                    
                    # Update processing log
                    self.processing_collection.update_one(
                        {"process_id": process_id},
                        {"$set": {
                            "status": "chunking_complete",
                            "chunks_created": len(chunks),
                            "end_time": datetime.now(),
                            "final_chunk_size": current_size,
                            "final_chunk_overlap": current_overlap
                        }}
                    )
                    
                    return chunks
                    
                current_size = max(current_size // 2, min_chunk_size)
                current_overlap = max(current_overlap // 2, min_chunk_overlap)
                logger.info(f"Adjusting to chunk_size={current_size}, overlap={current_overlap}")
                
                if current_size <= min_chunk_size and current_overlap <= min_chunk_overlap:
                    raise ValueError("Failed to produce valid chunks")

            except Exception as e:
                error_msg = f"Critical chunking error: {str(e)}"
                logger.error(error_msg)
                
                # Update processing log with failure
                self.processing_collection.update_one(
                    {"process_id": process_id},
                    {"$set": {
                        "status": "chunking_failed",
                        "error": error_msg,
                        "end_time": datetime.now()
                    }}
                )
                
                return []

    def ensure_collection_exists(self, recreate: bool = False) -> bool:
        """Ensure Qdrant collection exists with proper schema"""
        # Check if we need to recreate the collection
        if recreate:
            logger.info(f"Attempting to delete collection '{self.collection}'...")
            try:
                self.qdrant.delete_collection(self.collection)
                logger.info("Collection deleted")
            except Exception as e:
                logger.warning(f"Collection deletion skipped: {str(e)[:100]}...")

        # Check if collection exists
        try:
            collection_info = self.qdrant.get_collection(self.collection)
            logger.info(f"Collection '{self.collection}' exists")
            return True
        except Exception:
            # Create collection with Qdrant 1.7 compatible schema
            logger.info(f"Creating collection '{self.collection}' (dim={self.dim})")
            try:
                # Create collection with optimized parameters for Qdrant 1.7
                self.qdrant.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(
                        size=self.dim,
                        distance=Distance.COSINE
                    ),
                    optimizers_config=rest.OptimizersConfigDiff(
                        indexing_threshold=20000,  # Larger batches before indexing
                        memmap_threshold=100000    # When to use memory-mapped files
                    ),
                    # Qdrant 1.7+ specific: Define payload schema for better performance
                    payload_schema={
                        "text": PayloadSchemaType.KEYWORD,
                        "metadata.doc_id": PayloadSchemaType.KEYWORD,
                        "metadata.filename": PayloadSchemaType.KEYWORD,
                        "metadata.page": PayloadSchemaType.INTEGER,
                        "metadata.source": PayloadSchemaType.KEYWORD,
                        "metadata.chunk_id": PayloadSchemaType.KEYWORD
                    }
                )
                
                # Create payload indexes for faster filtering
                self.qdrant.create_payload_index(
                    collection_name=self.collection,
                    field_name="metadata.doc_id",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                
                self.qdrant.create_payload_index(
                    collection_name=self.collection,
                    field_name="metadata.filename",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                
                return True
            except Exception as e:
                logger.error(f"Failed to create collection: {str(e)}")
                return False

    def batch_upsert(self, points: List[PointStruct]) -> int:
        """Robust batch upload with progress tracking and MongoDB updates"""
        total = len(points)
        successful = 0
        
        # Create processing log entry
        process_id = str(uuid.uuid4())
        self.processing_collection.insert_one({
            "process_id": process_id,
            "start_time": datetime.now(),
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
                
                # Update chunk status in MongoDB
                chunk_ids = [point.payload.get("metadata", {}).get("chunk_id") for point in batch]
                self.chunks_collection.update_many(
                    {"chunk_id": {"$in": chunk_ids}},
                    {"$set": {"indexed": True, "indexed_at": datetime.now()}}
                )
                
                progress = min(i + self.batch_size, total)
                successful += len(batch)
                
                # Update processing status
                self.processing_collection.update_one(
                    {"process_id": process_id},
                    {"$set": {"uploaded": progress}}
                )
                
                logger.info(f"Uploaded {progress}/{total} ({progress/total:.1%})")
                
            except Exception as e:
                error_msg = f"Batch upload failed: {str(e)[:100]}..."
                logger.error(error_msg)
                
                # Mark failed chunks
                chunk_ids = [point.payload.get("metadata", {}).get("chunk_id") for point in batch]
                self.chunks_collection.update_many(
                    {"chunk_id": {"$in": chunk_ids}},
                    {"$set": {"indexed": False, "error": error_msg}}
                )
        
        # Update final processing status
        self.processing_collection.update_one(
            {"process_id": process_id},
            {"$set": {
                "status": "upload_complete" if successful == total else "upload_partial",
                "uploaded": successful,
                "end_time": datetime.now()
            }}
        )
        
        logger.info(f"\nUpload completed: {successful}/{total} points successful")
        return successful

    def prepare_vectors(self, chunks: List[Document]) -> List[PointStruct]:
        """Prepare vector embeddings for chunks"""
        logger.info("Preparing embedding points...")
        points = []
        
        # Create processing log entry
        process_id = str(uuid.uuid4())
        self.processing_collection.insert_one({
            "process_id": process_id,
            "start_time": datetime.now(),
            "status": "embedding",
            "total_chunks": len(chunks),
            "processed": 0
        })
        
        for i, chunk in enumerate(chunks):
            try:
                # Generate embedding vector
                vector = self.model.encode(chunk.page_content).tolist()
                
                # Clean and prepare metadata
                metadata = {k: str(v) for k, v in chunk.metadata.items()}
                metadata["source"] = str(metadata.get("source", ""))
                
                # Convert page number to int if possible for better filtering
                if "page" in metadata:
                    try:
                        metadata["page"] = int(metadata["page"])
                    except (ValueError, TypeError):
                        pass
                
                # Create point with Qdrant 1.7 compatible structure
                point_id = metadata.get("chunk_id", str(uuid.uuid4()))
                
                points.append(PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "text": chunk.page_content,
                        "metadata": metadata
                    }
                ))
                
                # Update progress periodically
                if (i + 1) % 10 == 0:
                    self.processing_collection.update_one(
                        {"process_id": process_id},
                        {"$set": {"processed": i + 1}}
                    )
                    
            except Exception as e:
                error_msg = f"Chunk processing failed: {str(e)[:100]}..."
                logger.error(error_msg)
                
                # Update MongoDB with error
                chunk_id = chunk.metadata.get("chunk_id")
                if chunk_id:
                    self.chunks_collection.update_one(
                        {"chunk_id": chunk_id},
                        {"$set": {
                            "embedding_error": error_msg,
                            "embedding_failed": True
                        }}
                    )
        
        # Update final processing status
        self.processing_collection.update_one(
            {"process_id": process_id},
            {"$set": {
                "status": "embedding_complete",
                "processed": len(points),
                "end_time": datetime.now()
            }}
        )
                
        return points
    
    def update_document_status(self, doc_id: str, status: str, error: Optional[str] = None) -> None:
        """Update document status in MongoDB"""
        update = {"status": status, "updated_at": datetime.now()}
        if error:
            update["error"] = error
            
        self.docs_collection.update_one(
            {"doc_id": doc_id},
            {"$set": update}
        )
    
    def process_document_folder(self, folder: str, recreate: bool = False) -> Dict[str, Any]:
        """Process all documents in a folder and return statistics"""
        result = {
            "start_time": datetime.now(),
            "folder": folder,
            "success": False,
            "documents_loaded": 0,
            "chunks_created": 0,
            "vectors_uploaded": 0,
            "errors": []
        }
        
        try:
            # Step 1: Load documents
            logger.info(f"Loading documents from {folder}...")
            docs = self.load_documents(folder)
            if not docs:
                result["errors"].append("No loadable documents found")
                return result
                
            result["documents_loaded"] = len(docs)

            # Step 2: Chunk documents
            logger.info("Chunking documents...")
            chunks = self.chunk_documents(docs)
            if not chunks:
                result["errors"].append("Failed to generate valid chunks")
                return result
                
            result["chunks_created"] = len(chunks)

            # Step 3: Ensure collection exists
            if not self.ensure_collection_exists(recreate):
                result["errors"].append("Failed to create or access vector collection")
                return result

            # Step 4: Prepare vectors
            vectors = self.prepare_vectors(chunks)
            if not vectors:
                result["errors"].append("Failed to generate vector embeddings")
                return result

            # Step 5: Upload vectors
            logger.info(f"Uploading {len(vectors)} vectors to Qdrant...")
            uploaded = self.batch_upsert(vectors)
            result["vectors_uploaded"] = uploaded
            
            # Set success flag if we uploaded all vectors
            result["success"] = uploaded == len(vectors)
            
            # Update document status for all successfully processed docs
            doc_ids = set(doc.metadata.get("doc_id") for doc in docs if "doc_id" in doc.metadata)
            for doc_id in doc_ids:
                self.update_document_status(doc_id, "processed")
                
        except Exception as e:
            error_msg = f"Processing error: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            result["errors"].append(error_msg)
            
        finally:
            result["end_time"] = datetime.now()
            result["duration"] = (result["end_time"] - result["start_time"]).total_seconds()
            
            # Log overall result
            if result["success"]:
                logger.info(f"Processing completed successfully in {result['duration']:.1f} seconds")
            else:
                logger.error(f"Processing failed after {result['duration']:.1f} seconds")
                
            return result


def main():
    """Main entry point for command-line usage"""
    parser = argparse.ArgumentParser(description="Document Processor for RAG System")
    parser.add_argument("--input-folder", required=True, help="Path to documents folder")
    parser.add_argument("--qdrant-host", default="localhost", help="Qdrant host")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant port")
    parser.add_argument("--mongodb-uri", default="mongodb://localhost:27017", help="MongoDB URI")
    parser.add_argument("--mongodb-db", default="rag_system", help="MongoDB database name")
    parser.add_argument("--collection", default="documents", help="Qdrant collection name")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Initial chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Initial chunk overlap")
    parser.add_argument("--batch-size", type=int, default=100, help="Upsert batch size")
    parser.add_argument("--recreate", action="store_true", help="Recreate Qdrant collection")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Embedding model name")
    parser.add_argument("--log-level", default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                        help="Set logging level")
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create processor and process documents
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
    
    # Print summary
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