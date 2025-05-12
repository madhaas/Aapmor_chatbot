
"""
document processor with Qdrant 1.7 compatibility
"""

import argparse
import os
import uuid
import json
import traceback
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


def load_documents(folder: str) -> List[Document]:
    """Load documents with enhanced error recovery"""
    docs: List[Document] = []
    supported_extensions = ['.pdf', '.docx', '.txt']
    
    for root, _, files in os.walk(folder):
        for fname in files:
            path = Path(root) / fname
            ext = path.suffix.lower()
            
            if ext not in supported_extensions:
                continue

            try:
                loader = None
                if ext == '.pdf':
                    loader = PyMuPDFLoader(str(path))
                elif ext == '.docx':
                    loader = Docx2txtLoader(str(path))
                elif ext == '.txt':
                    loader = TextLoader(str(path))

                if loader:
                    documents = loader.load()
                    print(f" Loaded {len(documents)} pages from {fname}")
                    docs.extend(documents)
                    
            except Exception as e:
                print(f" Failed to load {fname}: {str(e)[:100]}...")
                print(traceback.format_exc())

    return docs


def chunk_documents(
    docs: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    min_chunk_size: int = 200,
    min_chunk_overlap: int = 50
) -> List[Document]:
    """Adaptive chunking with validation"""
    current_size = chunk_size
    current_overlap = chunk_overlap
    
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
                print(f" Chunking successful: {len(chunks)} chunks")
                print(f" Sample chunk ({len(chunks[0].page_content)} chars):")
                print(chunks[0].page_content[:200] + "...")
                return chunks
                
            current_size = max(current_size // 2, min_chunk_size)
            current_overlap = max(current_overlap // 2, min_chunk_overlap)
            print(f" Adjusting to chunk_size={current_size}, overlap={current_overlap}")
            
            if current_size <= min_chunk_size and current_overlap <= min_chunk_overlap:
                raise ValueError("Failed to produce valid chunks")

        except Exception as e:
            print(f" Critical chunking error: {e}")
            return []


def batch_upsert(
    client: QdrantClient,
    collection: str,
    points: List[PointStruct],
    batch_size: int = 100
) -> None:
    """Robust batch upload with progress tracking"""
    total = len(points)
    for i in range(0, total, batch_size):
        batch = points[i:i+batch_size]
        try:
            client.upsert(
                collection_name=collection,
                points=batch,
                wait=True
            )
            progress = min(i + batch_size, total)
            print(f" Uploaded {progress}/{total} ({progress/total:.1%})", end='\r')
        except Exception as e:
            print(f" Batch upload failed: {str(e)[:100]}...")
    print("\n Upload completed")


def main():
    parser = argparse.ArgumentParser(description="Document Processor for RAG System")
    parser.add_argument("--input-folder", required=True, help="Path to documents folder")
    parser.add_argument("--qdrant-host", default="localhost", help="Qdrant host")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant port")
    parser.add_argument("--collection", default="documents", help="Collection name")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Initial chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Initial chunk overlap")
    parser.add_argument("--recreate", action="store_true", help="Recreate collection")
    args = parser.parse_args()

    # Document loading
    print("\n Loading documents...")
    docs = load_documents(args.input_folder)
    if not docs:
        print(" No loadable documents found")
        return

    # Document chunking
    print("\n Chunking documents...")
    chunks = chunk_documents(
        docs,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    if not chunks:
        print(" Failed to generate valid chunks")
        return

    # Qdrant connection (version-safe)
    print("\n Connecting to Qdrant...")
    client = QdrantClient(host=args.qdrant_host, port=args.qdrant_port)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    dim = model.get_sentence_embedding_dimension()

    # Collection management
    if args.recreate:
        print(f"\n Attempting to delete collection '{args.collection}'...")
        try:
            client.delete_collection(args.collection)
            print(" Collection deleted")
        except Exception as e:
            print(f" Collection deletion skipped: {str(e)[:100]}...")

    # Create collection if needed
    try:
        collection_info = client.get_collection(args.collection)
    except:
        print(f"\n Creating collection '{args.collection}' (dim={dim})")
        client.create_collection(
            collection_name=args.collection,
            vectors_config=VectorParams(
                size=dim,
                distance=Distance.COSINE
            )
        )

    # Prepare points
    print("\n Preparing embedding points...")
    points = []
    for chunk in chunks:
        try:
            vector = model.encode(chunk.page_content).tolist()
            metadata = {k: str(v) for k, v in chunk.metadata.items()}
            metadata["source"] = str(metadata.get("source", ""))
            
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": chunk.page_content,
                    "metadata": metadata
                }
            ))
        except Exception as e:
            print(f" Chunk processing failed: {str(e)[:100]}...")

    # Batch upsert
    if points:
        print(f"\nUploading {len(points)} vectors to Qdrant...")
        batch_upsert(client, args.collection, points)
        print("\nProcessing completed successfully")
    else:
        print(" No valid points to upload")


if __name__ == "__main__":
    main()