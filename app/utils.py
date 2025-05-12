from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from .models import DocumentChunk, Document
from .database import SessionLocal
from qdrant_client.http import models  # Required for Filter, FieldCondition
import os
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from types import SimpleNamespace
encoder = SentenceTransformer("all-MiniLM-L6-v2")
qdrant = QdrantClient(path="data/qdrant_storage")


def process_document(file_path: str) -> Document:
    """Process document and store in Qdrant"""
    db = SessionLocal()

    # Create document record
    doc = Document(filename=os.path.basename(file_path))
    db.add(doc)
    db.commit()
    db.refresh(doc)  # Ensure doc.id is available

    # Process chunks
def chunk_document(
    file_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> list:
    # 1. Load
    path = Path(file_path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        loader = PyMuPDFLoader(str(path))
    elif ext == ".docx":
        loader = Docx2txtLoader(str(path))
    elif ext == ".txt":
        loader = TextLoader(str(path))
    else:
        return []

    docs = loader.load()  # returns List[Document] with .page_content/.metadata

    # 2. Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True
    )
    chunks = splitter.split_documents(docs)

    # 3. Wrap into simple objects with .text and .metadata
    result = []
    for c in chunks:
        result.append(
            SimpleNamespace(
                text=c.page_content,
                metadata=c.metadata
            )
        )
    return result

    # Store vectors
    points = []
    for idx, chunk in enumerate(chunks):
        vector = encoder.encode(chunk.text).tolist()
        point = models.PointStruct(
            id=f"{doc.id}_{idx}",
            vector=vector,
            payload={
                "text": chunk.text,
                "doc_id": doc.id,
                "metadata": chunk.metadata
            }
        )
        points.append(point)

        # Store chunk reference
        db_chunk = DocumentChunk(
            document_id=doc.id,
            chunk_id=f"{doc.id}_{idx}",
            vector=vector
        )
        db.add(db_chunk)

    qdrant.upsert(collection_name="documents", points=points)
    db.commit()
    return doc


def delete_document(doc_id: int):
    """Delete document and associated vectors"""
    db = SessionLocal()

    # Delete vectors from Qdrant
    qdrant.delete(
        collection_name="documents",
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchValue(value=doc_id)
                    )
                ]
            )
        )
    )

    # Delete from DB
    db.query(DocumentChunk).filter(DocumentChunk.document_id == doc_id).delete()
    db.query(Document).filter(Document.id == doc_id).delete()
    db.commit()
