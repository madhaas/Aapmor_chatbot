# app/models.py
from sqlalchemy import Column, Integer, String, JSON, ForeignKey
from .database import Base

class Document(Base):
    __tablename__ = "documents"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True, index=True)

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    chunk_id = Column(String, index=True)
    vector = Column(JSON)
