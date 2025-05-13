#!/usr/bin/env python3
"""
Enhanced RAG Chatbot with Qdrant Integration and GPU Support
"""
from pathlib import Path
from typing import List, Optional
import logging
import argparse
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class RAGChatbot:
    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection: str = "documents",
        model_path: str = "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        n_threads: int = 8,
        gpu_layers: int = 35
    ):
        """
        Initialize RAG Chatbot components
        """
        self._validate_model_path(model_path)
        
        # Initialize Qdrant client and ensure collection
        self.qdrant = self._init_qdrant(qdrant_host, qdrant_port, collection)
        self.collection = collection
        
        # Initialize embedding model
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Initialize LLM with GPU acceleration
        self.llm = self._init_llm(model_path, n_threads, gpu_layers)

    def _validate_model_path(self, model_path: str) -> None:
        """Verify the model file exists"""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

    def _init_qdrant(self, host: str, port: int, collection: str) -> QdrantClient:
        """Initialize and verify Qdrant connection and collection"""
        client = QdrantClient(host=host, port=port)
        try:
            # Check if collection exists by fetching info
            client.get_collection(collection_name=collection)
        except Exception:
            # Create collection with embedding dim & cosine distance
            dim = self.encoder.get_sentence_embedding_dimension()
            logger.info(f"Creating Qdrant collection '{collection}' with dimension {dim}")
            client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(
                    size=dim,
                    distance=Distance.COSINE
                )
            )
        return client

    def _init_llm(self, model_path: str, n_threads: int, gpu_layers: int) -> Llama:
        """Initialize Mistral LLM with GPU support"""
        try:
            return Llama(
                model_path=model_path,
                n_ctx=4096,
                n_threads=n_threads,
                n_gpu_layers=gpu_layers,
                verbose=False
            )
        except Exception as e:
            logger.error("LLM initialization failed: %s", str(e))
            raise

    def retrieve_context(
        self,
        question: str,
        top_k: int = 3,
        score_threshold: Optional[float] = None
    ) -> List[str]:
        """Retrieve relevant context from Qdrant"""
        try:
            query_vector = self.encoder.encode(question).tolist()
            search_kwargs = dict(
                collection_name=self.collection,
                query_vector=query_vector,
                limit=top_k
            )
            if score_threshold is not None:
                search_kwargs['score_threshold'] = score_threshold
            results = self.qdrant.search(**search_kwargs)
            return [hit.payload.get("text", "") for hit in results]
        except Exception as e:
            logger.error("Retrieval error: %s", str(e))
            return []

    def generate_response(
        self,
        question: str,
        context: List[str],
        max_tokens: int = 512,
        temperature: float = 0.3
    ) -> str:
        """Generate answer using Mistral-7B with retrieved context"""
        try:
            context_str = "\n".join(context)
            prompt = f"""<s>[INST] <<SYS>>
Answer using only the provided context. Be concise and factual.
If unsure, say "I don't know".
<</SYS>>

Context:
{context_str}

Question: {question} [/INST]"""
            out = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["</s>"]
            )
            return out["choices"][0]["text"].strip()
        except Exception as e:
            logger.error("Generation error: %s", str(e))
            return "Error generating response."

    def query(
        self,
        question: str,
        top_k: int = 3,
        max_tokens: int = 512,
        temperature: float = 0.3
    ) -> str:
        """Complete RAG pipeline execution"""
        if not question.strip():
            return "Please provide a valid question."
        context = self.retrieve_context(question, top_k)
        if not context:
            return "No relevant information found."
        return self.generate_response(question, context, max_tokens, temperature)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Chatbot")
    parser.add_argument("--qdrant-host", default="localhost")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument("--model-path", required=True,
                        help="Path to GGUF model file")
    parser.add_argument("--threads", type=int, default=8,
                        help="Number of CPU threads for processing")
    parser.add_argument("--gpu-layers", type=int, default=35,
                        help="Number of transformer layers to offload to GPU")
    parser.add_argument("--collection", default="documents",
                        help="Qdrant collection name")
    args = parser.parse_args()

    try:
        bot = RAGChatbot(
            qdrant_host=args.qdrant_host,
            qdrant_port=args.qdrant_port,
            collection=args.collection,
            model_path=args.model_path,
            n_threads=args.threads,
            gpu_layers=args.gpu_layers
        )
        logger.info("RAG Chatbot Ready (Ctrl+C to exit)")
        while True:
            try:
                question = input("\nQuestion: ").strip()
                if question.lower() in {"exit", "quit"}:
                    break
                answer = bot.query(question)
                print(f"\nAnswer: {answer}\n")
            except KeyboardInterrupt:
                break
    except Exception as e:
        logger.critical("Initialization failed: %s", str(e))
        exit(1)
