#!/usr/bin/env python3
"""
RAG Chatbot using Qdrant Vector Search and Mistral-7B
"""
from llama_cpp import Llama
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import argparse

class RAGChatbot:
    def __init__(
        self,
        qdrant_host="localhost",
        qdrant_port=6333,
        collection="documents",
        model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        n_threads=8
    ):
        # Initialize Vector Store (Qdrant)
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection = collection
        
        # Initialize Embedding Model (Vectorization)
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Initialize LLM (Mistral-7B)
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,        # Context window size
            n_threads=n_threads, # CPU threads
            n_gpu_layers=0    # 0 for CPU-only
        )

    def retrieve_documents(self, query: str, top_k: int = 3) -> list:
        """Vector Search with Cosine Similarity"""
        # Convert text to vector
        query_vector = self.encoder.encode(query).tolist()
        
        # Search Qdrant (cosine similarity is default)
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=top_k
        )
        
        return [hit.payload["text"] for hit in results]

    def generate_answer(self, context: list, question: str) -> str:
        """LLM Generation with Context"""
        context_str = "\n".join(context)
        prompt = f"""<s>[INST] Use this context to answer:
{context_str}

Question: {question} [/INST]"""
        
        output = self.llm(
            prompt,
            max_tokens=256,
            temperature=0.7,
            stop=["</s>"]
        )
        
        return output["choices"][0]["text"].strip()

    def query(self, question: str, top_k: int = 3) -> str:
        """Full RAG Pipeline"""
        # 1. Retrieve relevant documents
        context = self.retrieve_documents(question, top_k)
        
        # 2. Generate answer using LLM
        return self.generate_answer(context, question)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qdrant-host", default="localhost")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument("--model-path", default="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    bot = RAGChatbot(
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        model_path=args.model_path,
        n_threads=args.threads
    )

    print("RAG Chatbot Ready (Ctrl+C to exit)")
    while True:
        try:
            question = input("\nQuestion: ").strip()
            if question.lower() in {"exit", "quit"}:
                break
            print("Processing...")
            answer = bot.query(question)
            print(f"\nAnswer: {answer}\n")
        except KeyboardInterrupt:
            break