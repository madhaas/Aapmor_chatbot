#!/usr/bin/env python3
#chatbot.py
"""
Enhanced RAG Chatbot with Qdrant Integration and GGUF Model Support
Features:
- Advanced prompt engineering with structured context formatting
- Local GGUF model support via llama-cpp-python
- SentenceTransformer embeddings for semantic search
- Qdrant vector database for efficient retrieval
- Interactive CLI interface with debug mode
"""
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import logging
import argparse
import torch
import json
import textwrap
import os
import time
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from qdrant_client.http import models as rest

if torch.cuda.is_available:
    name=torch.cuda.get_device_name(0)
    print(name)
# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO  
)
logger = logging.getLogger(__name__)

# Advanced system prompt with clear instructions
SYSTEM_PROMPT = """You are a precise, helpful expert assistant with access to a specialized knowledge base.
Your goal is to provide factual, accurate information based solely on the context provided.

Guidelines:
- Answer ONLY from the provided context; do not use outside knowledge
- If the context doesn't contain the answer, say "I don't have information about that in my knowledge base"
- Cite specific parts of the context when appropriate using [Source #]
- Maintain a professional, concise tone
- Format responses for readability when appropriate
- Be direct and avoid unnecessary qualifiers or hedging
- Do not mention the context or retrieval system in your responses unless asked
"""

# Few-shot examples demonstrating ideal response format
FEW_SHOT_EXAMPLES = [
    {
        "context": [
            "AAPMOR specializes in enterprise-grade AI solutions with a focus on natural language processing and computer vision. The company was founded in 2018 by Dr. Sarah Chen and has offices in Boston and Singapore.",
            "AAPMOR's flagship product, DataSense, enables organizations to process unstructured data at scale through a combination of deep learning and symbolic AI approaches."
        ],
        "question": "What is AAPMOR?",
        "answer": "AAPMOR is an enterprise-grade AI solutions company founded in 2018 by Dr. Sarah Chen. It specializes in natural language processing and computer vision, with offices in Boston and Singapore. Their flagship product is DataSense, which processes unstructured data at scale using a combination of deep learning and symbolic AI. [Source 1, 2]"
    },
    {
        "context": [
            "Document ingestion requires a POST request to the /ingest endpoint with files uploaded as multipart/form-data under the 'files' key.",
            "The /ingest endpoint supports PDF, DOCX, TXT, and HTML formats. Maximum file size is 50MB.",
            "After successful ingestion, the API returns a JSON response with document IDs and processing status."
        ],
        "question": "How do I upload documents to the system?",
        "answer": "To upload documents, send a POST request to the /ingest endpoint with your files as multipart/form-data under the 'files' key. The system supports PDF, DOCX, TXT, and HTML formats with a maximum file size of 50MB. Upon successful upload, you'll receive a JSON response containing document IDs and processing status. [Source 1, 2, 3]"
    }
]

# Main prompt template with sophisticated context structuring
PROMPT_TEMPLATE = """<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{examples}

I'll help you answer the following question based on the retrieved context.

Context information:
{context}

Question: {question} [/INST]"""

# Context formatting function
def format_context(contexts: List[str]) -> str:
    """Format retrieved context chunks with source indicators"""
    formatted = []
    for i, ctx in enumerate(contexts, 1):
        # Trim long contexts to a reasonable length
        if len(ctx) > 1500:
            ctx = ctx[:1500] + "..."
        formatted.append(f"[Source {i}]: {ctx}")
    return "\n\n".join(formatted)

# Examples formatting function
def format_examples(use_examples: bool = True) -> str:
    """Format few-shot examples for the prompt"""
    if not use_examples:
        return ""
        
    examples_text = "Here are examples of how to answer questions based on context:"
    
    for ex in FEW_SHOT_EXAMPLES:
        examples_text += "\n\nContext information:\n"
        examples_text += format_context(ex["context"])
        examples_text += f"\n\nQuestion: {ex['question']}\n"
        examples_text += f"Answer: {ex['answer']}"
    
    examples_text += "\n\n"
    return examples_text


class RAGChatbot:
    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection: str = "documents",
        model_path: str = "/home/haas/rag_proj/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        embedding_model: str = "all-MiniLM-L6-v2",
        n_threads: int = 8
    ):
        """
        Initialize RAG Chatbot components
        """
        # Verify the model path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.encoder = SentenceTransformer(embedding_model, device="cuda")
        
        # Initialize Qdrant client and ensure collection
        self.qdrant = self._init_qdrant(qdrant_host, qdrant_port, collection)
        self.collection = collection
        
        # Initialize LLM - using llama-cpp-python for GGUF model
        logger.info(f"Loading GGUF model: {model_path}")
        self.llm = self._init_llm(model_path, n_threads)

    def _init_qdrant(self, host: str, port: int, collection: str) -> QdrantClient:
        """Initialize and verify Qdrant connection and collection"""
        logger.info(f"Connecting to Qdrant at {host}:{port}")
        client = QdrantClient(host=host, port=port)
        
        try:
            # Check if collection exists by fetching info
            client.get_collection(collection_name=collection)
            logger.info(f"Found existing collection: {collection}")
        except Exception:
            # Create collection with embedding dim & cosine distance
            dim = self.encoder.get_sentence_embedding_dimension()
            logger.info(f"Creating Qdrant collection '{collection}' with dimension {dim}")
            
            client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                optimizers_config=rest.OptimizersConfigDiff(
                    indexing_threshold=20_000,
                    memmap_threshold=100_000
                )
            )
        client = QdrantClient(host=host, port=port, prefer_grpc=True)
        logger.info(f"Connected to Qdrant collection: {collection}")
        # Ensure collection is ready
        
        return client

    def _init_llm(self, model_path: str, n_threads: int) -> Any:
        """Initialize LLM using llama-cpp-python for GGUF model"""
        try:
            # Configure model parameters
            llm = Llama(
                model_path=model_path,
                n_ctx=4096,  # Context window size
                n_threads=n_threads,  # CPU threads
                use_gpu=True,  # Use GPU
                n_gpu_layers=40,  # Number of GPU layers
                low_vram=True,  # Low RAM mode
                verbose=False

            )
            
            logger.info(f"Successfully loaded model {model_path}")
            return llm
            
        except Exception as e:
            logger.error(f"Failed to load LLM: {str(e)}")
            raise

    
    def generate_response(
        self,
        question: str,
        contexts: List[Dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.1,
        use_examples: bool = True
    ) -> str:
        """Generate answer using retrieved context with llama-cpp"""
        try:
            # Extract just the text for the prompt
            context_texts = [ctx["text"] for ctx in contexts]
            formatted_context = format_context(context_texts)
            
            # Prepare the prompt with optional examples
            examples_section = format_examples(use_examples)
            
            prompt = PROMPT_TEMPLATE.format(
                system_prompt=SYSTEM_PROMPT,
                examples=examples_section,
                context=formatted_context,
                question=question
            )
            
            logger.info(f"Generating response with {len(contexts)} context chunks")
            
            # Generate response using llama-cpp
            result = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["</s>", "[INST]", "<s>"],
                echo=False  # Don't include the prompt in the output
            )
            
            # Extract the generated text
            if isinstance(result, dict) and "choices" in result:
                response_text = result["choices"][0]["text"].strip()
            else:
                response_text = result.strip()
            
            return response_text
        except Exception as e:
            logger.error("Generation error: %s", str(e))
            return "Error generating response."

    def query(
        self,
        question: str,
        top_k: int = 5,
        max_tokens: int = 512,
        temperature: float = 0.1,
        use_examples: bool = True
    ) -> Dict[str, Any]:
        """Complete RAG pipeline with guaranteed metadata"""
        base_metadata = {
            "question": question,
            "top_k": top_k,
            "temperature": temperature
        }
        
        if not question.strip():
            return {
                "answer": "Please provide a valid question.",
                "contexts": [],
                "metadata": base_metadata
            }
                
        contexts = self.retrieve_context(question, top_k)
        
        if not contexts:
            return {
                "answer": "I don't have information about that in my knowledge base.",
                "contexts": [],
                "metadata": base_metadata
            }
                
        answer = self.generate_response(
            question, 
            contexts, 
            max_tokens, 
            temperature,
            use_examples
        )
        
        return {
            "answer": answer,
            "contexts": contexts,
            "metadata": base_metadata
        }

    def retrieve_context(
        self,
        question: str,
        top_k: int = 5,
        score_threshold: float = 0.4  # Increased recall
    ) -> List[Dict[str, Any]]:
        """Retrieve context with optimized search parameters"""
        try:
            query_vector = self.encoder.encode(question).tolist()
            
            results = self.qdrant.search(
                collection_name=self.collection,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                search_params=rest.SearchParams(
                    hnsw_ef=128,  # Better search accuracy
                    exact=False
                ),
                with_payload=True
            )
            
            contexts = []
            for hit in results:
                text = hit.payload.get("text", "")
                contexts.append({
                    "text": text,
                    "score": hit.score,
                    "metadata": {
                        k: v for k, v in hit.payload.items()
                        if k != "text"
                    }
                })
                
            return contexts
            
        except Exception as e:
            logger.error("Retrieval failed: %s", e)
            return []


# Interactive mode function - moved out of the class to be a module-level function
def interactive_mode(bot: RAGChatbot):
    """Run the chatbot in interactive mode"""
    logger.info("RAG Chatbot Ready (Ctrl+C to exit)")
    print("\n" + "="*50)
    print("RAG Chatbot Interactive Mode")
    print("="*50)
    print("Type 'exit' or 'quit' to end the session")
    print("Type 'debug on' to enable detailed context display")
    print("Type 'debug off' to disable detailed context display")
    print("Type 'settings' to view current settings")
    print("="*50 + "\n")
        
    debug_mode = False
    top_k = 5
    temperature = 0.1
    use_examples = True
        
    while True:
        try:
            question = input("\nQuestion: ").strip()
            
            if question.lower() in {"exit", "quit"}:
                break
            elif question.lower() == "debug on":
                debug_mode = True
                print("✓ Debug mode enabled - will show retrieved context")
                continue
            elif question.lower() == "debug off":
                debug_mode = False
                print("✓ Debug mode disabled")
                continue
            elif question.lower() == "settings":
                print("\nCurrent Settings:")
                print(f"- Debug Mode: {debug_mode}")
                print(f"- Retrieved Contexts: {top_k}")
                print(f"- Temperature: {temperature}")
                print(f"- Use Examples: {use_examples}")
                continue
            elif question.lower().startswith("set "):
                # Handle settings changes
                parts = question.lower().split()
                if len(parts) >= 3:
                    setting, value = parts[1], parts[2]
                    if setting == "top_k" and value.isdigit():
                        top_k = int(value)
                        print(f"✓ Set top_k to {top_k}")
                    elif setting == "temp" and value.replace(".", "", 1).isdigit():
                        temperature = float(value)
                        print(f"✓ Set temperature to {temperature}")
                    elif setting == "examples" and value in ["on", "off"]:
                        use_examples = (value == "on")
                        print(f"✓ Set examples to {use_examples}")
                    else:
                        print("Invalid setting. Available settings: top_k, temp, examples")
                else:
                    print("Usage: set [setting] [value]")
                continue
                
            # Process the actual query
            if not question:
                continue

            start = time.time()
            result = bot.query(
                question,
                top_k=top_k,
                temperature=temperature,
                use_examples=use_examples
            )
            elapsed = time.time() - start
            print(f"\n⏱ Query took {elapsed:.2f} seconds\n")
 

                

            result = bot.query(
                question, 
                top_k=top_k, 
                temperature=temperature,
                use_examples=use_examples
            )
            
            # For debug mode, show retrieved context first
            if debug_mode and result["contexts"]:
                print("\n" + "="*50)
                print("RETRIEVED CONTEXT")
                print("="*50)
                for i, ctx in enumerate(result["contexts"], 1):
                    score = ctx["score"]
                    text = ctx["text"]
                    # Truncate and format text for display
                    preview = textwrap.shorten(text, width=300, placeholder="...")
                    print(f"\n[Source {i}] (Score: {score:.4f}):\n{preview}")
                print("\n" + "="*50 + "\n")
            
            # Display the answer
            print("\nAnswer:")
            print(result["answer"])
            
        except KeyboardInterrupt:
            print("\nExiting interactive mode...")
            break
        except Exception as e:
            logger.error(f"Error in interactive mode: {e}")
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced RAG Chatbot")
    parser.add_argument("--qdrant-host", default="localhost", help="Qdrant server host")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant server port")
    parser.add_argument("--collection", default="documents", help="Qdrant collection name")
    parser.add_argument("--model", default="/home/haas/rag_proj/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf", 
                       help="Path to GGUF model file")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", 
                       help="SentenceTransformer model to use for embeddings")
    parser.add_argument("--threads", type=int, default=8, help="Number of CPU threads for processing")
    parser.add_argument("--no-examples", action="store_true", help="Disable few-shot examples in prompts")
    parser.add_argument("--top-k", type=int, default=5, help="Number of context chunks to retrieve")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature for generation")
    parser.add_argument("--debug", action="store_true", help="Start in debug mode")
    
    args = parser.parse_args()
    
    try:
        print("\nInitializing RAG Chatbot...")
        print(f"- Model: {args.model}")
        print(f"- Embedding: {args.embedding_model}")
        print(f"- Qdrant: {args.qdrant_host}:{args.qdrant_port}/{args.collection}")
        
        bot = RAGChatbot(
            qdrant_host=args.qdrant_host,
            qdrant_port=args.qdrant_port,
            collection=args.collection,
            model_path=args.model,
            embedding_model=args.embedding_model,
            n_threads=args.threads
        )
        
        interactive_mode(bot)
        
    except Exception as e:
        logger.critical("Initialization failed: %s", str(e))
        exit(1)
