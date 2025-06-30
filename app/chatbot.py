#!/usr/bin/env python3
import argparse
import logging
import textwrap
import time
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from groq import Groq
import httpx   
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from qdrant_client.http import models as rest
from sentence_transformers import SentenceTransformer
import torch
from app_config import settings

load_dotenv()

SYSTEM_PROMPT = """
You are the Aapmor AI Assistant, a professional, helpful, and friendly conversational AI. Your primary mission is to provide accurate and concise answers based *only* on the information provided in the conversation history and the knowledge base context.

**Directive for Using <CONVERSATION_HISTORY>:**
You have access to the conversation history. It is CRUCIAL that you actively use this history to:
1.  Understand the full context of the user's current query and the overall dialogue.
2.  Recall and refer to specific details mentioned earlier in the conversation, such as the user's name, to provide a personalized experience.
3.  Maintain conversational coherence and avoid asking for information that the user has already provided.

**Directive for Using <KNOWLEDGE_BASE_CONTEXT>:**
For substantive questions, you will be provided with <KNOWLEDGE_BASE_CONTEXT>. You must:
1.  Thoroughly review this context to find relevant facts, figures, and information that directly answer the user's question.
2.  Base your answer for any substantive question strictly on the information found within this context.

**Core Reasoning and Response Protocol:**
You must follow this protocol for every query:
-   **Analyze Intent:** First, determine if the user's query is simple small talk (e.g., 'hi', 'thanks') or a substantive question.
-   **Handle Small Talk:** If the query is small talk, respond politely and conversationally without relying on the knowledge base.
-   **Answer Substantive Questions:** If it's a substantive question, synthesize a concise answer based *exclusively* on the information found in the <CONVERSATION_HISTORY> and <KNOWLEDGE_BASE_CONTEXT>.

**Strict Rules of Engagement:**
-   **Never Invent Answers:** Your answers must be 100% derived from the provided context. Do not, under any circumstances, use external knowledge or make assumptions.
-   **Handle Lack of Information:** If you cannot find a relevant answer in the provided context, do not apologize. State clearly: "I don't have information on that topic. Can I help with something else?"
-   **Be Proactive and Polite:** Always maintain a polite tone. After providing an answer, proactively ask if the user requires further assistance.
"""

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

logger.info(f"DEBUG: Value of settings.GROQ_API_KEY (first 5 chars): {settings.GROQ_API_KEY.get_secret_value()[:5] + '...' if settings.GROQ_API_KEY.get_secret_value() else 'NOT SET'}")

def _approx_num_tokens(text: str) -> int:
    """Approximate token count for text using word count + char_count / 4 heuristic."""
    return len(text.split()) + len(text) // 4

class RAGChatbot:
    def __init__(
        self,
        qdrant_host: str = settings.QDRANT_HOST,
        qdrant_port: int = settings.QDRANT_PORT,
        collection: str = settings.QDRANT_COLLECTION,
        embedding_model: str = settings.EMBEDDING_MODEL,
        score_threshold: float = 0.25
    ):
        self.encoder = SentenceTransformer(
            embedding_model,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        self.qdrant_main_collection = collection  
        self.qdrant_chat_collection = settings.QDRANT_CHAT_HISTORY_COLLECTION  
        self.score_threshold = score_threshold

        httpx_client_without_proxies = httpx.Client(
        )

        self.llm_client = Groq(
            api_key=settings.GROQ_API_KEY.get_secret_value(),
            # Pass the pre-configured httpx client to Groq
            http_client=httpx_client_without_proxies 
        )
        self.llm_model = settings.GROQ_MODEL_NAME

        self.qdrant = self._init_qdrant_client(qdrant_host, qdrant_port)
        self._ensure_document_collection(self.qdrant_main_collection)
        self._ensure_chat_history_collection(self.qdrant_chat_collection)

    def _init_qdrant_client(self, host: str, port: int) -> QdrantClient:
        """Initialize Qdrant client"""
        client = QdrantClient(url=f"http://{host}:{port}", prefer_grpc=False)
        logger.info(f"Qdrant client initialized, connected to http://{host}:{port}")
        return client

    def _ensure_document_collection(self, collection_name: str):
        """Ensure the main document collection exists"""
        try:
            self.qdrant.get_collection(collection_name)
            logger.info(f"Connected to existing document collection: {collection_name}")
        except Exception:
            logger.info(f"Creating new document collection: {collection_name}")
            dim = self.encoder.get_sentence_embedding_dimension()
            
            self.qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=dim,
                    distance=Distance.COSINE
                )
            )
            
            try:
                self.qdrant.create_payload_index(
                    collection_name=collection_name,
                    field_name="text",
                    field_schema=rest.PayloadSchemaType.TEXT
                )
                self.qdrant.create_payload_index(
                    collection_name=collection_name,
                    field_name="source",
                    field_schema=rest.PayloadSchemaType.KEYWORD
                )
                logger.info(f"Created payload indexes for document collection '{collection_name}'")
            except Exception as e:
                logger.warning(f"Could not create payload indexes for '{collection_name}': {e}")

    def _ensure_chat_history_collection(self, collection_name: str):
        """Ensure the chat history Qdrant collection exists"""
        try:
            self.qdrant.get_collection(collection_name)
            logger.info(f"Connected to existing chat history collection: {collection_name}")
        except Exception:
            logger.info(f"Creating new chat history collection: {collection_name}")
            dim = self.encoder.get_sentence_embedding_dimension()
            
            self.qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=dim,
                    distance=Distance.COSINE
                )
            )
            
            try:
                self.qdrant.create_payload_index(
                    collection_name=collection_name,
                    field_name="session_id",
                    field_schema=rest.PayloadSchemaType.KEYWORD
                )
                self.qdrant.create_payload_index(
                    collection_name=collection_name,
                    field_name="timestamp_iso",
                    field_schema=rest.PayloadSchemaType.KEYWORD
                )
                self.qdrant.create_payload_index(
                    collection_name=collection_name,
                    field_name="user_query_preview",
                    field_schema=rest.PayloadSchemaType.TEXT
                )
                logger.info(f"Created payload indexes for chat history collection '{collection_name}'")
            except Exception as e:
                logger.warning(f"Could not create payload indexes for chat history collection '{collection_name}': {e}")

    def _trim_history_by_tokens(
        self,
        history: List[Tuple[Optional[str], str]],
        max_tokens: int
    ) -> List[Tuple[Optional[str], str]]:
        """
        Trims the conversation history from the oldest entries to fit within
        the specified maximum token limit.
        Each turn (user + bot message) counts towards the token limit.
        """
        if not history or max_tokens <= 0:
            return []
        trimmed_history = []
        current_tokens = 0
        for user_msg, bot_msg in reversed(history):
            turn_tokens = 0
            if user_msg is not None:
                turn_tokens += _approx_num_tokens(user_msg)
            if bot_msg is not None:
                turn_tokens += _approx_num_tokens(bot_msg)
            if current_tokens + turn_tokens <= max_tokens:
                trimmed_history.insert(0, (user_msg, bot_msg))
                current_tokens += turn_tokens
            else:
                break
        logger.debug(f"History trimmed from {len(history)} to {len(trimmed_history)} turns, totaling approx {current_tokens} tokens.")
        return trimmed_history

    def _retrieve_chat_history_context(
        self,
        query_text: str,
        session_id: str,
        top_n_turns: int = 3
    ) -> List[str]:
        """
        Retrieves semantically relevant past chat interactions from Qdrant
        for the current session, up to a configured token limit.
        """
        if not query_text.strip():
            return []
        retrieved_history_texts = []
        current_tokens = 0
        try:
            query_vector = self.encoder.encode(query_text).tolist()
            search_results = self.qdrant.search(
                collection_name=self.qdrant_chat_collection,
                query_vector=query_vector,
                limit=top_n_turns,
                with_payload=True,
                query_filter=rest.Filter(
                    must=[
                        rest.FieldCondition(
                            key="session_id",
                            match=rest.MatchValue(value=session_id)
                        )
                    ]
                ),
            )
            for hit in search_results:
                past_turn_text = hit.payload.get("full_interaction_text")
                if past_turn_text:
                    turn_tokens = _approx_num_tokens(past_turn_text)
                    if current_tokens + turn_tokens <= settings.MAX_RETRIEVED_CHAT_TOKENS:
                        retrieved_history_texts.append(past_turn_text)
                        current_tokens += turn_tokens
                    else:
                        break
            logger.debug(f"Retrieved {len(retrieved_history_texts)} semantically relevant chat turns (approx {current_tokens} tokens) for session {session_id}.")
            return retrieved_history_texts
        except Exception as e:
            logger.error(f"Error retrieving chat history context for session {session_id}: {e}", exc_info=True)
            return []

    def query(
        self,
        question: str,
        session_id: str,
        history: Optional[List[Tuple[Optional[str], str]]] = None,
        top_k: int = 5,
        temperature: float = 0.1,
        max_tokens: int = 512,
        score_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Full RAG pipeline execution with conversational memory"""
        if not question.strip():
            return {"answer": "Please provide a valid question.", "contexts": []}

        score_threshold = score_threshold or self.score_threshold
        
        contexts = self.retrieve_context(question, top_k, score_threshold)
        
        recent_history_summary = ""
        if history:
            temp_summary_tokens = 0
            for user_msg, bot_msg in reversed(history[-settings.MAX_CHAT_HISTORY:]):
                turn_text = ""
                if user_msg is not None: turn_text += user_msg + " "
                if bot_msg is not None: turn_text += bot_msg + " "
                if temp_summary_tokens + _approx_num_tokens(turn_text) <= 300:
                    recent_history_summary = turn_text + recent_history_summary
                    temp_summary_tokens += _approx_num_tokens(turn_text)
                else:
                    break
        combined_retrieval_query_text = f"{recent_history_summary.strip()} {question}" if recent_history_summary else question
        
        retrieved_chat_history_contexts = self._retrieve_chat_history_context(
            query_text=combined_retrieval_query_text,
            session_id=session_id,
            top_n_turns=settings.MAX_CHAT_HISTORY
        )
        
        answer = self.generate_response(
            question=question,
            contexts=contexts,
            history=history,
            retrieved_chat_history_contexts=retrieved_chat_history_contexts,
            temperature=temperature,
            max_tokens=max_tokens
        ) if contexts else "I couldn't find a specific answer for that in my knowledge base. Please try rephrasing your question or ask a different one."
        
        return {
            "answer": answer,
            "contexts": contexts,
            "metadata": {
                "question": question,
                "top_k": top_k,
                "temperature": temperature,
                "score_threshold": score_threshold,
                "max_tokens": max_tokens,
                "retrieved_chat_history_count": len(retrieved_chat_history_contexts)
            }
        }

    def retrieve_context(
        self,
        question: str,
        top_k: int,
        score_threshold: Optional[float]
    ) -> List[Dict[str, Any]]:
        """Configurable similarity search from main document collection"""
        query_vector = self.encoder.encode(question).tolist()
        
        results = self.qdrant.search(
            collection_name=self.qdrant_main_collection,  
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True
        )

        return [
            {
                "text": hit.payload["text"],
                "score": hit.score,
                "metadata": {k: v for k, v in hit.payload.items() if k != "text"}
            }
            for hit in results if "text" in hit.payload
        ]

    def generate_response(
        self,
        question: str,
        contexts: List[Dict],
        history: Optional[List[Tuple[Optional[str], str]]] = None,
        retrieved_chat_history_contexts: Optional[List[str]] = None,
        temperature: float = 0.1,
        max_tokens: int = 512
    ) -> str:
        """LLM response generation with conversational memory support"""
        context_text = "\n".join(f"- {ctx['text']}" for ctx in contexts)
        
        messages = []
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
        
        if history:
            trimmed_recent_history = self._trim_history_by_tokens(history, settings.MAX_HISTORY_TOKENS)
            for user_msg, bot_msg in trimmed_recent_history:
                if user_msg is not None:
                    messages.append({"role": "user", "content": user_msg})
                if bot_msg is not None:
                    messages.append({"role": "assistant", "content": bot_msg})
        
        if retrieved_chat_history_contexts:
            retrieved_chat_str = "\n".join(retrieved_chat_history_contexts)
            messages.append({"role": "user", "content": f"Here is some relevant past conversation for additional context:\n{retrieved_chat_str}"})
        
        if context_text:
            messages.append({"role": "user", "content": f"Here is some relevant document information:\n{context_text}"})
        
        messages.append({"role": "user", "content": question})
        
        response = self.llm_client.chat.completions.create(
            messages=messages,
            model=self.llm_model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()

def interactive_mode(bot: RAGChatbot, initial_params: Dict[str, Any], initial_debug: bool = False):
    """Enhanced CLI interface with parameter control"""
    debug_mode = initial_debug
    params = initial_params.copy()

    print("\n== RAG Chatbot ==")
    print("Commands: debug, params, set <param> <value>, exit")
    print("Available parameters: top_k, temperature, score_threshold, max_tokens\n")

    while True:
        try:
            query = input("\nQuestion: ").strip()
            
            if not query:
                continue
            if query.lower() in ("exit", "quit"):
                break
            elif query.lower() == "debug":
                debug_mode = not debug_mode
                print(f"Debug mode {'ON' if debug_mode else 'OFF'}")
                continue
            elif query.lower() == "params":
                print("\nCurrent Parameters:")
                for k, v in params.items():
                    print(f"  {k}: {v}")
                continue
            elif query.startswith("set "):
                try:
                    parts = query.split()
                    if len(parts) != 3:
                        print("Usage: set <parameter> <value>")
                        continue
                    _, param, value = parts
                    param = param.lower()
                    if param in params:
                        if param in ['temperature', 'score_threshold']:
                            params[param] = float(value)
                        else:
                            params[param] = int(value)
                        print(f"Set {param} to {value}")
                    else:
                        print(f"Invalid parameter. Valid options: {list(params.keys())}")
                except Exception as e:
                    print(f"Error setting parameter: {e}")
                continue

            start_time = time.time()
            session_id = "cli_session"
            result = bot.query(
                question=query,
                session_id=session_id,
                history=None,
                top_k=params['top_k'],
                temperature=params['temperature'],
                score_threshold=params['score_threshold'],
                max_tokens=params['max_tokens']
            )
            elapsed = time.time() - start_time

            if debug_mode:
                print(f"\n[Debug] Retrieved {len(result['contexts'])} contexts (Threshold: {params['score_threshold']})")
                print(f"[Debug] Retrieved {result['metadata']['retrieved_chat_history_count']} chat history contexts")
                for i, ctx in enumerate(result['contexts'], 1):
                    print(f"\nContext {i} - Score: {ctx['score']:.4f}")
                    print(textwrap.shorten(ctx['text'], width=300, placeholder="..."))
                    if ctx['metadata']:
                        print(f"Metadata: {ctx['metadata']}")
                    print("-" * 50)

            print(f"\nAnswer ({elapsed:.2f}s):")
            print(textwrap.fill(result['answer'], width=100))

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Chatbot")
    parser.add_argument("--qdrant-host", default=settings.QDRANT_HOST)
    parser.add_argument("--qdrant-port", type=int, default=settings.QDRANT_PORT)
    parser.add_argument("--collection", default=settings.QDRANT_COLLECTION)
    parser.add_argument("--embedding-model", default=settings.EMBEDDING_MODEL)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--score-threshold", type=float)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    try:
        bot = RAGChatbot(
            qdrant_host=args.qdrant_host,
            qdrant_port=args.qdrant_port,
            collection=args.collection,
            embedding_model=args.embedding_model,
            score_threshold=args.score_threshold or 0.4
        )
        
        initial_params = {
            'top_k': args.top_k,
            'temperature': args.temperature,
            'score_threshold': args.score_threshold or bot.score_threshold,
            'max_tokens': args.max_tokens
        }
        
        interactive_mode(
            bot,
            initial_params=initial_params,
            initial_debug=args.debug
        )
    except Exception as e:
        logger.critical(f"Initialization failed: {str(e)}")
        exit(1)