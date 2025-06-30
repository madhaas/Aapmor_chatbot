import sys
import requests
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from qdrant_client.http import models as rest
from qdrant_client.http.models import FilterSelector, Filter, FieldCondition, MatchValue
from pymongo import MongoClient
from uuid import uuid4
from datetime import datetime, timezone
import logging
import textwrap
import os
import asyncio
from functools import partial
from typing import List, Optional, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException, TimeoutException
from webdriver_manager.chrome import ChromeDriverManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("scraper.log")
    ]
)
logger = logging.getLogger("web_scraper")


QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://mongo:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "rag_system")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("SCRAPER_CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("SCRAPER_CHUNK_OVERLAP", 50))
BATCH_SIZE = int(os.getenv("SCRAPER_BATCH_SIZE", 100))
MIN_SCRAPE_WORD_COUNT = int(os.getenv("MIN_SCRAPE_WORD_COUNT", 20))
CHROME_BINARY_PATH = os.getenv("CHROME_BINARY_PATH", "/usr/bin/google-chrome")

# --- Initialize external clients ---
class WebScraper:
    def __init__(self):
        # Initialize MongoDB
        logger.info(f"Connecting to MongoDB at {MONGODB_URI}")
        try:
            self.mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
            self.mongo_client.admin.command('ping')
            self.db = self.mongo_client[MONGODB_DB]
            self.documents_collection = self.db["documents"]
            self.chunks_collection = self.db["chunks"]
            logger.info("Successfully connected to MongoDB.")
        except Exception as e:
            logger.critical(f"Error connecting to MongoDB at {MONGODB_URI}: {e}", exc_info=True)
            raise

        # Initialize Qdrant
        logger.info(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        self.qdrant_client_url = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
        try:
            self.qdrant_client = QdrantClient(
                url=self.qdrant_client_url,
                prefer_grpc=False,
                timeout=5,
                api_key=os.getenv("QDRANT_API_KEY", None),
            )
            self.qdrant_client.get_collections()
            logger.info("Successfully connected to Qdrant.")
            self._ensure_qdrant_collection()
        except Exception as e:
            logger.critical(f"Error connecting to Qdrant at {self.qdrant_client_url}: {e}", exc_info=True)
            raise

        # Initialize embedder
        logger.info(f"Loading SentenceTransformer model: {EMBEDDING_MODEL}")
        try:
            self.embedder = SentenceTransformer(EMBEDDING_MODEL)
            self.vector_size = self.embedder.get_sentence_embedding_dimension()
            logger.info(f"SentenceTransformer model '{EMBEDDING_MODEL}' loaded.")
        except Exception as e:
            logger.critical(f"Error loading SentenceTransformer model '{EMBEDDING_MODEL}': {e}", exc_info=True)
            raise

        # Configure Selenium options for headless execution
        self.chrome_options = Options()
        self.chrome_options.add_argument("--headless")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        self.chrome_options.add_argument("--window-size=1920,1080")
        self.chrome_options.add_argument("--disable-gpu")
        self.chrome_options.add_argument("--disable-extensions")
        self.chrome_options.add_argument("--disable-setuid-sandbox")

        self.chrome_options.binary_location = CHROME_BINARY_PATH
        
        self.chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

    def _ensure_qdrant_collection(self):
        try:
            self.qdrant_client.get_collection(collection_name=QDRANT_COLLECTION)
            logger.info(f"Found existing Qdrant collection: '{QDRANT_COLLECTION}'")
        except Exception:
            logger.info(f"Creating Qdrant collection '{QDRANT_COLLECTION}' with dimension {self.vector_size}")
            self.qdrant_client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
                optimizers_config=rest.OptimizersConfigDiff(
                    indexing_threshold=20_000,
                    memmap_threshold=100_000
                ),
                payload_schema={
                    "url": rest.PayloadSchemaType.KEYWORD,
                    "doc_id": rest.PayloadSchemaType.KEYWORD,
                    "chunk_id": rest.PayloadSchemaType.KEYWORD,
                    "chunk_index": rest.PayloadSchemaType.INTEGER,
                    "text": rest.PayloadSchemaType.TEXT
                }
            )
            self.qdrant_client.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name="url",
                field_schema=rest.PayloadSchemaType.KEYWORD
            )
            logger.info(f"Collection '{QDRANT_COLLECTION}' created and 'url' index added.")

    def _scrape_text_sync(self, url: str) -> str | None:
        """
        Synchronous scraping function using Selenium.
        To be run in a thread pool by the async wrapper.
        """
        driver = None
        page_content = None
        try:
            #service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(options=self.chrome_options)
 
            driver.set_page_load_timeout(30)  # 30 seconds timeout for page load
            
            logger.info(f"Selenium driver initialized for {url}.")
            driver.get(url)
            
            # Wait for body to load
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

            # Get the full HTML content after JS execution
            page_content = driver.page_source

            if not page_content:
                logger.error(f"Selenium returned empty HTML content for {url}.")
                return None

            soup = BeautifulSoup(page_content, "html.parser")

            # Remove unwanted elements
            for element_type in [
                "script", "style", "nav", "footer", "header", "aside", "form", "svg",
                "noscript", "iframe", "meta", "link", "img", "button", "input",
                "select", "textarea", "video", "audio", "canvas", "picture", "source"
            ]:
                for element in soup.find_all(element_type):
                    element.decompose()
            
            # Find main content
            content_tags = soup.find("main") or \
                           soup.find("article") or \
                           soup.find("div", role="main") or \
                           soup.find("div", class_=lambda x: x and any(keyword in x.lower() for keyword in ["content", "body", "section", "main", "page", "article-body", "post-content"])) or \
                           soup.find("div", id=lambda x: x and any(keyword in x.lower() for keyword in ["content", "body", "section", "main", "page", "article-body", "post-content"])) or \
                           soup.find("body")

            extracted_texts = []
            if content_tags:
                text_elements = content_tags.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
                for elem in text_elements:
                    text_line = elem.get_text(separator=" ", strip=True)
                    if text_line:
                        extracted_texts.append(text_line)
            else:
                for p_tag in soup.find_all('p'):
                    text_line = p_tag.get_text(separator=" ", strip=True)
                    if text_line:
                        extracted_texts.append(text_line)
                    
            text = " ".join(extracted_texts)
            text = " ".join(text.split())

            if not text:
                logger.warning(f"No meaningful text extracted from {url} after cleanup and content targeting.")
                return None

            scraped_word_count = len(text.split())
            if scraped_word_count < MIN_SCRAPE_WORD_COUNT:
                logger.warning(f"Scraped text from {url} is too short ({scraped_word_count} words < {MIN_SCRAPE_WORD_COUNT} threshold). It might be irrelevant.")
                return None
            
            logger.info(f"Successfully scraped {scraped_word_count} words from {url}.")
            return text
            
        except (WebDriverException, TimeoutException) as e:
            logger.error(f"Selenium WebDriver error during scraping for {url}: {e}", exc_info=True)
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP Request error scraping URL {url}: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred while scraping {url}: {e}", exc_info=True)
            return None
        finally:
            if driver:
                driver.quit()  # Ensure driver is closed

    async def scrape_text(self, url: str) -> str | None:
        """
        Asynchronous wrapper for the synchronous Selenium scraping function.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            partial(self._scrape_text_sync, url=url)
        )

    def chunk_text(self, text: str, size: int, overlap: int) -> list[str]:
        words = text.split()
        chunks = []
        if not words:
            return chunks

        actual_step = size - overlap
        if actual_step <= 0:
            actual_step = 1 if size == 1 else size // 2
            logger.warning(f"Chunk overlap ({overlap}) is too large for chunk size ({size}). Adjusted step to {actual_step}.")

        for i in range(0, len(words), actual_step):
            part = words[i : i + size]
            if part:
                chunks.append(" ".join(part))
        return chunks

    async def _delete_url_data(self, url: str) -> Tuple[int, int]:
        """
        Deletes all Qdrant points and MongoDB records associated with a given URL.
        Uses the *instance's* QdrantClient for deletion.
        """
        mongo_deleted_count = 0
        qdrant_deleted_count = 0
        try:
            logger.info(f"Attempting to delete existing data for URL: {url} from Qdrant and MongoDB.")
            
            # Delete from Qdrant using FilterSelector (the correct way)
            delete_result = await asyncio.get_event_loop().run_in_executor(
                None,
                partial(
                    self.qdrant_client.delete,
                    collection_name=QDRANT_COLLECTION,
                    points_selector=FilterSelector(
                        filter=Filter(
                            must=[
                                FieldCondition(
                                    key="url",
                                    match=MatchValue(value=url)
                                )
                            ]
                        )
                    ),
                    wait=True
                )
            )
            
            # Handle the delete result properly
            logger.debug(f"Qdrant delete response for {url}: {delete_result}")
            qdrant_deleted_count = (
                delete_result.result.points_deleted
                if delete_result and hasattr(delete_result, 'result') and delete_result.result 
                else 0
            )
            logger.info(f"Deleted {qdrant_deleted_count} points from Qdrant for URL: {url}.")

            # MongoDB deletion
            doc_records = list(self.documents_collection.find({"url": url}, {"_id": 1}))
            doc_ids = [record["_id"] for record in doc_records]

            if doc_ids:
                mongo_chunks_del_result = self.chunks_collection.delete_many({"doc_id": {"$in": doc_ids}})
                logger.info(f"Deleted {mongo_chunks_del_result.deleted_count} chunks from MongoDB for URL: {url}.")
                mongo_docs_del_result = self.documents_collection.delete_many({"_id": {"$in": doc_ids}})
                mongo_deleted_count = mongo_docs_del_result.deleted_count
                logger.info(f"Deleted {mongo_deleted_count} documents from MongoDB for URL: {url}.")

        except Exception as e:
            logger.error(f"Error during deletion of data for URL {url}: {e}", exc_info=True)
            return 0, 0

        return mongo_deleted_count, qdrant_deleted_count

    async def process_url(self, url: str, replace_existing: bool = True) -> Dict[str, Any]:
        start_time = datetime.now(timezone.utc)
        result = {
            "url": url,
            "success": False,
            "scrape_status": "failed",
            "chunks_created": 0,
            "vectors_uploaded": 0,
            "mongo_records_created": 0,
            "qdrant_records_deleted": 0,
            "mongo_records_deleted": 0,
            "errors": []
        }
        logger.info(f"Starting processing for URL: {url}")

        if replace_existing:
            mongo_del, qdrant_del = await self._delete_url_data(url)
            result["mongo_records_deleted"] = mongo_del
            result["qdrant_records_deleted"] = qdrant_del

        text = await self.scrape_text(url)
        
        if not text:
            result["errors"].append(f"Scraping failed or returned insufficient content for {url}.")
            result["scrape_status"] = "failed"
            logger.error(result["errors"][-1])
            return result

        scraped_word_count = len(text.split())
        result["scrape_status"] = "success"
        logger.info(f"Scraped {scraped_word_count} words from {url}.")
        
        doc_id = str(uuid4())
        try:
            self.documents_collection.insert_one({
                "_id": doc_id,
                "url": url,
                "content": text,
                "word_count": scraped_word_count,
                "processed_at": datetime.now(timezone.utc),
                "status": "active"
            })
            result["mongo_records_created"] = 1
            logger.info(f"Stored full document for {url} in MongoDB with doc_id: {doc_id}.")
        except Exception as e:
            result["errors"].append(f"Error storing document for {url} in MongoDB: {e}")
            logger.error(result["errors"][-1], exc_info=True)
            return result

        chunks_text = self.chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks_text:
            result["errors"].append(f"No chunks generated for {url}. Text might be too short or an issue with chunking.")
            logger.warning(result["errors"][-1])
            self.documents_collection.delete_one({"_id": doc_id})
            result["mongo_records_created"] = 0
            return result

        result["chunks_created"] = len(chunks_text)
        logger.info(f"Split text from {url} into {len(chunks_text)} chunks.")

        points_to_upsert = []
        chunks_mongo_records = []
        try:
            logger.info(f"Embedding {len(chunks_text)} chunks for {url}...")
            loop = asyncio.get_event_loop()
            vectors = await loop.run_in_executor(
                None,
                partial(self.embedder.encode, chunks_text, convert_to_numpy=True, show_progress_bar=False)
            )

            for idx, (vec, chunk_text_content) in enumerate(zip(vectors, chunks_text)):
                chunk_id = str(uuid4())
                points_to_upsert.append(
                    PointStruct(
                        id=chunk_id,
                        vector=vec.tolist(),
                        payload={
                            "doc_id": doc_id,
                            "chunk_id": chunk_id,
                            "chunk_index": idx,
                            "url": url,
                            "text": chunk_text_content
                        }
                    )
                )
                chunks_mongo_records.append({
                    "_id": chunk_id,
                    "doc_id": doc_id,
                    "url": url,
                    "chunk_index": idx,
                    "length": len(chunk_text_content),
                    "indexed_at": datetime.now(timezone.utc)
                })

            logger.info(f"Prepared {len(points_to_upsert)} points for Qdrant for URL: {url}.")
        except Exception as e:
            result["errors"].append(f"Error during embedding or point preparation for {url}: {e}")
            logger.error(result["errors"][-1], exc_info=True)
            self.documents_collection.delete_one({"_id": doc_id})
            result["mongo_records_created"] = 0
            return result

        if points_to_upsert:
            try:
                uploaded_count = 0
                for i in range(0, len(points_to_upsert), BATCH_SIZE):
                    batch = points_to_upsert[i:i+BATCH_SIZE]
                    qdrant_res = await loop.run_in_executor(
                        None,
                        partial(self.qdrant_client.upsert,
                                collection_name=QDRANT_COLLECTION,
                                points=batch,
                                wait=True)
                    )
                    if qdrant_res.status == rest.UpdateStatus.COMPLETED:
                        uploaded_count += len(batch)
                    else:
                        raise Exception(f"Qdrant batch upsert failed with status: {qdrant_res.status}")

                result["vectors_uploaded"] = uploaded_count
                logger.info(f"Successfully uploaded {uploaded_count} points for {url} to Qdrant.")

                if chunks_mongo_records:
                    self.chunks_collection.insert_many(chunks_mongo_records)
                    logger.info(f"Stored {len(chunks_mongo_records)} chunk metadata records in MongoDB for {url}.")

                result["success"] = True
            except Exception as e:
                result["errors"].append(f"Error uploading points to Qdrant for {url}: {e}")
                logger.error(result["errors"][-1], exc_info=True)
                await self._delete_url_data(url)
                self.documents_collection.delete_one({"_id": doc_id})
                result["mongo_records_created"] = 0
        else:
            result["errors"].append(f"No points to upload for {url}. Content might be too short or chunking failed.")
            self.documents_collection.delete_one({"_id": doc_id})
            result["mongo_records_created"] = 0

        logger.info(f"Finished processing for URL: {url} with success={result['success']}.")
        return result

    async def delete_url_data_api(self, url: str) -> Dict[str, Any]:
        """Public method to delete data for a URL, used by the API endpoint."""
        logger.info(f"Initiating delete for URL: {url}.")
        start_time = datetime.now(timezone.utc)
        mongo_deleted, qdrant_deleted = await self._delete_url_data(url)
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()

        status = "success" if mongo_deleted > 0 or qdrant_deleted > 0 else "not_found"

        return {
            "url": url,
            "status": status,
            "mongo_records_deleted": mongo_deleted,
            "qdrant_records_deleted": qdrant_deleted,
            "duration": duration,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def __del__(self):
        """Cleanup method to close connections."""
        try:
            if hasattr(self, 'mongo_client'):
                self.mongo_client.close()
                logger.info("MongoDB connection closed.")
        except Exception as e:
            logger.warning(f"Error closing MongoDB connection: {e}")


# Initialize the scraper
try:
    web_scraper = WebScraper()
    logger.info("WebScraper initialized successfully.")
except Exception as e:
    logger.critical(f"Failed to initialize WebScraper: {e}", exc_info=True)
    raise