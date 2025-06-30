import os
import json
from typing import List, Any, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator, SecretStr

class Settings(BaseSettings):
    is_docker_env: bool = os.getenv("USE_DOCKER_HOSTNAMES", "false").lower() == "true"
    
    # --- Database Settings ---
    MONGODB_URL: str = Field(
        default="mongodb://localhost:27017" if not is_docker_env else "mongodb://mongo:27017",
        env="MONGODB_URI",
        description="MongoDB connection URL"
    )
    DB_NAME: str = Field(
        default="rag_system",
        env="DB_NAME",
        description="MongoDB database name"
    )
    MONGODB_CHAT_HISTORY_COLLECTION: str = Field(
        default="chat_history",
        env="MONGODB_CHAT_HISTORY_COLLECTION",
        description="MongoDB collection name for storing chat history"
    )
    QDRANT_CHAT_HISTORY_COLLECTION: str = Field(
        default="chat_history_vectors",
        env="QDRANT_CHAT_HISTORY_COLLECTION",
        description="Qdrant collection name for storing chat history embeddings"
    )
    MONGODB_INQUIRIES_COLLECTION: str = Field(
        default="inquiries",
        env="MONGODB_INQUIRIES_COLLECTION",
        description="MongoDB collection name for storing inquiries"
    )
    
    # --- Logging to MongoDB ---
    LOG_MONGODB_COLLECTION: str = Field(
        default="app_logs",
        env="LOG_MONGODB_COLLECTION",
        description="MongoDB collection name for storing application logs"
    )
    
    # --- Vector Database Settings ---
    QDRANT_HOST: str = Field(
        default="localhost" if not is_docker_env else "qdrant",
        env="QDRANT_HOST",
        description="Qdrant vector database host"
    )
    QDRANT_PORT: int = Field(
        default=6333,
        env="QDRANT_PORT",
        description="Qdrant vector database port"
    )
    QDRANT_COLLECTION: str = Field(
        default="documents",
        env="QDRANT_COLLECTION",
        description="Qdrant collection name for storing document embeddings"
    )
    
    # --- Redis Settings ---
    REDIS_HOST: str = Field(
        default="localhost",
        env="REDIS_HOST",
        description="Redis server hostname"
    )
    REDIS_PORT: int = Field(
        default=6379,
        env="REDIS_PORT",
        description="Redis server port"
    )
    
    # --- Mail Service Settings ---
    MAIL_SERVER: str = Field(
        ...,
        env="MAIL_SERVER",
        description="SMTP mail server hostname"
    )
    MAIL_PORT: int = Field(
        default=587,
        env="MAIL_PORT",
        description="SMTP mail server port"
    )
    MAIL_USERNAME: SecretStr = Field(
        ...,
        env="MAIL_USERNAME",
        description="SMTP mail server username"
    )
    MAIL_PASSWORD: SecretStr = Field(
        ...,
        env="MAIL_PASSWORD",
        description="SMTP mail server password"
    )
    MAIL_FROM: SecretStr = Field(
        ...,
        env="MAIL_FROM",
        description="From email address for outgoing mails"
    )
    MAIL_ENQUIRY_RECIPIENT: SecretStr = Field(
        ...,
        env="MAIL_ENQUIRY_RECIPIENT",
        description="Email address to receive inquiry notifications"
    )
    
    # --- AI/ML API Settings ---
    GROQ_API_KEY: SecretStr = Field(
        ...,
        env="GROQ_API_KEY",
        description="Groq API key for LLM inference (required)"
    )
    GROQ_MODEL_NAME: str = Field(
        default="llama3-8b-8192",
        env="GROQ_MODEL_NAME",
        description="Groq model name to use for chat completions"
    )
    
    # --- Embedding Model Settings ---
    EMBEDDING_MODEL: str = Field(
        default="all-MiniLM-L6-v2",
        env="EMBEDDING_MODEL",
        description="HuggingFace model name for generating embeddings"
    )
    
    # --- Local Model Settings ---
    MODEL_PATH: Optional[str] = Field(
        default=None,
        env="MODEL_PATH",
        description="Path to local model file (if using local inference)"
    )
    THREADS: Optional[int] = Field(
        default=None,
        env="THREADS",
        description="Number of threads for local model inference"
    )
    
    # --- Application Settings ---
    LOG_LEVEL: str = Field(
        default="INFO",
        env="LOG_LEVEL",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    
    # --- CORS Settings ---
    CORS_ORIGINS: List[str] = Field(
        default=[
            "http://localhost:3000",
            "http://127.0.0.1:8000",
            "http://localhost:8000",
            "http://localhost:5173",
            "http://localhost:8080",
        ],
        env="CORS_ORIGINS",
        description="List of allowed CORS origins"
    )
    
    # --- File Processing Settings ---
    MAX_FILE_SIZE: int = Field(
        default=50 * 1024 * 1024,
        env="MAX_FILE_SIZE",
        description="Maximum file size for uploads in bytes"
    )
    ALLOWED_FILE_TYPES: List[str] = Field(
        default=[".pdf", ".txt", ".docx", ".md", ".html"],
        env="ALLOWED_FILE_TYPES",
        description="Allowed file extensions for document uploads"
    )
    
    # --- Text Processing Settings ---
    CHUNK_SIZE: int = Field(
        default=1000,
        env="CHUNK_SIZE",
        description="Default chunk size for text splitting"
    )
    CHUNK_OVERLAP: int = Field(
        default=200,
        env="CHUNK_OVERLAP",
        description="Overlap between text chunks"
    )
    
    # --- Chat Settings ---
    MAX_CHAT_HISTORY: int = Field(
        default=10,
        env="MAX_CHAT_HISTORY",
        description="Maximum number of chat messages to keep in memory"
    )
    MAX_HISTORY_TOKENS: int = Field(
        default=2000,
        env="MAX_HISTORY_TOKENS",
        description="Maximum number of tokens for chat history"
    )
    MAX_RETRIEVED_CHAT_TOKENS: int = Field(
        default=1000,
        env="MAX_RETRIEVED_CHAT_TOKENS",
        description="Maximum number of tokens for retrieved chat history"
    )
    DEFAULT_TEMPERATURE: float = Field(
        default=0.7,
        env="DEFAULT_TEMPERATURE",
        description="Default temperature for LLM responses"
    )
    MAX_TOKENS: int = Field(
        default=4096,
        env="MAX_TOKENS",
        description="Maximum tokens for LLM responses"
    )
    
    # --- Web Scraping Settings ---
    SCRAPER_USER_AGENT: str = Field(
        default="RAG-System-Bot/1.0",
        env="SCRAPER_USER_AGENT",
        description="User agent string for web scraping"
    )
    SCRAPER_TIMEOUT: int = Field(
        default=30,
        env="SCRAPER_TIMEOUT",
        description="Timeout for web scraping requests in seconds"
    )
    MAX_SCRAPE_DEPTH: int = Field(
        default=3,
        env="MAX_SCRAPE_DEPTH",
        description="Maximum depth for recursive web scraping"
    )
    
    # --- Web Content Synchronization Settings ---
    SITEMAP_POLL_INTERVAL_SECONDS: int = Field(
        default=86400,
        description="Interval in seconds for polling website sitemaps.",
        env="SITEMAP_POLL_INTERVAL_SECONDS"
    )
    CONTENT_CHECK_INTERVAL_SECONDS: int = Field(
        default=21600,
        description="Interval in seconds for checking known URLs for content updates.",
        env="CONTENT_CHECK_INTERVAL_SECONDS"
    )
    POTENTIALLY_DELETED_STREAK_LIMIT: int = Field(
        default=3,
        description="Number of consecutive sitemap checks a URL must be missing before being confirmed deleted.",
        env="POTENTIALLY_DELETED_STREAK_LIMIT"
    )
    SYNC_REQUESTS_TIMEOUT_SECONDS: int = Field(
        default=30,
        description="Default timeout in seconds for HTTP requests during content synchronization.",
        env="SYNC_REQUESTS_TIMEOUT_SECONDS"
    )
    SYNC_USER_AGENT: str = Field(
        default="RAGContentSynchronizer/1.0 (+https://your-project-url-or-contact)",
        description="User agent string for content synchronization tasks.",
        env="SYNC_USER_AGENT"
    )
    TARGET_COMPANY_SITEMAPS: List[str] = Field(
        default_factory=list,
        description="A list of root sitemap URLs to monitor (e.g., ['https://company-a.com/sitemap.xml']). Set via env as JSON string or comma-separated.",
        env="TARGET_COMPANY_SITEMAPS"
    )
    FAILURE_THRESHOLD_FOR_URL_STATUS_CHANGE: int = Field(
        default=5,
        description="Number of consecutive failed checks before a URL's status is changed (e.g., to 'unreachable').",
        env="FAILURE_THRESHOLD_FOR_URL_STATUS_CHANGE"
    )
    CLEANUP_INTERVAL_HOURS: int = Field(
        default=24,
        description="Interval in hours for cleanup of deleted URLs from the knowledge base.",
        env="CLEANUP_INTERVAL_HOURS"
    )
    
    # --- Security Settings ---
    API_KEY: Optional[SecretStr] = Field(
        default=None,
        env="API_KEY",
        description="API key for secure access"
    )
    SECRET_KEY: Optional[SecretStr] = Field(
        default=None,
        env="SECRET_KEY",
        description="Secret key for cryptographic operations"
    )
    
    # --- Development Settings ---
    DEBUG: bool = Field(
        default=False,
        env="DEBUG",
        description="Enable debug mode"
    )
    RELOAD: bool = Field(
        default=False,
        env="RELOAD",
        description="Enable auto-reload for development"
    )
    
    # --- Server Settings ---
    HOST: str = Field(
        default="0.0.0.0",
        env="HOST",
        description="Server host address"
    )
    PORT: int = Field(
        default=8000,
        env="PORT",
        description="Server port"
    )
    
    # --- Pydantic Settings Configuration ---
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
        case_sensitive=False
    )
    
    @field_validator("TARGET_COMPANY_SITEMAPS", mode="before")
    @classmethod
    def parse_comma_separated_list(cls, v: Any) -> Any:
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return [item.strip() for item in v.split(',') if item.strip()]
        if isinstance(v, list):
            return v
        return []
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.DEBUG:
            self.LOG_LEVEL = "DEBUG"
    
    @property
    def mongodb_url_safe(self) -> str:
        if "://" in self.MONGODB_URL and "@" in self.MONGODB_URL:
            parts = self.MONGODB_URL.split("://")
            if len(parts) == 2:
                protocol = parts[0]
                rest = parts[1]
                if "@" in rest:
                    creds, host_part = rest.split("@", 1)
                    return f"{protocol}://***:***@{host_part}"
        return self.MONGODB_URL
    
    def get_cors_origins(self) -> List[str]:
        if isinstance(self.CORS_ORIGINS, str):
            return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]
        return self.CORS_ORIGINS

settings = Settings()

def validate_required_settings():
    required_settings = [
        ("GROQ_API_KEY", settings.GROQ_API_KEY.get_secret_value() if settings.GROQ_API_KEY else None),
        ("MONGODB_URL", settings.MONGODB_URL),
        ("QDRANT_HOST", settings.QDRANT_HOST),
    ]
    
    missing_settings = []
    for setting_name, setting_value in required_settings:
        if not setting_value or setting_value == "...":
            missing_settings.append(setting_name)
    
    if missing_settings:
        raise ValueError(
            f"Missing required settings: {', '.join(missing_settings)}. "
            "Please check your .env file or environment variables."
        )