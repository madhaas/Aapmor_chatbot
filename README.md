#  Aapmor Chatbot System
## Key Features
- **Multi-Source Knowledge Base**: Ingests and processes data from PDFs, DOCX files, TXT files, and dynamic websites.
- **Advanced Conversational Memory**: Utilizes both short-term and long-term (vectorized) memory for coherent, context-aware dialogue.
- **Stateful Human-in-the-Loop**: A robust, Redis-backed inquiry portal to handle user queries when the bot needs help or is explicitly asked.
- **Autonomous Content Sync**: A scheduler runs background jobs to poll sitemaps, check for content changes, and automatically keep the knowledge base up-to-date.
- **Containerized & Scalable**: Fully containerized with Docker Compose for easy, reproducible deployment of all system components (FastAPI, Qdrant, MongoDB, Redis).
- **Secure by Design**: Centralized configuration management with strict separation of secrets.
## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
You will need the following tools installed on your system:
- **Docker** and **Docker Compose**
- **Git** for cloning the repository.
- A text editor or IDE (e.g., VS Code).
- An API key from any inference provider.




### Installation & Setup
1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```
2.  **Create the Environment File**
    Create a file named `.env` in the root of the project directory. Copy the contents of `.env.example` (if provided) or use the template below and fill in your details.

    **`.env` File Template:**
    ```env
    # --- Critical Settings ---
    INferencing_API_KEY=gsk_YourApiKeyHere
    
    # --- Email Notification Settings (for the inquiry portal) ---
    MAIL_SERVER=smtp.gmail.com
    MAIL_PORT=587
    MAIL_USERNAME=your-email@gmail.com
    MAIL_PASSWORD=your-gmail-app-password
    MAIL_FROM=your-email@gmail.com
    MAIL_ENQUIRY_RECIPIENT=recipient-email@example.com
    
    # --- Optional: Target Sitemaps for Auto-Sync ---
    # Comma-separated or a JSON array string
    # TARGET_COMPANY_SITEMAPS="[https://www.aapmor.com/sitemap.xml](https://www.aapmor.com/sitemap.xml)"
    ```
    *Note: For `MAIL_PASSWORD`, you will likely need to generate an "App Password" from your Google account security settings if you are using Gmail.*

3.  **Build and Run with Docker Compose**
    From the root of the project directory, run the following command:
    ```bash
    docker-compose up --build
    ```
    This will build the Docker image for the backend, pull the images for Qdrant, MongoDB, and Redis, and start all the services. The FastAPI server will be available at `http://localhost:8000`.

### Using the API
You can interact with the API using any HTTP client (like `curl` or Postman) or by using the interactive documentation.
- **Interactive Docs (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Alternative Docs (ReDoc):** [http://localhost:8000/redoc](http://localhost:8000/redoc)

#### Example: Ingesting a File
```bash
curl -X POST -F "files=@/path/to/your/document.pdf" http://localhost:8000/api/ingest/
```

#### Example: Scraping a URL
```bash
curl -X POST -H "Content-Type: application/json" -d '{"urls": ["[https://www.aapmor.com/about](https://www.aapmor.com/about)"], "replace_existing": true}' http://localhost:8000/api/scraper/scrape
```

#### Example: Asking a Question
```bash
curl -X POST -H "Content-Type: application/json" -d '{"question": "What does Aapmor specialize in?", "session_id": "user123"}' http://localhost:8000/api/chat/ask
