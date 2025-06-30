import gradio as gr
import requests
import uuid
from typing import List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_BASE_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 60

def chat_function_with_state(message: str, history: list, session_id: str):
    if not message.strip():
        return history, ""

    try:
        logger.info(f"Chat request with session_id: {session_id[:8]}...")
        
        payload = {
            "question": message,
            "history": history,
            "session_id": session_id
        }
        
        response = requests.post(
            f"{API_BASE_URL}/api/chat/ask",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        response_data = response.json()
        
        answer = response_data.get("answer", "No response received")
        history.append((message, answer))
        return history, ""

    except requests.exceptions.Timeout:
        error_msg = "Request timed out. Please try again."
        history.append((message, error_msg))
        return history, ""
    except requests.exceptions.ConnectionError:
        error_msg = "Cannot connect to the server. Please ensure the API is running."
        history.append((message, error_msg))
        return history, ""
    except requests.exceptions.HTTPError as e:
        error_msg = f"Server error: {e.response.status_code}"
        history.append((message, error_msg))
        return history, ""
    except Exception as e:
        logger.error(f"Unexpected error in chat: {str(e)}")
        error_msg = "An unexpected error occurred. Please try again."
        history.append((message, error_msg))
        return history, ""

def ingest_url(url: str) -> str:
    if not url.strip():
        return "Please enter a URL"
    
    if not (url.startswith('http://') or url.startswith('https://')):
        return "Please enter a valid URL starting with http:// or https://"
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/scraper/scrape",
            json={"urls": [url], "replace_existing": True},
            headers={"Content-Type": "application/json"},
            timeout=REQUEST_TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            status = data.get("status", "Unknown")
            results = data.get("results", [])
            
            result_text = f"Status: {status}\n\n"
            for result in results:
                url_status = result.get("scrape_status", "Unknown")
                result_url = result.get("url", "Unknown URL")
                result_text += f"URL: {result_url}\nScrape Status: {url_status}\n\n"
            
            return result_text
        else:
            return f"Error {response.status_code}: {response.text}"
            
    except requests.exceptions.Timeout:
        return "Request timed out. The URL might be taking too long to process."
    except requests.exceptions.ConnectionError:
        return "Cannot connect to the server. Please ensure the API is running."
    except Exception as e:
        logger.error(f"Error in ingest_url: {str(e)}")
        return f"Unexpected error: {str(e)}"

def upload_document(file) -> str:
    if file is None:
        return "Please select a file to upload"
    
    max_size_mb = 50
    if hasattr(file, 'size') and file.size > max_size_mb * 1024 * 1024:
        return f"File too large. Maximum size is {max_size_mb}MB."
    
    try:
        files = {"file": (file.name, file, getattr(file, 'type', 'application/octet-stream'))}
        response = requests.post(
            f"{API_BASE_URL}/api/documents/upload",
            files=files,
            timeout=120
        )
        
        if response.status_code == 200:
            data = response.json()
            return f"Success! Document uploaded and processed.\nStatus: {data.get('status', 'Completed')}\nFile: {file.name}"
        else:
            return f"Error {response.status_code}: {response.text}"
            
    except requests.exceptions.Timeout:
        return "Upload timed out. Large files may take longer to process."
    except requests.exceptions.ConnectionError:
        return "Cannot connect to the server. Please ensure the API is running."
    except Exception as e:
        logger.error(f"Error in upload_document: {str(e)}")
        return f"Unexpected error: {str(e)}"

def delete_url(url: str) -> str:
    if not url.strip():
        return "Please enter a URL to delete"
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/scraper/delete-url-data",
            json={"url": url},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            status = data.get("status", "Unknown")
            mongo_deleted = data.get("mongo_records_deleted", 0)
            qdrant_deleted = data.get("qdrant_records_deleted", 0)
            
            return f"Status: {status}\nMongo records deleted: {mongo_deleted}\nQdrant records deleted: {qdrant_deleted}"
        else:
            return f"Error {response.status_code}: {response.text}"
            
    except requests.exceptions.ConnectionError:
        return "Cannot connect to the server. Please ensure the API is running."
    except Exception as e:
        logger.error(f"Error in delete_url: {str(e)}")
        return f"Unexpected error: {str(e)}"

def clear_chat():
    return [], ""

css = """
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
}
.chat-container {
    height: 550px !important;
}
.input-row {
    margin-top: 10px !important;
}
.send-button {
    min-width: 100px !important;
}
"""

with gr.Blocks(title="Aapmor Chat Assistant", theme=gr.themes.Soft(), css=css) as gradio_app:
    gr.Markdown("# Aapmor Chat Assistant")
    gr.Markdown("## Ask anything here to know more about the innovation we bring in")
    gr.Markdown("### Note: This gradio and API setup is for demo purposes only.")
    
    with gr.Tabs():
        with gr.TabItem(" Chat"):
            gr.Markdown(" Type your question below and press Enter or click Send to start chatting.")
            
            session_id_state = gr.State(lambda: str(uuid.uuid4()))
            
            chatbot_display = gr.Chatbot(
                label="Aapmor Chat",
                value=[[None, "Hi! I am Aapmor's virtual assistant. How can I help you today?"]],
                bubble_full_width=False,
                
                show_copy_button=True,
                container=True,
                elem_classes=["chat-container"]
            )
            
            with gr.Row():
                message_input = gr.Textbox(
                    label="Your Message",
                    placeholder="Type your message here...",
                    lines=2,
                    max_lines=5,
                    show_label=False,
                    scale=4
                )
                send_button = gr.Button("Send", variant="primary", scale=1)

            message_input.submit(
                fn=chat_function_with_state,
                inputs=[message_input, chatbot_display, session_id_state],
                outputs=[chatbot_display, message_input],
                show_progress=True
            )
            
            send_button.click(
                fn=chat_function_with_state,
                inputs=[message_input, chatbot_display, session_id_state],
                outputs=[chatbot_display, message_input],
                show_progress=True
            )
        
        with gr.TabItem("Ingest URL"):
            gr.Markdown("### Ingest content from a URL")
            gr.Markdown("Enter a URL below to scrape and add its content to your knowledge base.")
            
            with gr.Row():
                with gr.Column(scale=3):
                    url_input = gr.Textbox(
                        label="URL",
                        placeholder="https://example.com/article",
                        lines=1
                    )
                with gr.Column(scale=1):
                    ingest_btn = gr.Button("Ingest URL", variant="primary", size="lg")
            
            url_output = gr.Textbox(
                label="Results",
                lines=8,
                interactive=False,
                show_copy_button=True
            )
            
            ingest_btn.click(ingest_url, inputs=url_input, outputs=url_output)
            
            gr.Markdown("### Example URLs")
            gr.Markdown("""
            Try these example URLs:
            - News articles
            - Documentation pages  
            - Blog posts
            - Product pages
            """)
        
        with gr.TabItem("Upload Document"):
            gr.Markdown("### Upload a document")
            gr.Markdown("Upload PDF, TXT, or DOCX files to add them to your knowledge base.")
            
            with gr.Row():
                with gr.Column(scale=2):
                    file_upload = gr.File(
                        label="Select Document",
                        file_types=[".pdf", ".txt", ".docx"],
                        file_count="single"
                    )
                with gr.Column(scale=1):
                    upload_btn = gr.Button("Upload Document", variant="primary", size="lg")
            
            upload_output = gr.Textbox(
                label="Results",
                lines=5,
                interactive=False,
                show_copy_button=True
            )
            
            upload_btn.click(upload_document, inputs=file_upload, outputs=upload_output)
            
            gr.Markdown("### Supported Formats")
            gr.Markdown("""
            - **PDF**: Text-based PDFs work best
            - **TXT**: Plain text files
            - **DOCX**: Microsoft Word documents
            """)
        
        with gr.TabItem("Delete URL"):
            gr.Markdown("### Delete URL data")
            gr.Markdown("Remove all data associated with a specific URL from the system.")
            
            with gr.Row():
                with gr.Column(scale=3):
                    delete_url_input = gr.Textbox(
                        label="URL to Delete",
                        placeholder="https://example.com/article",
                        lines=1
                    )
                with gr.Column(scale=1):
                    delete_btn = gr.Button("Delete URL Data", variant="secondary", size="lg")
            
            delete_output = gr.Textbox(
                label="Results",
                lines=5,
                interactive=False,
                show_copy_button=True
            )
            
            delete_btn.click(delete_url, inputs=delete_url_input, outputs=delete_output)
            
            gr.Markdown("### Warning")
            gr.Markdown("This action cannot be undone. Make sure you want to delete all data for the specified URL.")

    gr.Markdown("---")
    gr.Markdown("Powered by Gradio | AI Assistant | Knowledge Base Management")

if __name__ == "__main__":
    logger.info("Starting Gradio application...")
    gradio_app.launch(
        server_name="0.0.0.0",
        server_port=7861,
        debug=True,
        show_error=True
    )