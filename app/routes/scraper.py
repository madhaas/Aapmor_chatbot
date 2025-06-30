from fastapi import APIRouter, HTTPException, BackgroundTasks, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import logging
from app.scraper import web_scraper 

logger = logging.getLogger("api_scraper_router")

router = APIRouter()

# Pydantic models for request and response
class ScrapeRequest(BaseModel):
    urls: List[str] = Field(..., description="List of URLs to scrape.")
    replace_existing: bool = Field(True, description="If true, delete existing data for the URL(s) before re-ingesting.")

class ScrapeResult(BaseModel):
    url: str
    success: bool
    scrape_status: str
    chunks_created: int
    vectors_uploaded: int
    mongo_records_created: int
    qdrant_records_deleted: int
    mongo_records_deleted: int
    errors: List[str]

class ScrapeResponse(BaseModel):
    status: str
    total_urls_requested: int
    results: List[ScrapeResult]
    failed_urls: List[str]
    message: Optional[str] = None

class DeleteRequest(BaseModel):
    url: str = Field(..., description="The URL for which to delete all associated data.")

class DeleteResponse(BaseModel):
    url: str
    status: str # e.g., "success", "not_found", "failed"
    mongo_records_deleted: int
    qdrant_records_deleted: int
    duration: float
    timestamp: str


# Background task wrapper for scraping
async def _run_scrape_task(urls: List[str], replace_existing: bool, results_list: List[ScrapeResult]):
    """Runs the scraping process for multiple URLs in the background."""
    for url in urls:
        try:
            result = await web_scraper.process_url(url, replace_existing=replace_existing)
            results_list.append(ScrapeResult(**result)) # Append validated Pydantic model
            if not result["success"]:
                logger.error(f"Scraping failed for {url}: {result['errors']}")
        except Exception as e:
            logger.error(f"Unhandled error during scraping {url}: {e}")
            results_list.append(ScrapeResult(
                url=url, success=False, scrape_status="failed",
                chunks_created=0, vectors_uploaded=0, mongo_records_created=0,
                qdrant_records_deleted=0, mongo_records_deleted=0,
                errors=[f"Unhandled internal error: {str(e)}"]
            ))

@router.post("/scrape", response_model=ScrapeResponse, status_code=status.HTTP_202_ACCEPTED, tags=["Scraping"])
async def scrape_urls_endpoint(request: ScrapeRequest, background_tasks: BackgroundTasks):
    if not request.urls:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one URL must be provided."
        )

    processing_results: List[ScrapeResult] = []

    background_tasks.add_task(
        _run_scrape_task,
        request.urls,
        request.replace_existing,
        processing_results
    )

    logger.info(f"Accepted {len(request.urls)} URLs for background scraping.")
    return ScrapeResponse(
        status="processing",
        total_urls_requested=len(request.urls),
        results=[], # Results will be populated asynchronously, so empty for initial response
        failed_urls=[],
        message="Scraping and ingestion initiated in the background. Check logs for progress."
    )

@router.post("/delete-url-data", response_model=DeleteResponse, tags=["Scraping"])
async def delete_url_data_endpoint(request: DeleteRequest):
    if not request.url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="URL must be provided for deletion."
        )
    try:
        delete_result = await web_scraper.delete_url_data_api(request.url)
        if delete_result["status"] == "not_found":
            return DeleteResponse(
                url=request.url,
                status="not_found",
                mongo_records_deleted=0,
                qdrant_records_deleted=0,
                duration=delete_result.get("duration", 0.0),
                timestamp=delete_result.get("timestamp", datetime.now(timezone.utc).isoformat())
            )
        return DeleteResponse(**delete_result)
    except Exception as e:
        logger.error(f"Error deleting data for URL {request.url}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete data for URL {request.url}: {str(e)}"
        )