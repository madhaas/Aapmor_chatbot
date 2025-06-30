from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.security import HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.status import HTTP_403_FORBIDDEN
import logging
import sys
from app.logging_handler import CustomMongoLogHandler
from app.database import get_db
from app.routes import documents, chat, scraper, ingestion, query
from app_config import settings, validate_required_settings
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.mongodb import MongoDBJobStore
from apscheduler.executors.asyncio import AsyncIOExecutor
from datetime import datetime, timedelta, timezone
from app.content_synchronizer import main_sitemap_poll, main_content_check, main_cleanup_deleted

try:
    validate_required_settings()
except ValueError as e:
    logging.critical(f"Configuration validation failed: {e}")
    sys.exit(1)

log_level = getattr(settings, 'LOG_LEVEL', 'INFO')
logging.basicConfig(
    level=getattr(logging, log_level.upper()),
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    stream=sys.stdout
)

mongo_db_instance = get_db()
mongo_handler = CustomMongoLogHandler(
    db=mongo_db_instance,
    collection_name=settings.LOG_MONGODB_COLLECTION
)
mongo_handler.setLevel(logging.INFO)

root_logger = logging.getLogger()
root_logger.addHandler(mongo_handler)

logger = logging.getLogger("main_app_startup")
logger.info("Successfully configured MongoDB logging using custom handler")

app = FastAPI(
    title="RAG API Server",
    description="A comprehensive RAG (Retrieval-Augmented Generation) API with document processing, chat, and web scraping capabilities",
    version="1.0.3",
    docs_url="/docs",
    redoc_url="/redoc"
)

try:
    mongo_db_for_scheduler = get_db()
    if mongo_db_for_scheduler is None:
        raise RuntimeError("Failed to get MongoDB database instance for scheduler")
    
    jobstores = {
        'default': MongoDBJobStore(database=settings.DB_NAME)
    }
    
    logger.info("MongoDBJobStore initialized successfully")
    
except Exception as e:
    logger.error(f"Failed to initialize MongoDBJobStore: {e}")
    logger.warning("Falling back to MemoryJobStore")
    from apscheduler.jobstores.memory import MemoryJobStore
    jobstores = {
        'default': MemoryJobStore()
    }

executors = {
    'default': AsyncIOExecutor()
}
job_defaults = {
    'coalesce': True,
    'max_instances': 1,
    'misfire_grace_time': 30
}

scheduler = AsyncIOScheduler(
    jobstores=jobstores,
    executors=executors,
    job_defaults=job_defaults,
    timezone=timezone.utc
)

allowed_origins = settings.get_cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)
logger.info(f"CORS middleware configured with origins: {allowed_origins}")

async def get_api_key(request: Request):
    api_key = request.headers.get("X-API-Key")
    if not api_key or settings.API_KEY is None or api_key != settings.API_KEY.get_secret_value():
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key"
        )
    return api_key

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "service": "RAG API Server", "version": "1.0.3"}

@app.get("/ping", tags=["Health"])
async def ping():
    return {"message": "pong"}

@app.get("/scheduler/status", tags=["Health"], dependencies=[Depends(get_api_key)])
async def scheduler_status():
    if not scheduler.running:
        return {"status": "scheduler_stopped", "jobs": []}
    
    jobs = []
    for job in scheduler.get_jobs():
        last_run_info = "Not directly tracked by job object"
        jobs.append({
            "id": job.id,
            "name": job.name or job.func.__name__,
            "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
            "trigger": str(job.trigger),
            "last_run": last_run_info,
            "max_instances": job.max_instances,
            "coalesce": job.coalesce
        })
    
    return {
        "status": "running",
        "scheduler_running": scheduler.running,
        "jobs": jobs,
        "job_count": len(jobs),
        "jobstore_type": type(scheduler._jobstores['default']).__name__,
        "timezone": str(scheduler.timezone)
    }

async def logged_job_execution(job_func, job_name):
    start_time = datetime.now(timezone.utc)
    logger.info(f"Starting {job_name} at {start_time.isoformat()}")
    
    try:
        await job_func()
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()
        logger.info(f"{job_name} completed successfully in {duration:.2f} seconds")
    except Exception as e:
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()
        logger.error(f"{job_name} failed after {duration:.2f} seconds: {str(e)}", exc_info=True)
        raise

app.include_router(documents.router, prefix="/api/documents", tags=["Document Management"])
app.include_router(chat.router, prefix="/api/chat", tags=["Chatbot"])
app.include_router(scraper.router, prefix="/api/scraper", tags=["Web Scraping"])
app.include_router(ingestion.router, prefix="/api/ingest", tags=["Document Ingestion"])
app.include_router(query.router, prefix="/api/query", tags=["Advanced Query"])

logger.info("All API routers included successfully.")

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(
        f"HTTPException: {exc.detail} (Status: {exc.status_code}) "
        f"for {request.method} {request.url}",
        exc_info=True
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "success": False,
            "status_code": exc.status_code,
            "path": str(request.url.path)
        }
    )

@app.exception_handler(StarletteHTTPException)
async def starlette_exception_handler(request: Request, exc: StarletteHTTPException):
    logger.warning(
        f"Starlette HTTPException: {exc.detail} (Status: {exc.status_code}) "
        f"for {request.method} {request.url}"
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail or f"HTTP {exc.status_code}",
            "success": False,
            "status_code": exc.status_code,
            "path": str(request.url.path)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(
        f"Unexpected error for {request.method} {request.url}: {str(exc)}",
        exc_info=True
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "success": False,
            "status_code": 500,
            "path": str(request.url.path)
        }
    )

logger.info("Exception handlers registered.")

@app.on_event("startup")
async def startup_event():
    logger.info("RAG API Server starting up...")

    try:
        sitemap_interval = getattr(settings, 'SITEMAP_POLL_INTERVAL_SECONDS', 3600)
        content_check_interval = getattr(settings, 'CONTENT_CHECK_INTERVAL_SECONDS', 1800)
        cleanup_interval_hours = getattr(settings, 'CLEANUP_INTERVAL_HOURS', 24)

        if sitemap_interval < 60:
            logger.warning(f"Sitemap poll interval ({sitemap_interval}s) is very frequent. Consider increasing.")
        if content_check_interval < 30:
            logger.warning(f"Content check interval ({content_check_interval}s) is very frequent. Consider increasing.")

        scheduler.add_job(
            logged_job_execution,
            'interval',
            args=[main_sitemap_poll, "Sitemap Polling"],
            seconds=sitemap_interval,
            id="sitemap_poll_job",
            name="Sitemap Polling",
            next_run_time=datetime.now(timezone.utc) + timedelta(seconds=15),
        )
        logger.info(f"Scheduled Sitemap Polling to run every {sitemap_interval} seconds.")

        scheduler.add_job(
            logged_job_execution,
            'interval',
            args=[main_content_check, "Content Change Detection"],
            seconds=content_check_interval,
            id="content_check_job",
            name="Content Change Detection",
            next_run_time=datetime.now(timezone.utc) + timedelta(seconds=30),
        )
        logger.info(f"Scheduled Content Change Checking to run every {content_check_interval} seconds.")

        scheduler.add_job(
            logged_job_execution,
            'interval',
            args=[main_cleanup_deleted, "Cleanup Deleted URLs"],
            hours=cleanup_interval_hours,
            id="cleanup_deleted_job",
            name="Cleanup Deleted URLs",
            next_run_time=datetime.now(timezone.utc) + timedelta(seconds=45),
        )
        logger.info(f"Scheduled Cleanup of Deleted URLs to run every {cleanup_interval_hours} hours.")

        scheduler.start()
        logger.info("Background job scheduler started successfully.")
        
        logger.info("Scheduled job summary:")
        for job in scheduler.get_jobs():
            next_run = job.next_run_time.strftime("%Y-%m-%d %H:%M:%S UTC") if job.next_run_time else "Not scheduled"
            logger.info(f"   {job.name}: Next run at {next_run}")

        logger.info("All services initialized successfully.")
        logger.info("Autonomous content synchronization is now active!")

    except Exception as e:
        logger.error(f"Failed to initialize scheduler: {str(e)}", exc_info=True)
        logger.warning("Server will continue without background jobs")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("RAG API Server shutting down...")
    
    try:
        if scheduler.running:
            scheduler.shutdown(wait=True)
            logger.info("Background job scheduler shut down gracefully.")
        else:
            logger.info("Scheduler was already stopped.")
    except Exception as e:
        logger.error(f"Error shutting down scheduler: {str(e)}", exc_info=True)
    
    logger.info("RAG API Server shutdown complete.")

@app.post("/api/admin/trigger-sitemap-poll", tags=["Administration"], dependencies=[Depends(get_api_key)])
async def trigger_sitemap_poll():
    try:
        logger.info("Manual sitemap poll triggered via API")
        await logged_job_execution(main_sitemap_poll, "Manual Sitemap Polling")
        return {"success": True, "message": "Sitemap polling completed successfully"}
    except Exception as e:
        logger.error(f"Manual sitemap poll failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Sitemap polling failed: {str(e)}")

@app.post("/api/admin/trigger-content-check", tags=["Administration"], dependencies=[Depends(get_api_key)])
async def trigger_content_check():
    try:
        logger.info("Manual content check triggered via API")
        await logged_job_execution(main_content_check, "Manual Content Change Detection")
        return {"success": True, "message": "Content change checking completed successfully"}
    except Exception as e:
        logger.error(f"Manual content check failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Content checking failed: {str(e)}")

@app.post("/api/admin/trigger-cleanup", tags=["Administration"], dependencies=[Depends(get_api_key)])
async def trigger_cleanup():
    try:
        logger.info("Manual cleanup triggered via API")
        await logged_job_execution(main_cleanup_deleted, "Manual Cleanup Deleted URLs")
        return {"success": True, "message": "Cleanup completed successfully"}
    except Exception as e:
        logger.error(f"Manual cleanup failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

@app.post("/api/admin/scheduler/pause", tags=["Administration"], dependencies=[Depends(get_api_key)])
async def pause_scheduler():
    try:
        scheduler.pause()
        logger.info("Scheduler paused via API")
        return {"success": True, "message": "Scheduler paused successfully"}
    except Exception as e:
        logger.error(f"Failed to pause scheduler: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to pause scheduler: {str(e)}")

@app.post("/api/admin/scheduler/resume", tags=["Administration"], dependencies=[Depends(get_api_key)])
async def resume_scheduler():
    try:
        scheduler.resume()
        logger.info("Scheduler resumed via API")
        return {"success": True, "message": "Scheduler resumed successfully"}
    except Exception as e:
        logger.error(f"Failed to resume scheduler: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to resume scheduler: {str(e)}")

@app.get("/api/admin/scheduler/jobs/{job_id}", tags=["Administration"], dependencies=[Depends(get_api_key)])
async def get_job_details(job_id: str):
    try:
        job = scheduler.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
        
        return {
            "id": job.id,
            "name": job.name or job.func.__name__,
            "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
            "trigger": str(job.trigger),
            "max_instances": job.max_instances,
            "coalesce": job.coalesce,
            "func_ref": str(job.func),
            "args": str(job.args) if job.args else None,
            "kwargs": str(job.kwargs) if job.kwargs else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job details: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get job details: {str(e)}")

if __name__ == "__main__":
    print("FastAPI application entry point")