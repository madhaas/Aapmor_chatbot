# app/content_synchronizer.py
import asyncio
import httpx 
from bs4 import BeautifulSoup
import logging
import hashlib
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse
from typing import List, Dict

from app_config import settings 
from app.database import get_db 
from app.scraper import web_scraper 

# Configure logging for the synchronizer
logger = logging.getLogger("content_synchronizer")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Let's assume you've chosen "url_sync_tracker" as the collection name
URL_TRACKER_COLLECTION_NAME = "url_sync_tracker" # Or your chosen name

class ContentSynchronizer:
    def __init__(self):
        self.db = get_db() # Get MongoDB database instance
        self.url_tracker_collection = self.db[URL_TRACKER_COLLECTION_NAME]
        # Consider creating indexes programmatically if they don't exist
        self._ensure_indexes()
        self.client = httpx.AsyncClient(
            timeout=settings.SYNC_REQUESTS_TIMEOUT_SECONDS,
            headers={"User-Agent": settings.SYNC_USER_AGENT}
        )
        logger.info("ContentSynchronizer initialized.")

    def _ensure_indexes(self):
        self.url_tracker_collection.create_index("domain")
        self.url_tracker_collection.create_index("status")
        self.url_tracker_collection.create_index("last_checked_for_changes_at")
        self.url_tracker_collection.create_index("last_known_sitemap_url")
        self.url_tracker_collection.create_index("consecutive_check_failures")
        logger.info(f"Ensured indexes for MongoDB collection: {URL_TRACKER_COLLECTION_NAME}")

    async def fetch_sitemap_content(self, sitemap_url: str) -> str | None:
        """Fetches the content of a given sitemap URL."""
        try:
            response = await self.client.get(sitemap_url)
            response.raise_for_status() # Raise an exception for HTTP error codes
            logger.info(f"Successfully fetched sitemap: {sitemap_url} (Status: {response.status_code})")
            return response.text
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching sitemap {sitemap_url}: {e.response.status_code} - {e.response.text[:200]}")
        except httpx.RequestError as e:
            logger.error(f"Request error fetching sitemap {sitemap_url}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching sitemap {sitemap_url}: {e}", exc_info=True)
        return None

    async def _extract_urls_from_urlset(self, xml_content: str) -> List[str]:
        """Parses XML content known to be a URL set and extracts all page URLs."""
        try:
            soup = BeautifulSoup(xml_content, 'xml')
            urls = []
            
            # Find all <url> tags
            url_tags = soup.find_all('url')
            
            for url_tag in url_tags:
                # Within each <url> tag, find the <loc> tag
                loc_tag = url_tag.find('loc')
                if loc_tag and loc_tag.text:
                    urls.append(loc_tag.text.strip())
            
            logger.info(f"Extracted {len(urls)} URLs from urlset")
            return urls
            
        except Exception as e:
            logger.error(f"Error parsing urlset XML content: {e}", exc_info=True)
            return []

    async def _extract_sitemaps_from_index(self, xml_content: str) -> List[str]:
        """Parses XML content known to be a sitemap index and extracts URLs of sub-sitemaps."""
        try:
            soup = BeautifulSoup(xml_content, 'xml')
            sitemap_urls = []
            
            # Find all <sitemap> tags
            sitemap_tags = soup.find_all('sitemap')
            
            for sitemap_tag in sitemap_tags:
                # Within each <sitemap> tag, find the <loc> tag
                loc_tag = sitemap_tag.find('loc')
                if loc_tag and loc_tag.text:
                    sitemap_urls.append(loc_tag.text.strip())
            
            logger.info(f"Extracted {len(sitemap_urls)} sub-sitemap URLs from sitemap index")
            return sitemap_urls
            
        except Exception as e:
            logger.error(f"Error parsing sitemap index XML content: {e}", exc_info=True)
            return []

    async def _get_all_page_urls_from_sitemap(self, sitemap_url: str, visited_sitemaps: set) -> List[str]:
        """
        Orchestrator for processing a single sitemap URL. It fetches, identifies type (index or urlset),
        and recursively processes if it's an index.
        """
        # Check if sitemap_url is already in visited_sitemaps to prevent infinite loops
        if sitemap_url in visited_sitemaps:
            logger.warning(f"Potential loop detected: sitemap {sitemap_url} already visited. Skipping.")
            return []
        
        # Add sitemap_url to visited_sitemaps
        visited_sitemaps.add(sitemap_url)
        
        # Fetch the XML content
        xml_content = await self.fetch_sitemap_content(sitemap_url)
        if not xml_content:
            logger.warning(f"Failed to fetch content for sitemap: {sitemap_url}")
            return []
        
        try:
            # Parse with BeautifulSoup
            soup = BeautifulSoup(xml_content, 'xml')
            
            # Check if the root tag is <sitemapindex>
            if soup.find('sitemapindex'):
                logger.info(f"Processing sitemap index: {sitemap_url}")
                # Extract sub-sitemap URLs
                sub_sitemap_urls = await self._extract_sitemaps_from_index(xml_content)
                
                # Recursively process each sub-sitemap
                all_page_urls = []
                for sub_sitemap_url in sub_sitemap_urls:
                    sub_urls = await self._get_all_page_urls_from_sitemap(sub_sitemap_url, visited_sitemaps)
                    all_page_urls.extend(sub_urls)
                
                return all_page_urls
            
            # Check if the root tag is <urlset>
            elif soup.find('urlset'):
                logger.info(f"Processing URL set: {sitemap_url}")
                # Extract page URLs
                page_urls = await self._extract_urls_from_urlset(xml_content)
                return page_urls
            
            else:
                # Unknown format
                logger.error(f"Unknown sitemap format for {sitemap_url}. Root tag is neither <sitemapindex> nor <urlset>")
                return []
                
        except Exception as e:
            logger.error(f"Error processing sitemap {sitemap_url}: {e}", exc_info=True)
            return []
    
    async def update_tracker_with_sitemap_data(
        self,
        discovered_urls_by_root_sitemap: Dict[str, List[str]]
    ) -> None:

        logger.info("Starting URL tracker update with sitemap data")
        
        for root_sitemap_url, discovered_page_urls in discovered_urls_by_root_sitemap.items():
            logger.info(f"Processing {len(discovered_page_urls)} discovered URLs from root sitemap: {root_sitemap_url}")
            
            # Convert to set for efficient lookups
            discovered_set = set(discovered_page_urls)
            
            # Extract domain from root sitemap URL
            parsed_url = urlparse(root_sitemap_url)
            current_domain = parsed_url.netloc
            
            try:
                # Fetch existing tracked URLs for this domain and sitemap
                existing_docs = list(self.url_tracker_collection.find(
                    {
                        "domain": current_domain,
                        "last_known_sitemap_url": root_sitemap_url
                    },
                    {"_id": 1, "sitemap_observed_missing_streak": 1}
                ))
                
                # Extract existing URLs into a set
                tracked_set = set(doc["_id"] for doc in existing_docs)
                
                logger.info(f"Found {len(tracked_set)} existing tracked URLs for domain {current_domain}")
                
                # Identify new URLs (in discovered but not in tracked)
                new_urls = discovered_set.difference(tracked_set)
                
                # Identify existing URLs (seen again)
                existing_urls = discovered_set.intersection(tracked_set)
                
                # Identify potentially deleted URLs (in tracked but not in discovered)
                potentially_deleted_urls = tracked_set.difference(discovered_set)
                
                logger.info(f"URL analysis for {current_domain}: {len(new_urls)} new, {len(existing_urls)} existing, {len(potentially_deleted_urls)} potentially deleted")
                
                # Handle new URLs
                if new_urls:
                    new_docs = []
                    current_time = datetime.now(timezone.utc)
                    
                    for new_url in new_urls:
                        new_document = {
                            "_id": new_url,
                            "url": new_url,
                            "domain": current_domain,
                            "doc_id": None,
                            "source_type": "sitemap",
                            "last_successfully_scraped_at": None,
                            "last_checked_for_changes_at": current_time,
                            "content_hash_sha256": None,
                            "http_last_modified": None,
                            "http_etag": None,
                            "sitemap_observed_missing_streak": 0,
                            "consecutive_check_failures": 0,  # Initialize failure counter
                            "last_error": None,  # Initialize error field
                            "status": "active",
                            "last_known_sitemap_url": root_sitemap_url
                        }
                        new_docs.append(new_document)
                    
                    # Use bulk upsert operations for efficiency
                    bulk_operations = []
                    for doc in new_docs:
                        bulk_operations.append({
                            "updateOne": {
                                "filter": {"_id": doc["_id"]},
                                "update": {"$setOnInsert": doc},
                                "upsert": True
                            }
                        })
                    
                    if bulk_operations:
                        result = self.url_tracker_collection.bulk_write(bulk_operations)
                        logger.info(f"Inserted {result.upserted_count} new URLs for domain {current_domain}")
                
                # Handle existing URLs (reset missing streak, update last seen)
                if existing_urls:
                    current_time = datetime.now(timezone.utc)
                    bulk_operations = []
                    
                    for existing_url in existing_urls:
                        bulk_operations.append({
                            "updateOne": {
                                "filter": {"_id": existing_url},
                                "update": {
                                    "$set": {
                                        "sitemap_observed_missing_streak": 0,
                                        "last_checked_for_changes_at": current_time,
                                        "last_known_sitemap_url": root_sitemap_url
                                    }
                                }
                            }
                        })
                    
                    if bulk_operations:
                        result = self.url_tracker_collection.bulk_write(bulk_operations)
                        logger.info(f"Updated {result.modified_count} existing URLs for domain {current_domain}")
                
                # Handle potentially deleted URLs (increment missing streak)
                if potentially_deleted_urls:
                    for missing_url in potentially_deleted_urls:
                        # Find current document to check streak
                        current_doc = self.url_tracker_collection.find_one(
                            {"_id": missing_url},
                            {"sitemap_observed_missing_streak": 1}
                        )
                        
                        if current_doc:
                            current_streak = current_doc.get("sitemap_observed_missing_streak", 0)
                            new_streak = current_streak + 1
                            
                            update_data = {
                                "sitemap_observed_missing_streak": new_streak
                            }
                            
                            # Check if we need to mark as pending deletion
                            if new_streak >= settings.POTENTIALLY_DELETED_STREAK_LIMIT:
                                update_data["status"] = "pending_deletion"
                                logger.warning(f"URL {missing_url} marked as pending_deletion (streak: {new_streak})")
                            
                            self.url_tracker_collection.update_one(
                                {"_id": missing_url},
                                {"$set": update_data}
                            )
                    
                    logger.info(f"Updated missing streak for {len(potentially_deleted_urls)} potentially deleted URLs")
                
            except Exception as e:
                logger.error(f"Error updating tracker for domain {current_domain}: {e}", exc_info=True)
                continue
        
        logger.info("Finished updating URL tracker with sitemap data")

    async def _get_http_headers(self, url: str) -> Dict[str, str]:
        """
        Fetches HTTP headers for a given URL.
        Returns a dictionary of relevant headers (Last-Modified, ETag) if found.
        """
        headers_to_check = {}
        try:
            # Use an HTTP HEAD request to get headers without downloading the body
            head_response = await self.client.head(url, follow_redirects=True)  # Allow redirects
            head_response.raise_for_status()  # Check for HTTP errors

            if 'last-modified' in head_response.headers:
                headers_to_check['last-modified'] = head_response.headers['last-modified']
            if 'etag' in head_response.headers:
                headers_to_check['etag'] = head_response.headers['etag']
            
            logger.debug(f"Fetched headers for {url}: {headers_to_check}")
            return headers_to_check
        except httpx.HTTPStatusError as e:
            logger.warning(f"HTTP error fetching headers for {url}: {e.response.status_code}")
        except httpx.RequestError as e:
            logger.warning(f"Request error fetching headers for {url}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching headers for {url}: {e}", exc_info=True)
        return headers_to_check  # Return empty or partially filled dict on error

    async def _get_content_and_hash(self, url: str) -> tuple[str | None, str | None]:

        logger.debug(f"Attempting to get content and hash for URL: {url}")
        
        try:
            # Use the global web_scraper instance to extract text with the same robust logic
            # that's used for the main RAG pipeline. This ensures consistency in what we
            # consider "meaningful content" for change detection.
            extracted_text = await web_scraper.scrape_text(url)
            
            if not extracted_text:
                logger.warning(f"WebScraper returned no text for {url}")
                return None, None
            
            # Check if extracted text meets minimum word count (similar to WebScraper logic)
            word_count = len(extracted_text.split())
            if word_count < settings.MIN_SCRAPE_WORD_COUNT:
                logger.warning(f"Extracted text from {url} has only {word_count} words, below minimum threshold of {settings.MIN_SCRAPE_WORD_COUNT}")
                return None, None
            
            # Compute SHA256 hash of the extracted text
            text_hash = hashlib.sha256(extracted_text.encode('utf-8')).hexdigest()
            
            logger.info(f"Successfully computed content hash for {url} (text length: {len(extracted_text)} chars, {word_count} words)")
            logger.debug(f"Content hash for {url}: {text_hash}")
            
            return extracted_text, text_hash
            
        except Exception as e:
            logger.error(f"Error extracting content and computing hash for {url}: {e}", exc_info=True)
            return None, None

    async def check_url_for_changes(self, tracked_url_doc: Dict) -> bool:
        """
        Checks if a tracked URL has changed since its last known state.
        Implements a robust failure policy and updates the URL tracker document.
        
        Args:
            tracked_url_doc: Dictionary representing a document from url_sync_tracker collection
            
        Returns:
            bool: True if the URL needs reprocessing (content changed), False otherwise
        """
        url = tracked_url_doc["_id"]
        logger.info(f"Checking for changes: {url}")
        
        needs_reprocessing = False
        current_time = datetime.now(timezone.utc)
        update_fields = {"last_checked_for_changes_at": current_time}

        # Extract stored values
        stored_last_modified = tracked_url_doc.get("http_last_modified")
        stored_etag = tracked_url_doc.get("http_etag")
        stored_content_hash = tracked_url_doc.get("content_hash_sha256")
        consecutive_check_failures = tracked_url_doc.get("consecutive_check_failures", 0)

        header_check_failed = False
        hash_check_failed = False

        
        try:
            logger.debug(f"Performing header check for {url}")
            current_headers = await self._get_http_headers(url)
            
            # Analyze header check results
            if not current_headers and not (stored_last_modified or stored_etag):
                # No headers ever, no headers now - this is normal for some sites
                logger.debug(f"No headers found now or previously for {url}. Proceeding to hash check.")
            elif not current_headers and (stored_last_modified or stored_etag):
                # Had headers before, now none - could indicate an issue
                logger.warning(f"Previously had headers for {url}, but none fetched now. Marking as header check issue.")
                header_check_failed = True
                update_fields["last_error"] = "Previously available HTTP headers (Last-Modified/ETag) are no longer accessible"
            else:
                # We got some headers - process them
                current_last_modified = current_headers.get('last-modified')
                current_etag = current_headers.get('etag')
                
                # Update stored headers
                if current_last_modified:
                    update_fields['http_last_modified'] = current_last_modified
                if current_etag:
                    update_fields['http_etag'] = current_etag
                
                # Check for changes via Last-Modified header
                if (stored_last_modified and current_last_modified and 
                    stored_last_modified != current_last_modified):
                    logger.info(f"Last-Modified header changed for {url}: {stored_last_modified} -> {current_last_modified}")
                    needs_reprocessing = True
                
                # Check for changes via ETag header (if Last-Modified didn't indicate change)
                elif (stored_etag and current_etag and 
                      stored_etag != current_etag):
                    logger.info(f"ETag header changed for {url}: {stored_etag} -> {current_etag}")
                    needs_reprocessing = True
                
                # If this is first time we're getting headers, we'll let content hash decide
                elif not stored_last_modified and not stored_etag and (current_last_modified or current_etag):
                    logger.debug(f"First time capturing headers for {url}. Will rely on content hash for change detection.")
                
                logger.debug(f"Header check successful for {url}")

        except Exception as e:
            logger.error(f"Exception during header check for {url}: {e}", exc_info=True)
            header_check_failed = True
            update_fields["last_error"] = f"Exception during header check: {str(e)[:200]}"
        
        if header_check_failed:
            consecutive_check_failures += 1
            update_fields["status"] = "check_failed_headers"
            logger.warning(f"Header check failed for {url}. Failure count: {consecutive_check_failures}")

        if not needs_reprocessing and not header_check_failed:
            try:
                logger.debug(f"Performing content hash check for {url}")
                extracted_text, current_content_hash = await self._get_content_and_hash(url)
                
                if current_content_hash:
                    # Successfully got content hash
                    update_fields['content_hash_sha256'] = current_content_hash
                    
                    if stored_content_hash and stored_content_hash != current_content_hash:
                        # Content has changed
                        logger.info(f"Content hash changed for {url}: {stored_content_hash[:16]}... -> {current_content_hash[:16]}...")
                        needs_reprocessing = True
                    elif not stored_content_hash:
                        # First time getting content hash - we should process this content
                        logger.info(f"First content hash captured for {url}: {current_content_hash[:16]}...")
                        needs_reprocessing = True
                    else:
                        # Content hash matches - no change
                        logger.debug(f"Content hash unchanged for {url}")
                    
                    logger.debug(f"Content hash check successful for {url}")
                else:
                    # Failed to get content hash
                    hash_check_failed = True
                    update_fields["last_error"] = "Failed to retrieve page content or compute content hash"
                    
            except Exception as e:
                logger.error(f"Exception during content hash check for {url}: {e}", exc_info=True)
                hash_check_failed = True
                update_fields["last_error"] = f"Exception during content hash check: {str(e)[:200]}"

            if hash_check_failed:
                consecutive_check_failures += 1
                update_fields["status"] = "check_failed_content_hash"
                logger.warning(f"Content hash check failed for {url}. Failure count: {consecutive_check_failures}")

        
        if not header_check_failed and not hash_check_failed:
            # All intended checks passed
            if needs_reprocessing:
                logger.info(f"URL {url} needs reprocessing due to detected changes")
                consecutive_check_failures = 0  # Reset on successful check
                update_fields["status"] = "active"
                update_fields["last_error"] = None  # Clear any previous errors
            else:
                # No changes detected but checks were successful
                logger.debug(f"No changes detected for {url}")
                consecutive_check_failures = 0  # Reset on successful check
                update_fields["status"] = "active"
                update_fields["last_error"] = None  # Clear any previous errors

        # Update failure counter
        update_fields["consecutive_check_failures"] = consecutive_check_failures
        
        # Step 4: Apply Failure Threshold Policy
        if consecutive_check_failures >= settings.FAILURE_THRESHOLD_FOR_URL_STATUS_CHANGE:
            # Don't override specific failure statuses, but mark as unreachable if not already marked
            if update_fields.get("status") not in ["check_failed_headers", "check_failed_content_hash"]:
                update_fields["status"] = "unreachable"
            logger.warning(f"URL {url} marked as '{update_fields['status']}' due to {consecutive_check_failures} consecutive failures (threshold: {settings.FAILURE_THRESHOLD_FOR_URL_STATUS_CHANGE})")
        
        # Step 5: Update MongoDB
        try:
            result = self.url_tracker_collection.update_one(
                {"_id": url}, 
                {"$set": update_fields}
            )
            if result.modified_count == 1:
                logger.debug(f"Successfully updated tracker document for {url}")
            else:
                logger.warning(f"No document was modified for {url} - this might indicate the document no longer exists")
                
        except Exception as e:
            logger.error(f"Critical error updating MongoDB document for {url}: {e}", exc_info=True)
            # The needs_reprocessing decision is still valid based on in-memory checks
        
        logger.info(f"Change check completed for {url}: needs_reprocessing={needs_reprocessing}, status={update_fields.get('status', 'unchanged')}")
        return needs_reprocessing

  
    async def mark_url_as_reprocessed(self, url: str):
        """
        Updates a URL's tracker document after it has been successfully reprocessed.
        Resets its status to active and updates the scrape timestamp.
        """
        logger.info(f"Marking URL as successfully reprocessed: {url}")
        try:
            update_data = {
                "status": "active",
                "last_successfully_scraped_at": datetime.now(timezone.utc),
                "consecutive_check_failures": 0, # Reset failure count after successful processing
                "last_error": None
            }
            self.url_tracker_collection.update_one(
                {"_id": url},
                {"$set": update_data}
            )
        except Exception as e:
            logger.error(f"Failed to mark URL as reprocessed in DB for {url}: {e}", exc_info=True)
    

    async def close_client(self):
        await self.client.aclose()



async def main_sitemap_poll():
    """Main function to orchestrate the sitemap polling for all target sitemaps."""
    synchronizer = ContentSynchronizer()
    try:
        target_sitemaps_roots = settings.TARGET_COMPANY_SITEMAPS
        if not target_sitemaps_roots:
            logger.warning("No target company sitemaps configured in settings.TARGET_COMPANY_SITEMAPS. Exiting sitemap poll.")
            return

        logger.info(f"Starting sitemap poll for {len(target_sitemaps_roots)} target root sitemap(s): {target_sitemaps_roots}")
        
        all_discovered_page_urls_by_root = {} # To store URLs per root sitemap processed

        for root_sitemap_url in target_sitemaps_roots:
            logger.info(f"Processing root sitemap: {root_sitemap_url}")
            visited_sitemaps_for_this_root = set() # Reset for each root sitemap to handle distinct trees
            
            # Get all page URLs from this root sitemap (and its potential sub-sitemaps)
            page_urls_from_root = await synchronizer._get_all_page_urls_from_sitemap(root_sitemap_url, visited_sitemaps_for_this_root)
            
            if page_urls_from_root:
                # Deduplicate URLs obtained from this root sitemap processing
                unique_page_urls = list(set(page_urls_from_root))
                all_discovered_page_urls_by_root[root_sitemap_url] = unique_page_urls
                logger.info(f"Discovered {len(unique_page_urls)} unique page URLs from root sitemap: {root_sitemap_url}")
                
                
                
            else:
                logger.warning(f"No page URLs discovered from root sitemap: {root_sitemap_url}")
        
        # --- Update URL tracker database with discovered sitemap data ---
        if all_discovered_page_urls_by_root:
            logger.info("Updating URL tracker database with discovered sitemap data...")
            await synchronizer.update_tracker_with_sitemap_data(all_discovered_page_urls_by_root)
            logger.info("Finished updating URL tracker database.")
        else:
            logger.info("No URLs discovered from sitemaps to update in the tracker.")

    except Exception as e:
        logger.error(f"An error occurred during the main sitemap poll: {e}", exc_info=True)
    finally:
        await synchronizer.close_client()
        logger.info("Sitemap poll process finished.")
    
    
        await synchronizer.close_client()
        logger.info("Sitemap poll process finished.")

async def main_content_check():
    """
    Main function to orchestrate checking active/failed URLs for content changes
    and trigger reprocessing for those that have changed.
    """
    synchronizer = ContentSynchronizer()
    urls_to_reprocess = []
    try:
        logger.info("Starting main content check process...")

        # Calculate the time threshold for when a check is due
        check_interval_seconds = settings.CONTENT_CHECK_INTERVAL_SECONDS
        # Ensure we use datetime.timedelta
        due_time = datetime.now(timezone.utc) - timedelta(seconds=check_interval_seconds)

        # Query MongoDB for URLs that are due for a check
        query = {
            "status": {"$in": ["active", "check_failed_headers", "check_failed_content_hash"]},
            "$or": [
                {"last_checked_for_changes_at": {"$lt": due_time}},
                {"last_checked_for_changes_at": None} # Never checked before
            ]
        }
        
        # Convert cursor to list to avoid cursor timeout if processing takes long
        documents_to_check = await asyncio.to_thread(list, synchronizer.url_tracker_collection.find(query))

        if not documents_to_check:
            logger.info("No URLs are currently due for a content check.")
            return

        logger.info(f"Found {len(documents_to_check)} URLs due for content check.")

        for doc in documents_to_check:
            try:
                if await synchronizer.check_url_for_changes(doc):
                    urls_to_reprocess.append(doc["_id"])
            except Exception as e:
                logger.error(f"Unhandled error during change check for {doc['_id']}: {e}", exc_info=True)

        # --- Re-processing Logic ---
        if urls_to_reprocess:
            logger.info(f"Attempting to reprocess {len(urls_to_reprocess)} URLs that have changed.")
            for url in urls_to_reprocess:
                logger.info(f"--- Starting reprocessing for: {url} ---")
                try:
                    # Call the WebScraper to re-scrape, re-process, and update the knowledge base
                    result = await web_scraper.process_url(url, replace_existing=True)
                    if result and result.get("success"):
                        logger.info(f"Successfully reprocessed: {url}")
                        # Update the tracker with the new status and timestamp
                        await synchronizer.mark_url_as_reprocessed(url)
                    else:
                        error_message = result.get('errors', ['Unknown processing error'])[0] if result else "Unknown processing error"
                        logger.error(f"Failed to reprocess {url}. Reason: {error_message}")
                        # Optional: Update status to "reprocessing_failed" in url_sync_tracker
                except Exception as e:
                    logger.error(f"Critical error during reprocessing of {url}: {e}", exc_info=True)
        else:
            logger.info("No URLs require reprocessing after the content check.")

    except Exception as e:
        logger.critical(f"Critical error in main_content_check process: {e}", exc_info=True)
    finally:
        await synchronizer.close_client()
        logger.info("Main content check process finished.")
        await synchronizer.close_client()
        logger.info("Main content check process finished.")

async def main_cleanup_deleted():
    """
    Finds URLs marked as 'pending_deletion' and removes their data from the
    knowledge base (Qdrant/MongoDB main collections) and the tracker itself.
    """
    synchronizer = ContentSynchronizer()
    try:
        logger.info("Starting cleanup process for deleted URLs...")
        
        # Find all documents marked for deletion
        urls_to_delete_cursor = synchronizer.url_tracker_collection.find(
            {"status": "pending_deletion"}
        )
        documents_to_delete = await asyncio.to_thread(list, urls_to_delete_cursor)

        if not documents_to_delete:
            logger.info("No URLs are currently pending deletion.")
            return

        logger.info(f"Found {len(documents_to_delete)} URLs pending deletion.")

        for doc in documents_to_delete:
            url = doc["_id"]
            logger.warning(f"--- Deleting all data for URL: {url} ---")
            try:
                # 1. Call the WebScraper's deletion API to remove data from Qdrant and main content collections
                delete_result = await web_scraper.delete_url_data_api(url)
                
                if delete_result and delete_result.get("status") in ["success", "not_found"]:
                    logger.info(f"Successfully deleted knowledge base data for {url}. Mongo records deleted: {delete_result.get('mongo_records_deleted')}, Qdrant records deleted: {delete_result.get('qdrant_records_deleted')}.")
                    
                    # 2. After successful deletion, remove the URL from our tracking collection
                    synchronizer.url_tracker_collection.delete_one({"_id": url})
                    logger.info(f"Removed {url} from the url_sync_tracker collection.")
                else:
                    error_msg = delete_result.get("status", "Unknown deletion error") if delete_result else "Unknown deletion error"
                    logger.error(f"Failed to delete knowledge base data for {url}. Reason: {error_msg}")

            except Exception as e:
                logger.error(f"Critical error during deletion of {url}: {e}", exc_info=True)

    except Exception as e:
        logger.critical(f"Critical error in main_cleanup_deleted process: {e}", exc_info=True)
    finally:
        await synchronizer.close_client()
        logger.info("Cleanup process for deleted URLs finished.")

