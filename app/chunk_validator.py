from qdrant_client import QdrantClient
from typing import List, Dict

class ChunkValidator:
    def __init__(self, host: str = "localhost", port: int = 6333):
        self.client = QdrantClient(host=host, port=port)
    
    def verify_chunks_exist(self, collection: str = "documents") -> bool:
        """Check if any chunks exist in the collection"""
        try:
            return self.client.count(collection) > 0
        except Exception as e:
            print(f"Verification failed: {str(e)}")
            return False
    
    def get_chunk_samples(self, collection: str = "documents", num_samples: int = 3) -> List[Dict]:
        """Retrieve sample chunks from Qdrant"""
        try:
            return self.client.scroll(
                collection_name=collection,
                limit=num_samples,
                with_payload=True
            )
        except Exception as e:
            print(f"Failed to retrieve samples: {str(e)}")
            return []