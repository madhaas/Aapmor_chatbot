# Add connection verification
from qdrant_client import QdrantClient

client = QdrantClient("localhost")
print("Qdrant collections:", client.get_collections())