#database.py
from pymongo import MongoClient
from config import settings

client = MongoClient(settings.MONGODB_URL)
db = client[settings.DB_NAME]

def get_db():
    return db