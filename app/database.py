#this is database.py
from pymongo import MongoClient
from app_config import settings # <--- CHANGE THIS LINE

client = MongoClient(settings.MONGODB_URL)
db = client[settings.DB_NAME]

def get_db():
    return db