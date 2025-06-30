import logging
from pymongo.database import Database
from app.database import get_db

class CustomMongoLogHandler(logging.Handler):
    def __init__(self, collection_name: str, db: Database, level=logging.NOTSET):
        super().__init__(level)
        self.collection = db[collection_name]

    def emit(self, record):
        try:
            log_entry = self.format(record)

            log_document = {
                "timestamp": record.created,
                "level": record.levelname,
                "message": record.getMessage(),
                "module": record.module,
                "funcName": record.funcName,
                "lineNo": record.lineno,

            }
            self.collection.insert_one(log_document)
        except Exception:
            self.handleError(record)