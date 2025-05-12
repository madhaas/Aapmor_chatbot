from app.database import Base, engine
from app import models      # this runs models.py exactly once

if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)
