import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """
    Central configuration object.
    This is the ONLY place environment variables are read.
    """

    MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://mongo:27017")
    MONGO_DB: str = os.getenv("MONGO_DB", "hiring")

settings = Settings()
