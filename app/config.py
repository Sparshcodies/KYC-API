import os
from dotenv import load_dotenv
from pydantic_settings import SettingsConfigDict

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_URL = os.getenv("DB_URL")

MODEL_DIR = os.getenv("MODEL_DIR", "models/auraface")