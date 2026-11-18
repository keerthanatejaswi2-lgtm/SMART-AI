import os
from pathlib import Path
from dotenv import load_dotenv
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
APP_NAME = os.getenv("APP_NAME", "SmartFlowAI")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG = os.getenv("DEBUG", "True").lower() == "true"
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///smartflow.db")
MODEL_PATH = ROOT / os.getenv("MODEL_PATH", "models/")
DATA_PATH = ROOT / "data"
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
DB_FILE = ROOT / "smartflow.db"
if DATABASE_URL.startswith("sqlite:///"):
    raw = DATABASE_URL.replace("sqlite:///", "")
    DB_FILE = Path(raw) if Path(raw).is_absolute() else ROOT / raw