"""
API Configuration
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
API_DIR = BASE_DIR / "api"

# Database
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{API_DIR}/truthshield.db")

# Security
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Rate limiting
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"

# Model paths
MODELS_DIR = BASE_DIR / "models"
NEWS_MODEL_PATH = MODELS_DIR / "news_model" / "roberta_news_best.pt"
SMS_MODEL_PATH = MODELS_DIR / "sms_model" / "sms_model.pt"
SMS_VOCAB_PATH = MODELS_DIR / "sms_model" / "vocab.pkl"
MULTIMODAL_MODEL_PATH = MODELS_DIR / "multimodal_model" / "best_model.pt"

# Model configs
NEWS_MODEL_NAME = "roberta-large"
SMS_MAX_LEN = 300
SMS_EMBED_SIZE = 128
SMS_HIDDEN_SIZE = 128
SMS_DROPOUT = 0.4
MULTIMODAL_MAX_LEN = 128
MULTIMODAL_IMAGE_SIZE = 224

# Detection settings
DEFAULT_THRESHOLD = 0.7

# Background job settings
JOB_TIMEOUT_SECONDS = 300
JOB_CLEANUP_HOURS = 24

# API settings
API_VERSION = "1.0.0"
API_TITLE = "TruthShield API"
API_DESCRIPTION = """
TruthShield API: AI-powered misinformation detection system.

Supports:
- Fake news article detection (RoBERTa)
- SMS/Email phishing detection (Bi-LSTM + CNN)
- Multimodal fake news detection (ResNet50 + BERT)
"""

# CORS
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# Admin credentials (for demo - in production use proper user management)
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
