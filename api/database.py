from datetime import datetime
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from config import DATABASE_URL

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String, nullable=False)
    api_key = Column(String, unique=True, index=True)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    request_count = Column(Integer, default=0)
    quota_limit = Column(Integer, default=1000)  # requests per day


class DetectionHistory(Base):
    __tablename__ = "detection_history"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String, unique=True, index=True, nullable=False)
    user_id = Column(Integer, index=True)
    model_type = Column(String, nullable=False)  # news, sms, multimodal
    input_text = Column(Text)
    input_image_url = Column(String)
    verdict = Column(String, nullable=False)  # Real, Fake
    confidence = Column(Float, nullable=False)
    prob_real = Column(Float)
    prob_fake = Column(Float)
    processing_time_ms = Column(Float)
    threshold_used = Column(Float, default=0.7)
    created_at = Column(DateTime, default=datetime.utcnow)


class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, unique=True, index=True, nullable=False)
    user_id = Column(Integer, index=True)
    model_type = Column(String, nullable=False)
    status = Column(String, default="pending")  # pending, processing, completed, failed, cancelled
    input_data = Column(Text)
    result_data = Column(Text)
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String, index=True, nullable=False)
    user_id = Column(Integer, index=True)
    predicted_label = Column(String, nullable=False)
    true_label = Column(String, nullable=False)
    model_type = Column(String, nullable=False)
    submitted_at = Column(DateTime, default=datetime.utcnow)
    processed = Column(Boolean, default=False)


class APIMetrics(Base):
    __tablename__ = "api_metrics"

    id = Column(Integer, primary_key=True, index=True)
    endpoint = Column(String, nullable=False)
    user_id = Column(Integer)
    status_code = Column(Integer)
    response_time_ms = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)


# Dependency for getting database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)
