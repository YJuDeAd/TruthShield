from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ========== Authentication Schemas ==========
class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str
    password: str = Field(..., min_length=6)


class UserLogin(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"


class TokenData(BaseModel):
    username: Optional[str] = None


class UserInfo(BaseModel):
    id: int
    username: str
    email: Optional[str]
    api_key: str
    is_active: bool
    is_admin: bool
    request_count: int
    quota_limit: int
    created_at: datetime

    class Config:
        from_attributes = True


# ========== Detection Schemas ==========
class DetectionRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=10000)
    threshold: Optional[float] = Field(default=0.7, ge=0.0, le=1.0)


class MultimodalDetectionRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=10000)
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    threshold: Optional[float] = Field(default=0.7, ge=0.0, le=1.0)


class AutoDetectionRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=10000)
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    model_type: Optional[str] = None  # auto-detect if None
    threshold: Optional[float] = Field(default=0.7, ge=0.0, le=1.0)


class BatchDetectionRequest(BaseModel):
    items: list[AutoDetectionRequest] = Field(..., max_items=50)


class DetectionResponse(BaseModel):
    request_id: str
    model: str
    verdict: str  # "Real" or "Fake"
    confidence: float
    probabilities: dict[str, float]
    processing_time_ms: float
    timestamp: datetime


class BatchDetectionResponse(BaseModel):
    results: list[DetectionResponse]
    total_items: int
    total_processing_time_ms: float


# ========== Job Schemas ==========
class JobSubmitResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # pending, processing, completed, failed, cancelled
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_message: Optional[str]


class JobResultResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[DetectionResponse]
    error_message: Optional[str]


# ========== Explainability Schemas ==========
class ExplainRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=10000)


class ExplainResponse(BaseModel):
    request_id: str
    model: str
    verdict: str
    confidence: float
    explanation: dict  # Model-specific explanation data
    processing_time_ms: float


# ========== Model Management Schemas ==========
class ModelInfo(BaseModel):
    name: str
    version: str
    type: str
    loaded: bool
    accuracy: Optional[float]
    last_trained: Optional[str]


class ModelsListResponse(BaseModel):
    models: list[ModelInfo]


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict[str, bool]
    database_connected: bool
    version: str


class MetricsResponse(BaseModel):
    total_requests: int
    requests_per_model: dict[str, int]
    average_response_time_ms: float
    uptime_seconds: float


# ========== History Schemas ==========
class HistoryItem(BaseModel):
    request_id: str
    model_type: str
    verdict: str
    confidence: float
    created_at: datetime

    class Config:
        from_attributes = True


class HistoryResponse(BaseModel):
    items: list[HistoryItem]
    total: int
    page: int
    page_size: int


class HistoryDetailResponse(BaseModel):
    request_id: str
    model_type: str
    input_text: Optional[str]
    input_image_url: Optional[str]
    verdict: str
    confidence: float
    prob_real: float
    prob_fake: float
    processing_time_ms: float
    threshold_used: float
    created_at: datetime

    class Config:
        from_attributes = True


# ========== Feedback Schemas ==========
class FeedbackSubmit(BaseModel):
    request_id: str
    true_label: str = Field(..., pattern="^(Real|Fake)$")


class FeedbackResponse(BaseModel):
    message: str
    feedback_id: int


class FeedbackItem(BaseModel):
    id: int
    request_id: str
    predicted_label: str
    true_label: str
    model_type: str
    submitted_at: datetime
    processed: bool

    class Config:
        from_attributes = True


# ========== Stats Schemas ==========
class UserStatsResponse(BaseModel):
    total_requests: int
    requests_by_model: dict[str, int]
    quota_remaining: int
    quota_limit: int


class GlobalStatsResponse(BaseModel):
    total_users: int
    total_requests: int
    requests_by_model: dict[str, int]
    average_confidence: float
    verdicts_distribution: dict[str, int]
