from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from auth import get_current_active_user, get_current_admin_user
from config import API_VERSION
from database import APIMetrics, User, get_db
from ml_models import model_manager
from schemas import HealthResponse, MetricsResponse, ModelInfo, ModelsListResponse

router = APIRouter(prefix="/api/v1/models", tags=["Model Management"])

# Track startup time for uptime calculation
SERVER_START_TIME = datetime.utcnow()


@router.get("", response_model=ModelsListResponse)
async def list_models(current_user: User = Depends(get_current_active_user)):
    """List all available models and their status"""
    models = [
        ModelInfo(
            name="news",
            version="1.0",
            type="RoBERTa-large",
            loaded=model_manager.models_loaded["news"],
            accuracy=0.947,  # Update with actual metrics
            last_trained="2026-02-15",
        ),
        ModelInfo(
            name="sms",
            version="1.0",
            type="Bi-LSTM + CNN",
            loaded=model_manager.models_loaded["sms"],
            accuracy=0.983,
            last_trained="2026-02-15",
        ),
        ModelInfo(
            name="multimodal",
            version="1.0",
            type="ResNet50 + BERT",
            loaded=model_manager.models_loaded["multimodal"],
            accuracy=0.891,
            last_trained="2026-02-15",
        ),
    ]
    
    return ModelsListResponse(models=models)


@router.get("/health", response_model=HealthResponse, tags=["Utilities"])
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint"""
    # Check database
    db_connected = True
    try:
        db.execute("SELECT 1")
    except Exception:
        db_connected = False
    
    all_models_loaded = all(model_manager.models_loaded.values())
    status = "healthy" if (db_connected and all_models_loaded) else "degraded"
    
    return HealthResponse(
        status=status,
        models_loaded=model_manager.models_loaded,
        database_connected=db_connected,
        version=API_VERSION,
    )


@router.post("/reload")
async def reload_models(current_user: User = Depends(get_current_admin_user)):
    """Hot-reload all models (admin only)"""
    try:
        model_manager.load_all_models()
        return {
            "message": "Models reloaded successfully",
            "status": model_manager.models_loaded
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {str(e)}")


@router.get("/{model_name}", response_model=ModelInfo)
async def get_model_info(
    model_name: str,
    current_user: User = Depends(get_current_active_user),
):
    """Get detailed information about a specific model"""
    model_info_map = {
        "news": ModelInfo(
            name="news",
            version="1.0",
            type="RoBERTa-large",
            loaded=model_manager.models_loaded["news"],
            accuracy=0.947,
            last_trained="2026-02-15",
        ),
        "sms": ModelInfo(
            name="sms",
            version="1.0",
            type="Bi-LSTM + CNN",
            loaded=model_manager.models_loaded["sms"],
            accuracy=0.983,
            last_trained="2026-02-15",
        ),
        "multimodal": ModelInfo(
            name="multimodal",
            version="1.0",
            type="ResNet50 + BERT",
            loaded=model_manager.models_loaded["multimodal"],
            accuracy=0.891,
            last_trained="2026-02-15",
        ),
    }
    
    if model_name not in model_info_map:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    return model_info_map[model_name]


@router.get("/metrics", response_model=MetricsResponse, tags=["Utilities"])
async def get_metrics(
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db),
):
    """Get API metrics (admin only)"""
    from collections import Counter
    from database import DetectionHistory
    
    # Total requests
    total_requests = db.query(DetectionHistory).count()
    
    # Requests per model
    results = db.query(DetectionHistory.model_type).all()
    model_counts = Counter(r[0] for r in results)
    
    # Average response time
    avg_time_result = db.query(DetectionHistory).all()
    if avg_time_result:
        avg_time = sum(r.processing_time_ms for r in avg_time_result) / len(avg_time_result)
    else:
        avg_time = 0.0
    
    # Uptime
    uptime = (datetime.utcnow() - SERVER_START_TIME).total_seconds()
    
    return MetricsResponse(
        total_requests=total_requests,
        requests_per_model=dict(model_counts),
        average_response_time_ms=avg_time,
        uptime_seconds=uptime,
    )
