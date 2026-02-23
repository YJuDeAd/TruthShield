import json
import uuid
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from auth import get_current_active_user
from database import Job, User, get_db
from ml_models import load_image_from_base64, load_image_from_url, model_manager
from schemas import (
    AutoDetectionRequest,
    JobResultResponse,
    JobStatusResponse,
    JobSubmitResponse,
)

router = APIRouter(prefix="/api/v1/jobs", tags=["Async Jobs"])


def process_detection_job(job_id: str, request_data: dict, db_url: str):
    """Background task to process detection job"""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    engine = create_engine(db_url)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    
    try:
        job = db.query(Job).filter(Job.job_id == job_id).first()
        if not job:
            return
        
        job.status = "processing"
        job.started_at = datetime.utcnow()
        db.commit()
        
        # Parse request
        content = request_data["content"]
        threshold = request_data.get("threshold", 0.7)
        model_type = request_data.get("model_type")
        image_url = request_data.get("image_url")
        image_base64 = request_data.get("image_base64")
        
        has_image = bool(image_url or image_base64)
        
        # Auto-detect model if not specified
        if not model_type:
            model_type = model_manager.auto_detect_model_type(content, has_image)
        
        # Run prediction
        if model_type == "multimodal":
            if image_url:
                image = load_image_from_url(image_url)
            else:
                image = load_image_from_base64(image_base64)
            result = model_manager.predict_multimodal(content, image, threshold)
        elif model_type == "sms":
            result = model_manager.predict_sms(content, threshold)
        elif model_type == "news":
            result = model_manager.predict_news(content, threshold)
        else:
            raise ValueError(f"Invalid model_type: {model_type}")
        
        # Store result
        result_data = {
            "request_id": str(uuid.uuid4()),
            "model": model_type,
            "verdict": result["verdict"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "processing_time_ms": result["processing_time_ms"],
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        job.result_data = json.dumps(result_data)
        job.status = "completed"
        job.completed_at = datetime.utcnow()
        db.commit()
    
    except Exception as e:
        job.status = "failed"
        job.error_message = str(e)
        job.completed_at = datetime.utcnow()
        db.commit()
    
    finally:
        db.close()


@router.post("/detect", response_model=JobSubmitResponse)
async def submit_detection_job(
    request: AutoDetectionRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Submit async detection job"""
    job_id = str(uuid.uuid4())
    
    request_data = {
        "content": request.content,
        "threshold": request.threshold,
        "model_type": request.model_type,
        "image_url": request.image_url,
        "image_base64": request.image_base64,
    }
    
    job = Job(
        job_id=job_id,
        user_id=current_user.id,
        model_type=request.model_type or "auto",
        status="pending",
        input_data=json.dumps(request_data),
    )
    
    db.add(job)
    db.commit()
    
    # Get database URL for background task
    from config import DATABASE_URL
    
    # Schedule background task
    background_tasks.add_task(process_detection_job, job_id, request_data, DATABASE_URL)
    
    return JobSubmitResponse(
        job_id=job_id,
        status="pending",
        message="Job submitted successfully"
    )


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Get job status"""
    job = db.query(Job).filter(Job.job_id == job_id).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Check ownership (non-admin users can only see their own jobs)
    if not current_user.is_admin and job.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error_message=job.error_message,
    )


@router.get("/{job_id}/result", response_model=JobResultResponse)
async def get_job_result(
    job_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Get job result"""
    job = db.query(Job).filter(Job.job_id == job_id).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Check ownership
    if not current_user.is_admin and job.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    result_dict = None
    if job.status == "completed" and job.result_data:
        result_dict = json.loads(job.result_data)
    
    return JobResultResponse(
        job_id=job.job_id,
        status=job.status,
        result=result_dict,
        error_message=job.error_message,
    )


@router.delete("/{job_id}")
async def cancel_job(
    job_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Cancel a pending job"""
    job = db.query(Job).filter(Job.job_id == job_id).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Check ownership
    if not current_user.is_admin and job.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    if job.status not in ["pending", "processing"]:
        raise HTTPException(status_code=400, detail="Job cannot be cancelled")
    
    job.status = "cancelled"
    job.completed_at = datetime.utcnow()
    db.commit()
    
    return {"message": "Job cancelled successfully"}
