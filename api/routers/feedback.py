from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from auth import get_current_active_user, get_current_admin_user
from database import DetectionHistory, Feedback, User, get_db
from schemas import FeedbackItem, FeedbackResponse, FeedbackSubmit

router = APIRouter(prefix="/api/v1/feedback", tags=["Feedback & Retraining"])


@router.post("", response_model=FeedbackResponse)
async def submit_feedback(
    feedback: FeedbackSubmit,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Submit correction feedback for a prediction"""
    # Verify request exists and belongs to user
    detection = db.query(DetectionHistory).filter(
        DetectionHistory.request_id == feedback.request_id
    ).first()
    
    if not detection:
        raise HTTPException(status_code=404, detail="Request not found")
    
    # Check ownership
    if not current_user.is_admin and detection.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Check if feedback already exists
    existing = db.query(Feedback).filter(
        Feedback.request_id == feedback.request_id
    ).first()
    
    if existing:
        raise HTTPException(status_code=400, detail="Feedback already submitted for this request")
    
    # Create feedback entry
    feedback_entry = Feedback(
        request_id=feedback.request_id,
        user_id=current_user.id,
        predicted_label=detection.verdict,
        true_label=feedback.true_label,
        model_type=detection.model_type,
        processed=False,
    )
    
    db.add(feedback_entry)
    db.commit()
    db.refresh(feedback_entry)
    
    return FeedbackResponse(
        message="Feedback submitted successfully. Thank you for helping improve TruthShield!",
        feedback_id=feedback_entry.id,
    )


@router.get("", response_model=list[FeedbackItem])
async def get_feedback_queue(
    processed: bool = Query(False),
    limit: int = Query(100, ge=1, le=1000),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db),
):
    """Get feedback queue for retraining (admin only)"""
    query = db.query(Feedback).filter(Feedback.processed == processed)
    items = query.order_by(Feedback.submitted_at.desc()).limit(limit).all()
    
    return items


@router.post("/retrain/trigger")
async def trigger_retraining(
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db),
):
    """Trigger model retraining with feedback data (admin only)"""
    # Count unprocessed feedback
    unprocessed = db.query(Feedback).filter(Feedback.processed == False).count()
    
    if unprocessed < 100:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient feedback for retraining. Need 100, have {unprocessed}"
        )
    
    # In production, this would:
    # 1. Export feedback to training data format
    # 2. Schedule retraining job on ML infrastructure
    # 3. Validate new model performance
    # 4. Deploy if improvement threshold met
    
    return {
        "message": "Retraining job scheduled",
        "feedback_items": unprocessed,
        "note": "This is a demo endpoint. Production implementation would trigger actual retraining pipeline."
    }


@router.get("/retrain/status")
async def get_retraining_status(
    current_user: User = Depends(get_current_admin_user),
):
    """Check retraining job status (admin only)"""
    # In production, this would query the ML training job status
    
    return {
        "status": "No active retraining jobs",
        "last_retrain": "2026-02-15",
        "note": "This is a demo endpoint. Production implementation would track actual jobs."
    }
