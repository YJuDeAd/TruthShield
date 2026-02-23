from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from auth import get_current_active_user, get_current_admin_user
from database import DetectionHistory, User, get_db
from schemas import (
    GlobalStatsResponse,
    HistoryDetailResponse,
    HistoryResponse,
    UserStatsResponse,
)

router = APIRouter(prefix="/api/v1", tags=["History & Analytics"])


@router.get("/history", response_model=HistoryResponse)
async def get_user_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Get user's detection history (paginated)"""
    offset = (page - 1) * page_size
    
    query = db.query(DetectionHistory).filter(DetectionHistory.user_id == current_user.id)
    total = query.count()
    
    items = query.order_by(DetectionHistory.created_at.desc()).offset(offset).limit(page_size).all()
    
    return HistoryResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/history/{request_id}", response_model=HistoryDetailResponse)
async def get_history_detail(
    request_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Get detailed information for a specific detection request"""
    item = db.query(DetectionHistory).filter(
        DetectionHistory.request_id == request_id
    ).first()
    
    if not item:
        raise HTTPException(status_code=404, detail="Request not found")
    
    # Check ownership (non-admin users can only see their own history)
    if not current_user.is_admin and item.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return item


@router.get("/stats", response_model=UserStatsResponse)
async def get_user_stats(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Get user's statistics"""
    from collections import Counter
    
    # Get all user requests
    requests = db.query(DetectionHistory).filter(
        DetectionHistory.user_id == current_user.id
    ).all()
    
    total_requests = len(requests)
    
    # Count by model
    model_counts = Counter(r.model_type for r in requests)
    
    quota_remaining = current_user.quota_limit - current_user.request_count
    
    return UserStatsResponse(
        total_requests=total_requests,
        requests_by_model=dict(model_counts),
        quota_remaining=quota_remaining,
        quota_limit=current_user.quota_limit,
    )


@router.get("/stats/global", response_model=GlobalStatsResponse)
async def get_global_stats(
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db),
):
    """Get global platform statistics (admin only)"""
    from collections import Counter
    
    # Total users
    total_users = db.query(User).count()
    
    # All requests
    all_requests = db.query(DetectionHistory).all()
    total_requests = len(all_requests)
    
    # Requests by model
    model_counts = Counter(r.model_type for r in all_requests)
    
    # Average confidence
    if all_requests:
        avg_confidence = sum(r.confidence for r in all_requests) / len(all_requests)
    else:
        avg_confidence = 0.0
    
    # Verdicts distribution
    verdict_counts = Counter(r.verdict for r in all_requests)
    
    return GlobalStatsResponse(
        total_users=total_users,
        total_requests=total_requests,
        requests_by_model=dict(model_counts),
        average_confidence=avg_confidence,
        verdicts_distribution=dict(verdict_counts),
    )
