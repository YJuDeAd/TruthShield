import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from auth import check_quota, get_current_active_user, increment_request_count
from database import DetectionHistory, User, get_db
from ml_models import load_image_from_base64, load_image_from_url, model_manager
from schemas import (
    AutoDetectionRequest,
    BatchDetectionRequest,
    BatchDetectionResponse,
    DetectionRequest,
    DetectionResponse,
    MultimodalDetectionRequest,
)

router = APIRouter(prefix="/api/v1/detect", tags=["Detection"])


@router.post("/news", response_model=DetectionResponse)
async def detect_news(
    request: DetectionRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Detect fake news articles using RoBERTa model"""
    check_quota(current_user, db)
    
    try:
        result = model_manager.predict_news(request.content, request.threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    request_id = str(uuid.uuid4())
    
    # Save to history
    history_entry = DetectionHistory(
        request_id=request_id,
        user_id=current_user.id,
        model_type="news",
        input_text=request.content[:1000],  # Truncate for storage
        verdict=result["verdict"],
        confidence=result["confidence"],
        prob_real=result["probabilities"]["Real"],
        prob_fake=result["probabilities"]["Fake"],
        processing_time_ms=result["processing_time_ms"],
        threshold_used=request.threshold,
    )
    db.add(history_entry)
    increment_request_count(current_user, db)
    db.commit()
    
    return DetectionResponse(
        request_id=request_id,
        model="news",
        verdict=result["verdict"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        processing_time_ms=result["processing_time_ms"],
        timestamp=datetime.utcnow(),
    )


@router.post("/sms", response_model=DetectionResponse)
async def detect_sms(
    request: DetectionRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Detect SMS/Email phishing using Bi-LSTM + CNN model"""
    check_quota(current_user, db)
    
    try:
        result = model_manager.predict_sms(request.content, request.threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    request_id = str(uuid.uuid4())
    
    # Save to history
    history_entry = DetectionHistory(
        request_id=request_id,
        user_id=current_user.id,
        model_type="sms",
        input_text=request.content[:1000],
        verdict=result["verdict"],
        confidence=result["confidence"],
        prob_real=result["probabilities"]["Real"],
        prob_fake=result["probabilities"]["Fake"],
        processing_time_ms=result["processing_time_ms"],
        threshold_used=request.threshold,
    )
    db.add(history_entry)
    increment_request_count(current_user, db)
    db.commit()
    
    return DetectionResponse(
        request_id=request_id,
        model="sms",
        verdict=result["verdict"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        processing_time_ms=result["processing_time_ms"],
        timestamp=datetime.utcnow(),
    )


@router.post("/multimodal", response_model=DetectionResponse)
async def detect_multimodal(
    request: MultimodalDetectionRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Detect fake news using text + image (ResNet50 + BERT)"""
    check_quota(current_user, db)
    
    if not request.image_url and not request.image_base64:
        raise HTTPException(status_code=400, detail="Either image_url or image_base64 required")
    
    try:
        # Load image
        if request.image_url:
            image = load_image_from_url(request.image_url)
        else:
            image = load_image_from_base64(request.image_base64)
        
        result = model_manager.predict_multimodal(request.content, image, request.threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    request_id = str(uuid.uuid4())
    
    # Save to history
    history_entry = DetectionHistory(
        request_id=request_id,
        user_id=current_user.id,
        model_type="multimodal",
        input_text=request.content[:1000],
        input_image_url=request.image_url,
        verdict=result["verdict"],
        confidence=result["confidence"],
        prob_real=result["probabilities"]["Real"],
        prob_fake=result["probabilities"]["Fake"],
        processing_time_ms=result["processing_time_ms"],
        threshold_used=request.threshold,
    )
    db.add(history_entry)
    increment_request_count(current_user, db)
    db.commit()
    
    return DetectionResponse(
        request_id=request_id,
        model="multimodal",
        verdict=result["verdict"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        processing_time_ms=result["processing_time_ms"],
        timestamp=datetime.utcnow(),
    )


@router.post("", response_model=DetectionResponse)
async def detect_auto(
    request: AutoDetectionRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Auto-route to appropriate model based on input"""
    check_quota(current_user, db)
    
    has_image = bool(request.image_url or request.image_base64)
    
    # Determine model type
    if request.model_type:
        model_type = request.model_type
    else:
        model_type = model_manager.auto_detect_model_type(request.content, has_image)
    
    try:
        if model_type == "multimodal":
            if not has_image:
                raise HTTPException(status_code=400, detail="Image required for multimodal model")
            
            if request.image_url:
                image = load_image_from_url(request.image_url)
            else:
                image = load_image_from_base64(request.image_base64)
            
            result = model_manager.predict_multimodal(request.content, image, request.threshold)
        
        elif model_type == "sms":
            result = model_manager.predict_sms(request.content, request.threshold)
        
        elif model_type == "news":
            result = model_manager.predict_news(request.content, request.threshold)
        
        else:
            raise HTTPException(status_code=400, detail=f"Invalid model_type: {model_type}")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    request_id = str(uuid.uuid4())
    
    # Save to history
    history_entry = DetectionHistory(
        request_id=request_id,
        user_id=current_user.id,
        model_type=model_type,
        input_text=request.content[:1000],
        input_image_url=request.image_url if has_image else None,
        verdict=result["verdict"],
        confidence=result["confidence"],
        prob_real=result["probabilities"]["Real"],
        prob_fake=result["probabilities"]["Fake"],
        processing_time_ms=result["processing_time_ms"],
        threshold_used=request.threshold,
    )
    db.add(history_entry)
    increment_request_count(current_user, db)
    db.commit()
    
    return DetectionResponse(
        request_id=request_id,
        model=model_type,
        verdict=result["verdict"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        processing_time_ms=result["processing_time_ms"],
        timestamp=datetime.utcnow(),
    )


@router.post("/batch", response_model=BatchDetectionResponse)
async def detect_batch(
    request: BatchDetectionRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Batch detection (up to 50 items)"""
    # Check quota for all items
    items_count = len(request.items)
    if current_user.request_count + items_count > current_user.quota_limit:
        raise HTTPException(
            status_code=429,
            detail=f"Batch would exceed quota. Remaining: {current_user.quota_limit - current_user.request_count}"
        )
    
    results = []
    total_time = 0
    
    for item in request.items:
        has_image = bool(item.image_url or item.image_base64)
        
        # Determine model type
        if item.model_type:
            model_type = item.model_type
        else:
            model_type = model_manager.auto_detect_model_type(item.content, has_image)
        
        try:
            if model_type == "multimodal":
                if not has_image:
                    continue  # Skip invalid items
                
                if item.image_url:
                    image = load_image_from_url(item.image_url)
                else:
                    image = load_image_from_base64(item.image_base64)
                
                result = model_manager.predict_multimodal(item.content, image, item.threshold)
            
            elif model_type == "sms":
                result = model_manager.predict_sms(item.content, item.threshold)
            
            elif model_type == "news":
                result = model_manager.predict_news(item.content, item.threshold)
            
            else:
                continue
            
            request_id = str(uuid.uuid4())
            
            # Save to history
            history_entry = DetectionHistory(
                request_id=request_id,
                user_id=current_user.id,
                model_type=model_type,
                input_text=item.content[:1000],
                input_image_url=item.image_url if has_image else None,
                verdict=result["verdict"],
                confidence=result["confidence"],
                prob_real=result["probabilities"]["Real"],
                prob_fake=result["probabilities"]["Fake"],
                processing_time_ms=result["processing_time_ms"],
                threshold_used=item.threshold,
            )
            db.add(history_entry)
            
            results.append(DetectionResponse(
                request_id=request_id,
                model=model_type,
                verdict=result["verdict"],
                confidence=result["confidence"],
                probabilities=result["probabilities"],
                processing_time_ms=result["processing_time_ms"],
                timestamp=datetime.utcnow(),
            ))
            
            total_time += result["processing_time_ms"]
        
        except Exception:
            continue  # Skip failed items
    
    # Increment request count for all processed items
    current_user.request_count += len(results)
    db.commit()
    
    return BatchDetectionResponse(
        results=results,
        total_items=len(results),
        total_processing_time_ms=total_time,
    )
