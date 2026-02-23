import uuid
from datetime import datetime

import numpy as np
import torch
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from auth import get_current_active_user
from database import User, get_db
from ml_models import model_manager
from schemas import ExplainRequest, ExplainResponse

router = APIRouter(prefix="/api/v1/explain", tags=["Explainability"])


@router.post("/news", response_model=ExplainResponse)
async def explain_news(
    request: ExplainRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Get attention weights and top influential tokens for news prediction"""
    if not model_manager.models_loaded["news"]:
        raise HTTPException(status_code=503, detail="News model not loaded")
    
    import time
    start = time.time()
    
    try:
        # Tokenize and get prediction
        encoding = model_manager.news_tokenizer(
            request.content,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        encoding = {k: v.to(model_manager.device) for k, v in encoding.items()}
        
        with torch.no_grad():
            outputs = model_manager.news_model(**encoding, output_attentions=True)
            probs = torch.softmax(outputs.logits, dim=1)[0]
            prob_fake = probs[1].item()
            verdict = "Fake" if prob_fake >= 0.7 else "Real"
            confidence = max(probs[0].item(), prob_fake)
            
            # Get attention weights from last layer
            attentions = outputs.attentions[-1][0]  # (num_heads, seq_len, seq_len)
            # Average across heads and take attention to CLS token
            avg_attention = attentions.mean(dim=0)[0].cpu().numpy()  # (seq_len,)
            
            # Get tokens
            tokens = model_manager.news_tokenizer.convert_ids_to_tokens(
                encoding["input_ids"][0]
            )
            
            # Top 10 most attended tokens
            top_indices = np.argsort(avg_attention)[-10:][::-1]
            top_tokens = [
                {"token": tokens[i], "attention": float(avg_attention[i])}
                for i in top_indices
                if tokens[i] not in ["<s>", "</s>", "<pad>"]
            ]
        
        processing_time_ms = (time.time() - start) * 1000
        
        explanation = {
            "top_attended_tokens": top_tokens[:10],
            "attention_method": "RoBERTa self-attention (last layer, CLS token)",
            "note": "Higher attention indicates tokens more influential in decision"
        }
        
        return ExplainResponse(
            request_id=str(uuid.uuid4()),
            model="news",
            verdict=verdict,
            confidence=confidence,
            explanation=explanation,
            processing_time_ms=processing_time_ms,
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@router.post("/sms", response_model=ExplainResponse)
async def explain_sms(
    request: ExplainRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Get feature importance for SMS prediction"""
    if not model_manager.models_loaded["sms"]:
        raise HTTPException(status_code=503, detail="SMS model not loaded")
    
    import time
    start = time.time()
    
    try:
        # Tokenize
        tokens = request.content.lower().split()
        seq = [model_manager.sms_vocab.get(tok, 1) for tok in tokens]
        
        # Get prediction
        from config import SMS_MAX_LEN
        if len(seq) < SMS_MAX_LEN:
            seq.extend([0] * (SMS_MAX_LEN - len(seq)))
        else:
            seq = seq[:SMS_MAX_LEN]
        
        tensor = torch.tensor([seq], dtype=torch.long).to(model_manager.device)
        
        with torch.no_grad():
            prob_spam = model_manager.sms_model(tensor).item()
            verdict = "Fake" if prob_spam >= 0.7 else "Real"
            confidence = max(prob_spam, 1.0 - prob_spam)
        
        # Simple feature importance: count known spam/ham words
        spam_keywords = ["free", "win", "prize", "click", "urgent", "congratulations", 
                        "offer", "limited", "cash", "www", "http"]
        
        found_spam_keywords = [
            kw for kw in spam_keywords
            if kw in request.content.lower()
        ]
        
        processing_time_ms = (time.time() - start) * 1000
        
        explanation = {
            "spam_keywords_found": found_spam_keywords,
            "total_tokens": len(tokens),
            "unknown_tokens": sum(1 for tok in tokens if tok.lower() not in model_manager.sms_vocab),
            "note": "Presence of spam keywords and URL patterns increase spam probability"
        }
        
        return ExplainResponse(
            request_id=str(uuid.uuid4()),
            model="sms",
            verdict=verdict,
            confidence=confidence,
            explanation=explanation,
            processing_time_ms=processing_time_ms,
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@router.post("/multimodal", response_model=ExplainResponse)
async def explain_multimodal(
    request: ExplainRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Get cross-modal attention for multimodal prediction"""
    # Note: This is a simplified version. Full implementation would require
    # modifying the model to output attention weights
    
    raise HTTPException(
        status_code=501,
        detail="Multimodal explainability requires model modification to expose attention weights. Coming soon."
    )
