from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from auth import (
    authenticate_user,
    create_access_token,
    create_refresh_token,
    generate_api_key,
    get_current_active_user,
    get_password_hash,
    get_user_by_username,
)
from config import ADMIN_PASSWORD, ADMIN_USERNAME
from database import User, get_db
from schemas import Token, UserInfo, UserLogin, UserRegister

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])


@router.post("/register", response_model=UserInfo)
async def register(user_data: UserRegister, db: Session = Depends(get_db)):
    # Check if user exists
    existing_user = get_user_by_username(db, user_data.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Check email
    existing_email = db.query(User).filter(User.email == user_data.email).first()
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create user
    hashed_password = get_password_hash(user_data.password)
    api_key = generate_api_key()
    
    new_user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hashed_password,
        api_key=api_key,
        is_active=True,
        is_admin=False,
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return new_user


@router.post("/login", response_model=Token)
async def login(user_data: UserLogin, db: Session = Depends(get_db)):
    user = authenticate_user(db, user_data.username, user_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(data={"sub": user.username})
    refresh_token = create_refresh_token(data={"sub": user.username})
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer"
    )


@router.post("/refresh", response_model=Token)
async def refresh(current_user: User = Depends(get_current_active_user)):
    """Refresh JWT access token"""
    access_token = create_access_token(data={"sub": current_user.username})
    
    return Token(
        access_token=access_token,
        token_type="bearer"
    )


@router.delete("/revoke")
async def revoke_api_key(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Revoke and regenerate API key"""
    current_user.api_key = generate_api_key()
    db.commit()
    
    return {
        "message": "API key revoked and regenerated",
        "new_api_key": current_user.api_key
    }


@router.get("/users/me", response_model=UserInfo)
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    return current_user


def init_admin_user(db: Session):
    """Initialize admin user if not exists"""
    admin = get_user_by_username(db, ADMIN_USERNAME)
    if not admin:
        hashed_password = get_password_hash(ADMIN_PASSWORD)
        api_key = generate_api_key()
        
        admin = User(
            username=ADMIN_USERNAME,
            email="admin@truthshield.com",
            hashed_password=hashed_password,
            api_key=api_key,
            is_active=True,
            is_admin=True,
            quota_limit=999999,
        )
        
        db.add(admin)
        db.commit()
        print(f"✓ Admin user created: {ADMIN_USERNAME} / {ADMIN_PASSWORD}")
        print(f"  Admin API Key: {api_key}")
