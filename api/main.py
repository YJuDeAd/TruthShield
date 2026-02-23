from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import API_DESCRIPTION, API_TITLE, API_VERSION, CORS_ORIGINS
from database import SessionLocal, init_db
from ml_models import model_manager
from routers import (
    auth_router,
    detection,
    explainability,
    feedback,
    history,
    jobs,
    model_management,
    utilities,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    print("\n" + "=" * 60)
    print("TruthShield API Starting...")
    print("=" * 60)
    
    # Initialize database
    print("\nInitializing database...")
    init_db()
    
    # Create admin user
    db = SessionLocal()
    try:
        auth_router.init_admin_user(db)
    finally:
        db.close()
    
    # Load ML models
    print("\nLoading ML models...")
    model_manager.load_all_models()
    
    print("\n" + "=" * 60)
    print("✓ TruthShield API Ready!")
    print("=" * 60)
    print(f"\n📖 API Docs: http://localhost:8000/docs")
    print(f"🔑 Admin User: admin / admin123")
    print(f"🚀 Status: Models loaded - {model_manager.models_loaded}\n")
    
    yield
    
    # Shutdown
    print("\nShutting down TruthShield API...")


# Create FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(utilities.router)
app.include_router(auth_router.router)
app.include_router(detection.router)
app.include_router(jobs.router)
app.include_router(explainability.router)
app.include_router(model_management.router)
app.include_router(history.router)
app.include_router(feedback.router)


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
