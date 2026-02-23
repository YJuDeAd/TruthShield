from fastapi import APIRouter
from fastapi.responses import RedirectResponse

from config import API_DESCRIPTION, API_TITLE, API_VERSION

router = APIRouter(tags=["Utilities"])


@router.get("/")
async def root():
    """Redirect to API documentation"""
    return RedirectResponse(url="/docs")


@router.get("/api/v1/")
async def api_root():
    """API root information"""
    return RedirectResponse(url="/docs")


@router.get("/api/v1/version")
async def get_version():
    """Get API version information"""
    return {
        "api_title": API_TITLE,
        "version": API_VERSION,
        "description": API_DESCRIPTION,
        "endpoints": {
            "documentation": "/docs",
            "openapi_schema": "/openapi.json",
            "health": "/api/v1/models/health",
        }
    }
