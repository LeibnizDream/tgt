"""
Main FastAPI application entry point for the TGT (Transcription, Glossing, Translation) backend.

This module creates the FastAPI application, registers all API routers under
the ``/api`` prefix, and serves the compiled frontend SPA (Single Page Application)
as a catch-all static file handler.

Routers:
    - ``/api/auth``      – Microsoft OneDrive OAuth2 authentication
    - ``/api/inference`` – Data processing (transcription, translation, glossing, etc.)
    - ``/api/train``     – Model training pipeline
"""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from routers.auth import router as auth_router
from routers.inference.inference import router as inference_router
from routers.training.train import router as train_router

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"

app = FastAPI()

app.include_router(auth_router, prefix="/api/auth")
app.include_router(inference_router, prefix="/api/inference")
app.include_router(train_router, prefix="/api/train")


@app.get("/{full_path:path}", include_in_schema=False)
async def serve_spa(full_path: str):
    """
    Catch-all route that serves the compiled frontend SPA.

    If the requested path matches a real file inside the ``frontend/dist``
    directory the file is returned directly (CSS, JS, images, etc.).
    Otherwise ``index.html`` is returned so that client-side routing works.

    Raises:
        HTTPException(404): When neither the file nor ``index.html`` exist
            (i.e. the frontend has not been built yet).
    """
    candidate = FRONTEND_DIST / full_path
    index = FRONTEND_DIST / "index.html"

    if candidate.is_file():
        return FileResponse(candidate)

    if index.is_file():
        return FileResponse(index)

    raise HTTPException(status_code=404, detail="Frontend build not found")
