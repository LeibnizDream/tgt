"""
Inference API router for the TGT backend.

- ``POST /process``                  – Start a processing job. Returns a ``job_id``.
- ``GET /{job_id}/stream``           – SSE stream of progress messages.
- ``GET /{job_id}/download``         – Download the finished output ZIP archive.
- ``POST /cancel``                   – Signal a running job to stop.
- ``GET /models/{task}``             – List custom models available for a task.
- ``DELETE /models/{task}/{model}``  – Delete a custom model directory.
"""
import asyncio
import json
import logging
import shutil
from multiprocessing import Process
from pathlib import Path

from fastapi import APIRouter, Body, Form, HTTPException, Request, status
from inference.processing_options import ProcessingOptions
from routers.inference.onedrive_worker import OneDriveWorker
from routers.auth import get_fresh_token
from routers.helpers.job_manager import JobManager, normalize_model_name
from sse_starlette.sse import EventSourceResponse

logger = logging.getLogger(__name__)
router = APIRouter()

MODELS_BASE = Path(__file__).resolve().parent.parent.parent / "models"


@router.post("/process")
async def process(
    request: Request,
    base_dir: str = Form(...),
    action: str = Form(...),
    language: str = Form(...),
    model: str | None = Form(None),
    instruction: str | None = Form(None),
    format: str | None = Form(None),
):
    """Process files from OneDrive."""

    options = ProcessingOptions(
        language=language,
        action=action,
        format=format,
        instruction=instruction,
        model=normalize_model_name(model),
    )

    try:
        access_token = get_fresh_token()
        job = JobManager.create()
        job.token = access_token


        worker = OneDriveWorker(base_dir, options, access_token, job.publisher)
        process = Process(target=worker.run, daemon=True)
        process.start()
        job.process = process

        return {"job_id": job.id}

    except Exception as e:
        logger.error(f"Failed to create worker for job {job.id}: {e}")
        job.publisher.inform(f"Failed to initialize processing: {str(e)}", level="error")
        job.publisher.done()
        return {"job_id": job.id}


@router.get("/{job_id}/stream")
async def stream(job_id: str):
    """Stream job progress events via Server-Sent Events."""
    job = JobManager.get(job_id)

    async def event_generator():
        loop = asyncio.get_event_loop()
        try:
            while True:
                msg = await loop.run_in_executor(None, job.publisher.get_message)
                if isinstance(msg, dict) and msg.get("type") == "zip_path":
                    job.zip_path = msg["path"]
                    continue
                yield {"data": json.dumps(msg)}
                if isinstance(msg, dict) and msg.get("type") in ("done", "cancelled", "error"):
                    break
        except Exception as e:
            logger.error(f"Error in event generator for job {job_id}: {e}")
            yield {"data": json.dumps({"type": "error", "message": f"Stream error: {str(e)}"})}
        finally:
            JobManager.remove(job_id)

    return EventSourceResponse(event_generator())


@router.post("/cancel")
async def cancel(payload: dict = Body(...)):
    """Cancel a running job."""
    job_id = payload.get("job_id")
    if not job_id:
        raise HTTPException(status_code=400, detail="Missing job_id")

    job = JobManager.get(job_id)

    try:
        job.cancel_event.set()  # signal worker to stop gracefully
        job.publisher.cancelled()  # fallback if process is killed before finally runs
        if job.process and job.process.is_alive():
            job.process.terminate()
            job.process.join(timeout=5)
            if job.process.is_alive():
                logger.warning(f"Force killing process for job {job_id}")
                job.process.kill()
    except Exception as e:
        logger.error(f"Error cancelling job {job_id}: {e}")
    finally:
        JobManager.remove(job_id)

    return {"status": "cancelled"}


@router.get("/models/{task}")
async def list_models(task: str):
    """List available models for a given task."""
    dir_path = MODELS_BASE / task
    if not dir_path.is_dir():
        return {"models": []}
    try:
        return {"models": sorted(d.name for d in dir_path.iterdir() if d.is_dir())}
    except Exception as e:
        logger.error(f"Error listing models in {dir_path}: {e}")
        return {"models": []}


@router.delete(
    "/models/{task}/{model_name}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a custom model",
)
async def delete_model(task: str, model_name: str):
    """Remove a custom model directory. Returns 204 on success, 404 if not found."""
    dir_path = MODELS_BASE / task / model_name
    if not dir_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    if not dir_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {dir_path}")
    try:
        shutil.rmtree(dir_path)
    except Exception as e:
        logger.error(f"Failed to delete model at {dir_path}: {e}")
        raise HTTPException(status_code=500, detail="Unable to delete model directory")
