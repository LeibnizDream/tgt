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
import logging
import os
import shutil
from multiprocessing import Process
from pathlib import Path

from fastapi import APIRouter, Body, Form, HTTPException, Request, status
from fastapi.responses import FileResponse
from inference.processing_options import ProcessingOptions
from routers.inference.inference_workers import OneDriveWorker
from routers.auth import get_fresh_token
from routers.helpers.job_manager import JobManager, normalize_model_name
from sse_starlette.sse import EventSourceResponse

logger = logging.getLogger(__name__)
router = APIRouter()

MODELS_BASE = Path(__file__).resolve().parent.parent.parent / "models"


@router.post("/process")
async def process(
    request: Request,
    action: str = Form(...),
    language: str = Form(...),
    model: str | None = Form(None),
    instruction: str | None = Form(None),
    format: str | None = Form(None),
    base_dir: str | None = Form(None),
):
    """Process files from OneDrive."""
    job = JobManager.create()

    options = ProcessingOptions(
        language=language,
        action=action,
        format=format,
        instruction=instruction,
        model=normalize_model_name(model),
    )

    logger.info(f"Processing job {job.id} - action {action}, format {format}, model: {options.model}")

    if not language:
        job.queue.put("[ERROR] Missing language")
        return {"job_id": job.id}

    try:
        if not base_dir:
            raise HTTPException(status_code=400, detail="Missing base_dir")
        try:
            access_token = get_fresh_token()
        except RuntimeError as e:
            raise HTTPException(status_code=401, detail=str(e))
        job.token = access_token

        worker = OneDriveWorker(base_dir, options, access_token, job)
        proc = Process(target=worker.run, daemon=True)
        proc.start()
        job.process = proc
        return {"job_id": job.id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create worker for job {job.id}: {e}")
        job.queue.put(f"[ERROR] Failed to initialize processing: {str(e)}")
        return {"job_id": job.id}


@router.get("/{job_id}/stream")
async def stream(job_id: str):
    """Stream job progress events via Server-Sent Events."""
    job = JobManager.get(job_id)

    async def event_generator():
        loop = asyncio.get_event_loop()
        while True:
            try:
                message = await loop.run_in_executor(None, job.queue.get)
                if isinstance(message, str) and message.startswith("[ZIP PATH] "):
                    job.zip_path = message.replace("[ZIP PATH] ", "").strip()
                    continue
                yield {"data": message}
                if message == "[DONE ALL]":
                    break
            except Exception as e:
                logger.error(f"Error in event generator for job {job_id}: {e}")
                yield {"data": f"[ERROR] Stream error: {str(e)}"}
                break

    return EventSourceResponse(event_generator())


@router.get("/{job_id}/download")
async def download(job_id: str):
    """Download processed results as zip file."""
    job = JobManager.get(job_id)

    if not job.zip_path or not os.path.exists(job.zip_path):
        raise HTTPException(status_code=404, detail="Results not ready")

    JobManager.remove(job_id)
    return FileResponse(
        path=job.zip_path,
        media_type="application/zip",
        filename=f"{job_id}_results.zip",
    )


@router.post("/cancel")
async def cancel(payload: dict = Body(...)):
    """Cancel a running job."""
    job_id = payload.get("job_id")
    if not job_id:
        raise HTTPException(status_code=400, detail="Missing job_id")

    job = JobManager.get(job_id)

    try:
        job.queue.put("[CANCELLED]")
        job.queue.put("[DONE ALL]")
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
