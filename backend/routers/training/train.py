"""
Training API router for the TGT backend.

- ``POST /process``        – Start a training job from OneDrive. Returns a ``job_id``.
- ``GET /{job_id}/stream`` – SSE stream of training progress.
- ``POST /cancel``         – Signal a running training job to stop.
"""
import asyncio
import logging
from multiprocessing import Process

from fastapi import APIRouter, Body, Form, HTTPException, Request
from routers.auth import get_fresh_token
from routers.helpers.job_manager import JobManager
from routers.training.train_workers import OneDriveWorker
from sse_starlette.sse import EventSourceResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/process")
async def process(
    request: Request,
    base_dir: str | None = Form(None),
    study: str | None = Form(None),
    action: str = Form(...),
    language: str = Form(...),
):
    """Create and start a training job from OneDrive. Returns the job_id."""
    job = JobManager.create()

    if not language:
        job.queue.put("[ERROR] Missing language")
        return {"job_id": job.id}

    if not base_dir:
        raise HTTPException(status_code=400, detail="Missing base_dir")
    try:
        access_token = get_fresh_token()
    except RuntimeError as e:
        raise HTTPException(status_code=401, detail=str(e))

    job.token = access_token
    worker = OneDriveWorker(base_dir, language, action, study, access_token, job)
    proc = Process(target=worker.run, daemon=True)
    proc.start()
    job.process = proc
    return {"job_id": job.id}


@router.get("/{job_id}/stream")
async def stream(job_id: str):
    """Stream training progress as Server-Sent Events until ``[DONE ALL]`` is received."""
    job = JobManager.get(job_id)

    async def event_generator():
        loop = asyncio.get_event_loop()
        while True:
            try:
                message = await loop.run_in_executor(None, job.queue.get)
                yield {"data": message}
                if message == "[DONE ALL]":
                    break
            except Exception as e:
                logger.error(f"Error in event generator for job {job_id}: {e}")
                yield {"data": f"[ERROR] Stream error: {str(e)}"}
                break

    return EventSourceResponse(event_generator())


@router.post("/cancel")
async def cancel(payload: dict = Body(...)):
    """Terminate a running training job and remove it from the job registry."""
    job_id = payload.get("job_id")
    if not job_id:
        raise HTTPException(status_code=400, detail="Missing job_id")
    job = JobManager.get(job_id)
    job.queue.put("[CANCELLED]")
    job.queue.put("[DONE ALL]")
    if job.process and job.process.is_alive():
        job.process.terminate()
    JobManager.remove(job_id)
    return {"status": "cancelled"}
