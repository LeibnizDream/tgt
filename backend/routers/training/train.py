"""
Training API router for the TGT backend.

- ``POST /process``        – Start a training job from OneDrive. Returns a ``job_id``.
- ``GET /{job_id}/stream`` – SSE stream of training progress.
- ``POST /cancel``         – Signal a running training job to stop.
"""
import asyncio
import json
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
        job.publisher.inform("Missing language", level="error")
        return {"job_id": job.id}

    if not base_dir:
        raise HTTPException(status_code=400, detail="Missing base_dir")
    try:
        user_id = request.session.get("user_id")
        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Not authenticated"
            )
        access_token = get_fresh_token(user_id)
    except RuntimeError as e:
        raise HTTPException(status_code=401, detail=str(e))

    job.token = access_token
    worker = OneDriveWorker(base_dir, language, action, study, access_token, job.publisher)
    proc = Process(target=worker.run, daemon=True)
    proc.start()
    job.process = proc
    return {"job_id": job.id}


@router.get("/{job_id}/stream")
async def stream(job_id: str):
    """Stream training progress as Server-Sent Events until done."""
    job = JobManager.get(job_id)

    async def event_generator():
        loop = asyncio.get_event_loop()
        try:
            while True:
                msg = await loop.run_in_executor(None, job.publisher.get_message)
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
    """Terminate a running training job and remove it from the job registry."""
    job_id = payload.get("job_id")
    if not job_id:
        raise HTTPException(status_code=400, detail="Missing job_id")
    job = JobManager.get(job_id)
    job.cancel_event.set()
    job.publisher.cancelled()  # fallback if process is killed before finally runs
    if job.process and job.process.is_alive():
        job.process.terminate()
    JobManager.remove(job_id)
    return {"status": "cancelled"}
