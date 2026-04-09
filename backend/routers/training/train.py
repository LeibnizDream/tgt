"""
Training API router for the TGT backend.

Exposes the following REST endpoints under ``/api/train``:

- ``POST /process``        – Start a training job for a given language and
  action.  Accepts either a ZIP file upload or a OneDrive share link.
  Returns a ``job_id`` UUID.
- ``GET /{job_id}/stream`` – Server-Sent Events stream that forwards training
  progress messages (preprocessing → training metrics) to the browser.
- ``POST /cancel``         – Signal a running training job to stop.
"""
import os
import uuid
import shutil
import tempfile
import asyncio
import logging
from pathlib import Path
from zipfile import ZipFile
from fastapi import APIRouter, HTTPException, Request, Form, UploadFile, File, Body, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from sse_starlette.sse import EventSourceResponse
from multiprocessing import Process, Queue, Event
from routers.training.train_workers import OneDriveWorker
from routers.helpers.job_manager import JobManager
from routers.auth import get_fresh_token

logger = logging.getLogger(__name__)
router = APIRouter()

MODELS_BASE = Path(__file__).resolve().parent.parent / "models"

async def run_worker(process_fn):
    """Spawn a daemon worker process that calls *process_fn* and return it."""
    proc = Process(target=process_fn, daemon=True)
    proc.start()
    return proc


@router.post("/process")
async def process(
    request: Request,
    base_dir: str | None = Form(None),
    study: str | None = Form(None),
    action: str = Form(...),
    language: str = Form(...),
    zipfile: UploadFile | None = File(None),
):
    """
    Create and start a training job.

    Accepts data either as an uploaded ZIP file or as a OneDrive share link
    (*base_dir*).  Returns the ``job_id`` that the client should use to
    stream progress via ``GET /{job_id}/stream``.
    """
    job = JobManager.create()

    if not language:
        job.queue.put("[ERROR] Missing language")
        return {"job_id": job.id}

    if zipfile:
        tmp_dir = tempfile.mkdtemp()
        archive_path = Path(tmp_dir) / "upload.zip"
        contents = await zipfile.read()
        archive_path.write_bytes(contents)
        with ZipFile(archive_path, 'r') as archive:
            archive.extractall(tmp_dir)
        archive_path.unlink()

        #TODO: Handle the case where multiple files are uploaded
    else:
        if not base_dir:
            raise HTTPException(status_code=400, detail="Missing base_dir for online processing")
        try:
            access_token = get_fresh_token()
        except RuntimeError as e:
            raise HTTPException(status_code=401, detail=str(e))
        job.token = access_token
        worker_fn = OneDriveWorker(base_dir, language, action, study, access_token, job)

    job.process = await run_worker(worker_fn.run)
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

@router.post("/cancel")
async def cancel(payload: dict = Body(...)):
    """Terminate a running training job and remove it from the job registry."""
    job_id = payload.get("job_id")
    job = JobManager.get(job_id)
    job.queue.put("[CANCELLED]")
    job.queue.put("[DONE ALL]")
    if job.process and job.process.is_alive():
        job.process.terminate()
    JobManager.remove(job_id)
    return {"status": "cancelled"}
