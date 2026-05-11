"""
Job lifecycle management helpers for the TGT backend.

This module provides three service classes used by both the inference and
training routers:

- :class:`Job`              – Lightweight container that tracks a single
  background processing job (ID, IPC queue, cancel event, temp paths, etc.).
- :class:`JobManager`       – Class-level registry that creates, retrieves,
  and removes ``Job`` instances by their UUID.
- :class:`ProcessingService`– Factory helpers that normalize model names,
  extract uploaded ZIP archives, and instantiate the correct worker type
  (``ZipWorker`` or ``OneDriveWorker``).
- :class:`JobCleanupService`– Removes temporary files and directories once a
  job's results have been downloaded or cancelled.
"""
import logging
import os
import shutil
import tempfile
import uuid
from multiprocessing import Event, Process, Queue
from pathlib import Path
from zipfile import ZipFile

from fastapi import APIRouter, HTTPException, UploadFile
from routers.inference.inference_workers import OneDriveWorker, ZipWorker

logger = logging.getLogger(__name__)
router = APIRouter()

MODELS_BASE = Path(__file__).resolve().parent.parent / "models"
DEFAULT_MODEL = "Default"

class Job:
    """
    Container for a single background processing job.

    Attributes:
        id (str): UUID identifying this job.
        queue (Queue): Multiprocessing queue used to stream status messages
            to the SSE endpoint.
        cancel_event (Event): Multiprocessing event that signals the worker
            process to abort.
        base_dir (str | None): Temporary directory holding the input files.
        zip_path (str | None): Path to the output ZIP archive once created.
        token (str | None): OAuth2 access token for OneDrive operations.
        process (Process | None): The spawned worker process.
    """
    def __init__(self, job_id: str):
        self.id = job_id
        self.queue: Queue = Queue()
        self.cancel_event: Event = Event()
        self.base_dir: str | None = None
        self.zip_path: str | None = None
        self.token: str | None = None
        self.process: Process | None = None

class JobManager:
    """
    In-process registry of active :class:`Job` objects, keyed by UUID.

    All methods are class-methods so no instance is needed.
    """
    _jobs: dict[str, Job] = {}

    @classmethod
    def create(cls) -> Job:
        """Create a new :class:`Job` with a fresh UUID and register it."""
        job_id = str(uuid.uuid4())
        job = Job(job_id)
        cls._jobs[job_id] = job
        return job

    @classmethod
    def get(cls, job_id: str) -> Job:
        """Return the :class:`Job` for *job_id*, or raise HTTP 404 if not found."""
        job = cls._jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
        return job

    @classmethod
    def remove(cls, job_id: str):
        """Remove *job_id* from the registry (no-op if already absent)."""
        cls._jobs.pop(job_id, None)


class ProcessingService:
    """Service class to handle processing logic."""
    
    @staticmethod
    async def create_worker_process(process_fn) -> Process:
        """Create and start a worker process."""
        proc = Process(target=process_fn, daemon=True)
        proc.start()
        return proc
    
    @staticmethod
    def normalize_model_name(model: str | None) -> str | None:
        """Normalize model name, converting 'Default' to None."""
        return None if model == DEFAULT_MODEL else model
    
    @staticmethod
    async def extract_zipfile(zipfile: UploadFile) -> str:
        """Extract uploaded zip file to temporary directory."""
        tmp_dir = tempfile.mkdtemp()
        archive_path = Path(tmp_dir) / "upload.zip"
        
        try:
            contents = await zipfile.read()
            archive_path.write_bytes(contents)
            
            with ZipFile(archive_path, 'r') as archive:
                archive.extractall(tmp_dir)
            
            archive_path.unlink()
            return tmp_dir
            
        except Exception as e:
            # Cleanup on failure
            if archive_path.exists():
                archive_path.unlink()
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail=f"Failed to extract zip file: {str(e)}")
    
    @staticmethod
    def create_zip_worker(tmp_dir: str, options, job) -> ZipWorker:
        return ZipWorker(tmp_dir, options, job)

    @staticmethod
    def create_onedrive_worker(base_dir: str, options, access_token: str, job) -> OneDriveWorker:
        return OneDriveWorker(base_dir, options, access_token, job)


class JobCleanupService:
    """Service class to handle job cleanup operations."""
    
    @staticmethod
    def cleanup_job(job_id: str, job):
        """Clean up job resources including files and directories."""
        try:
            if job.zip_path and os.path.exists(job.zip_path):
                os.remove(job.zip_path)
        except OSError as e:
            logger.warning(f"Failed to delete zip file {job.zip_path}: {e}")
        
        if job.base_dir and os.path.isdir(job.base_dir):
            try:
                shutil.rmtree(job.base_dir, ignore_errors=True)
            except Exception as e:
                logger.warning(f"Failed to remove base directory {job.base_dir}: {e}")
        
        JobManager.remove(job_id)