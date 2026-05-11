import logging
import uuid
from multiprocessing import Event, Process, Queue

from fastapi import HTTPException

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Default"


def normalize_model_name(model: str | None) -> str | None:
    return None if model == DEFAULT_MODEL else model


class Job:
    def __init__(self, job_id: str):
        self.id = job_id
        self.queue: Queue = Queue()
        self.cancel_event: Event = Event()
        self.base_dir: str | None = None
        self.zip_path: str | None = None
        self.token: str | None = None
        self.process: Process | None = None


class JobManager:
    _jobs: dict[str, Job] = {}

    @classmethod
    def create(cls) -> Job:
        job_id = str(uuid.uuid4())
        job = Job(job_id)
        cls._jobs[job_id] = job
        return job

    @classmethod
    def get(cls, job_id: str) -> Job:
        job = cls._jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
        return job

    @classmethod
    def remove(cls, job_id: str):
        cls._jobs.pop(job_id, None)
