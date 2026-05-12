import logging
import multiprocessing
import uuid
from multiprocessing import Process
from multiprocessing import Queue as Queue
from multiprocessing.synchronize import Event

from fastapi import HTTPException

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Default"


def normalize_model_name(model: str | None) -> str | None:
    return None if model == DEFAULT_MODEL else model


class JobPublisher:
    """Message bus passed to workers.

    With a queue → sends structured dicts consumed by the SSE stream.
    Without a queue (job_id defaults to "local") → prints to stdout for CLI runs.
    """

    def __init__(self, queue = None, cancel: Event | None = None, job_id: str = "local"):
        self._queue = queue
        self.cancel = cancel
        self.job_id = job_id

    def inform(self, msg: str, level: str = "info") -> None:
        if self._queue is not None:
            self._queue.put({"type": level, "message": msg})
        else:
            prefix = f"[{level.upper()}] " if level != "info" else ""
            print(f"{prefix}{msg}")

    def progress(self, current: int, total: int) -> None:
        if self._queue is not None:
            self._queue.put({"type": "progress", "current": current, "total": total})
        else:
            print(f"[PROGRESS] {current}/{total}")

    def done(self) -> None:
        if self._queue is not None:
            self._queue.put({"type": "done"})
        else:
            print("[DONE ALL]")

    def cancelled(self) -> None:
        if self._queue is not None:
            self._queue.put({"type": "cancelled"})
        else:
            print("[CANCELLED]")

    def get_message(self):
        return self._queue.get() if self._queue is not None else None


class Job:
    def __init__(self, job_id: str):
        self.id = job_id
        self._queue: Queue = Queue()
        self.cancel_event: Event = multiprocessing.Event()
        self.zip_path: str | None = None
        self.token: str | None = None
        self.process: Process | None = None
        self.publisher: JobPublisher = JobPublisher(self._queue, self.cancel_event, job_id)


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
