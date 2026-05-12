import os
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from huggingface_hub import login

from dotenv import load_dotenv
from inference.processing_options import ProcessingOptions
from inference.processors.processor_factory import ProcessorFactory
from routers.helpers.job_manager import JobPublisher
from utils.functions import find_language, set_global_variables

_SECRETS = Path(__file__).resolve().parent.parent / "materials" / "secrets.env"
REQUIRED_ENV_KEYS = [
    "HUGGING_KEY",
    "TENANT_ID",
    "CLIENT_ID",
    "CLIENT_SECRET",
    "DEEPL_API_KEY",
    "GOOGLE_API_KEY",
]

LANGUAGES, NO_LATIN, OBLIGATORY_COLUMNS = set_global_variables()


class AbstractInferenceWorker(ABC):
    """Orchestrates a full inference job across a directory tree.

    Responsibility split
    --------------------
    AbstractInferenceWorker handles everything outside the model call: loading
    secrets, building the processor, iterating over folders, cancellation
    checks, and reporting progress/errors to the job queue.  The actual NLP
    work is delegated to the AbstractProcessor subclass selected by
    ProcessorFactory.

    Why the processor is created inside run(), not __init__
    --------------------------------------------------------
    Processors hold large ML models (Whisper, MarianMT, Qwen, …).  When a job
    runs in a subprocess (via multiprocessing), the worker object is pickled and
    sent to the child process.  ML model objects are not picklable, so they must
    be instantiated after the fork — inside run().

    Subclasses must implement
    --------------------------
        _initial_message() — send job-start message (content differs per worker type)
        _after_process()   — hook called after each folder completes.
                             Use self.current_folder to know which folder just finished.
    """

    def __init__(self, base_dir: str, options: ProcessingOptions, publisher: JobPublisher | None = None):
        self.base_dir = base_dir
        self.current_folder = self.base_dir
        self.options = options
        self.options.language = find_language(options.language, LANGUAGES)
        self.publisher = publisher if publisher is not None else JobPublisher()
        self.cancel = self.publisher.cancel

    def _folder_to_process(self):
        """Yield immediate subdirectories of base_dir, or base_dir itself if flat."""
        subdirs = sorted(e.path for e in os.scandir(self.base_dir) if e.is_dir())
        if not subdirs:
            yield self.base_dir
            return
        for path in subdirs:
            if self.options.format == "labvanced" and not os.path.basename(path).lower().startswith("session"):
                continue
            yield path

    @property
    def _is_cancelled(self) -> bool:
        return bool(self.cancel and self.cancel.is_set())

    def inform(self, msg: str, level: str = "info") -> None:
        self.publisher.inform(msg, level)

    def _progress(self, current: int, total: int) -> None:
        self.publisher.progress(current, total)

    @abstractmethod
    def _initial_message(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _after_process(self) -> None:
        raise NotImplementedError

    @staticmethod
    def validate_env_keys(required_keys: list[str]) -> None:
        missing = [key for key in required_keys if key not in os.environ]
        if missing:
            raise OSError(
                "Missing required environment variable(s): "
                + ", ".join(missing)
                + "\nThey must exist in secrets.env, even if left empty."
            )

    def run(self) -> None:
        load_dotenv(_SECRETS, override=True)
        self.validate_env_keys(REQUIRED_ENV_KEYS)

        token = os.getenv("HUGGING_KEY", "").strip()
        if token:
            login(token=token)

        try:
            self._initial_message()

            # Processor is created once inside run() — multiprocessing requires
            # heavy objects (ML models) to be instantiated in the worker process.
            self.processor = ProcessorFactory.get_processor(self.options)
            self.publisher.inform(f"Using processor: {self.processor.__class__.__name__}")

            for folder in self._folder_to_process():
                self.current_folder = folder
                if self._is_cancelled:
                    break

                self.publisher.inform(f"Processing folder: {os.path.basename(os.path.normpath(folder))}")
                self.processor.process(folder, put=self.inform, progress=self._progress)
                self._after_process()

        except Exception as e:
            self.publisher.inform(str(e), level="error")
            self.publisher.inform(traceback.format_exc())
        finally:
            if self._is_cancelled:
                self.publisher.cancelled()
            else:
                self.publisher.done()
