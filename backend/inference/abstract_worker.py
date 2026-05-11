import os
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from huggingface_hub import login

from dotenv import load_dotenv
from inference.processing_options import ProcessingOptions
from inference.processors.processor_factory import ProcessorFactory
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
        _after_process()   — hook called after each folder completes, e.g. to
                             trigger a post-processing step or send a status update.
                             Use self.current_folder to know which folder just finished.
    """

    def __init__(self, base_dir: str, options: ProcessingOptions, job=None):
        self.base_dir = base_dir
        self.current_folder = self.base_dir
        self.options = options
        self.options.language = find_language(options.language, LANGUAGES)
        self.job = job

        self.job_id = job.id if job else 'local_job'
        self.q = getattr(job, 'queue', None)
        self.cancel = getattr(job, 'cancel_event', None)
        self.processor = None


    def _folder_to_process(self):
        """Yield immediate subdirectories of base_dir, or base_dir itself if flat.

        For labvanced format, only yields folders whose name starts with 'Session'
        (case-insensitive). Override in subclasses that need custom discovery.
        """
        subdirs = sorted(e.path for e in os.scandir(self.base_dir) if e.is_dir())
        if not subdirs:
            yield self.base_dir
            return
        for path in subdirs:
            if self.options.format == "labvanced" and not os.path.basename(path).lower().startswith("session"):
                continue
            yield path

    
    def _put(self, msg: str) -> None:
        if self.q:
            self.q.put(msg)
        else:
            print(msg)

    def _progress(self, cur: int, tot: int) -> None:
        self._put(f"[PROGRESS] {cur}/{tot}")
    
    @abstractmethod
    def _initial_message(self) -> None:
        """
        Hook for sending an initial start-up message.
        """
        raise NotImplementedError("Subclasses must implement initial_message()")
    
    @abstractmethod
    def _after_process(self) -> None:
        """
        Actions to perform immediately after preprocessing step completes.
        use self.current_folder to access the folder being processed.
        """
        raise NotImplementedError("Subclasses must implement after_process()")
    
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
        """
        Execute the inference workflow: send initial message, iterate over
        folders, instantiate the appropriate processor, and process each folder.
        Handles cancellation and exceptions.
        """
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
            self._put(f"Using processor: {self.processor.__class__.__name__}")

            for folder in self._folder_to_process():
                self.current_folder = folder
                if self.cancel and self.cancel.is_set():
                    self._put("[CANCELLED]")
                    break

                self._put(f"Processing folder: {os.path.basename(os.path.normpath(folder))}")
                self.processor.process(folder, put=self._put, progress=self._progress)
                self._after_process()

        except Exception as e:
            self._put(f"[ERROR] {e}")
            self._put(traceback.format_exc())
        finally:
            self._put("[DONE ALL]")