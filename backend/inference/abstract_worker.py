import os
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from dotenv import load_dotenv
from utils.functions import find_language, set_global_variables

from inference.processors.processor_factory import ProcessorFactory
from inference.processors.labvanced.glossing import GlossingProcessor

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

    def __init__(self, base_dir: str, action: str, language: str,
                 instruction: str | None = None,
                 translationModel: str = None, glossingModel: str = None,
                 format: str | None = None, job=None):
        """
        Initialize the inference worker with configuration parameters.

        Args:
            base_dir (str): Path to the root directory for processing.
            action (str): Action type (e.g., 'transcribe', 'translate', 'gloss').
            language (str): Language code or name.
            instruction (str, optional): Sub-mode for labvanced ('automatic',
                'corrected', 'sentences'). Not used for plain format.
            translationModel (str, optional): Name of translation model to use.
            glossingModel (str, optional): Name of glossing model to use.
            job (optional): Job object providing id, queue, and cancel_event.
        """
        self.base_dir = base_dir
        self.current_folder = self.base_dir
        self.action = action
        self.language = find_language(language, LANGUAGES)
        self.instruction = instruction
        self.translationModel = translationModel
        self.glossingModel = glossingModel
        self.format = format or "plain"
        self.job = job

        # Setup job identification and messaging queue
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
            if self.format == "labvanced" and not os.path.basename(path).lower().startswith("session"):
                continue
            yield path

    
    def _put(self, msg: str) -> None:
        """
        Send a status message to the job queue or print to console.

        Args:
            msg (str): Message content.
        """
        if self.q:
            self.q.put(msg)
        else:
            print(msg)
    
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
            raise EnvironmentError(
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

        print("SECRETS PATH:", _SECRETS)
        print("SECRETS EXISTS:", _SECRETS.exists())

        google_key = os.getenv("GOOGLE_API_KEY", "")
        print("GOOGLE_API_KEY:", repr(google_key[:8] + "..." if google_key else google_key))

        self.validate_env_keys(REQUIRED_ENV_KEYS)
        

        try:
            self._initial_message()

            # Processor is created once inside run() — multiprocessing requires
            # heavy objects (ML models) to be instantiated in the worker process.
            self.processor = ProcessorFactory.get_processor(
                language=self.language,
                action=self.action,
                format=self.format,
                instruction=self.instruction,
                translationModel=self.translationModel,
                glossingModel=self.glossingModel,
            )
            self.processor.set_progress_callback(
                lambda cur, tot: self._put(f"[PROGRESS] {cur}/{tot}")
            )
            self._put(f"Using processor: {self.processor.__class__.__name__}")

            for folder in self._folder_to_process():
                self.current_folder = folder
                if self.cancel and self.cancel.is_set():
                    self._put("[CANCELLED]")
                    break

                self._put(f"Processing folder: {os.path.basename(os.path.normpath(folder))}")
                self.processor.process(folder)
                self._after_process()

        except Exception as e:
            self._put(f"[ERROR] {e}")
            self._put(traceback.format_exc())
        finally:
            if isinstance(self.processor, GlossingProcessor):
                GlossingProcessor.reset_examples()
            self._put("[DONE ALL]")