import os
import traceback
import argparse
from abc import ABC, abstractmethod
from pathlib import Path
from dotenv import load_dotenv
from utils.functions import find_language, set_global_variables

from inference.processors.factory import ProcessorFactory
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
    """
    Abstract base class for inference workers that process data folders using
    dynamic processors from ProcessorFactory. Subclasses should implement
    lifecycle hooks: initial_message, folder_to_process, and after_process.
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

    @abstractmethod
    def _initial_message(self) -> None:
        """
        Hook for sending an initial start-up message.
        """
        raise NotImplementedError("Subclasses must implement initial_message()")

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

    @abstractmethod
    def _after_process(self) -> None:
        """
        Actions to perform immediately after preprocessing step completes.
        use self.current_folder to access the folder being processed.
        """
        raise NotImplementedError("Subclasses must implement after_process()")
    
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


class LocalWorker(AbstractInferenceWorker):
    """
    Local implementation of AbstractInferenceWorker for CLI or API usage.
    Defines simple folder iteration and messaging behavior.
    """

    def _initial_message(self) -> None:
        """
        Print or queue the initial job start message.
        """
        self._put(f"Starting job {self.job_id} – action: {self.action}")

    def _after_process(self) -> None:
        """
        Print or queue a message after processing a folder.
        """
        self._put(f"Processed folder {self.current_folder} for job {self.job_id}")


def main() -> None:
    """
    Command-line interface entry point for running inference workers.

    Positional arguments:
        action      : transcribe | translate | transliterate | gloss
        language    : language name/code
        base_dir    : directory to process

    Optional arguments:
        --format
        --instruction
        --translation-model
        --glossing-model
    """

    parser = argparse.ArgumentParser(
        description="Run inference worker from the command line."
    )

    # Positional arguments
    parser.add_argument(
        "action",
        choices=["transcribe", "translate", "transliterate", "gloss"],
        help="Action to perform"
    )

    parser.add_argument(
        "language",
        help="Language name or code"
    )

    parser.add_argument(
        "base_dir",
        help="Directory to process"
    )

    # Optional arguments
    parser.add_argument(
        "--format",
        default="plain",
        choices=["labvanced", "plain"],
        help="Input/output format"
    )

    parser.add_argument(
        "--instruction",
        default=None,
        choices=["automatic", "corrected", "sentences"],
        help="Required for labvanced format"
    )

    parser.add_argument(
        "--translation-model",
        default=None,
        help="Translation model name"
    )

    parser.add_argument(
        "--glossing-model",
        default=None,
        help="Glossing model name"
    )

    args = parser.parse_args()

    if args.format == "labvanced" and not args.instruction:
        parser.error("--instruction is required when --format is labvanced")

    worker = LocalWorker(
        base_dir=args.base_dir,
        action=args.action,
        language=args.language,
        format=args.format,
        instruction=args.instruction,
        translationModel=args.translation_model,
        glossingModel=args.glossing_model,
    )

    worker.run()

if __name__ == "__main__":
    main()