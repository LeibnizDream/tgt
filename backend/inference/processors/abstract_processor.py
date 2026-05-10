"""
Abstract base for all inference data processors.

Design rationale
----------------
Every processor (transcription, translation, glossing, …) in both the labvanced
and plain formats shares the same outer loop: find files, read each one,
transform it, write it back.  AbstractProcessor owns that loop so that concrete
subclasses only need to implement three things:

    _find_files        — how to discover input files under a root directory
    _write_file        — how to persist a transformed DataFrame
    _process_dataframe — the task-specific transformation itself

Everything else — logging setup, per-file error handling, skip-if-unchanged
logic, progress callbacks, strategy construction, and the few-shot LLM batch
mechanism — lives here and is shared by all subclasses.

Few-shot example accumulation
------------------------------
LLM-backed processors improve with examples.  As files are processed in a job,
rows that already carry a human-provided output (a translation, a gloss, …) are
harvested as few-shot examples for subsequent files.  This state is kept in
per-subclass class variables (_shared_examples, _shared_index) rather than
instance variables so that examples accumulate across processor instances within
the same job.  __init_subclass__ creates a fresh dict for every concrete
subclass automatically, preventing cross-task contamination (glossing examples
never leak into translation).  Call reset_examples() between jobs.

Strategy construction
---------------------
The concrete strategy (which model to use) is resolved by StrategyFactory
inside __init__, so subclasses never import individual strategy classes or
factories directly.
"""

from abc import ABC, abstractmethod
import logging
import os
from pathlib import Path

import pandas as pd
from rich.logging import RichHandler
from inference.strategies.strategy_factory import StrategyFactory


class AbstractProcessor(ABC):
    """Pipeline skeleton and shared infrastructure for all inference processors.

    Concrete subclasses pick a format (labvanced, plain) and a task
    (translate, gloss, …) and only implement _find_files, _write_file, and
    _process_dataframe.  The outer loop, logging, error handling, and LLM
    batch logic are inherited from here.

    Class-level state (_shared_examples, _shared_index)
    ----------------------------------------------------
    Defined per subclass via __init_subclass__ so that each processor type
    keeps its own example pool.  These are class variables (not instance
    variables) because the same pool must be visible across multiple processor
    instances that run sequentially in one job.
    """

    def __init_subclass__(cls, **kwargs):
        # Called once per subclass definition.  Gives each concrete class its
        # own example dict so GlossingProcessor and TranslationProcessor never
        # share state, even though both inherit this mechanism.
        super().__init_subclass__(**kwargs)
        cls._shared_examples: dict = {}
        cls._shared_index: int = 0

    @classmethod
    def reset_examples(cls) -> None:
        """Clear accumulated few-shot examples.  Must be called between jobs."""
        cls._shared_examples = {}
        cls._shared_index = 0

    def _separate_examples_and_todo(
        self,
        df: pd.DataFrame,
        source_col: str,
        target_col: str,
        example_target_key: str,
    ) -> tuple[bool, list]:
        """Split rows into already-processed (examples) and pending (todo) sets.

        Rows with a non-empty target are harvested into _shared_examples so
        later files in the same job can use them as few-shot context.  If any
        row already has a target the whole file is considered done and the
        caller should set file_changed=False and return early — we don't
        overwrite partial human corrections.
        """
        cls = type(self)
        had_examples = False
        todo_items = []

        for i in range(len(df)):
            source = df.at[i, source_col]
            target = df.at[i, target_col] if target_col in df.columns else None

            if not isinstance(source, str) or not source.strip():
                continue

            if isinstance(target, str) and target.strip():
                cls._shared_examples[cls._shared_index] = {"source": source, example_target_key: target}
                cls._shared_index += 1
                had_examples = True
            else:
                todo_items.append({"id": i, "text": source})

        return had_examples, todo_items

    def _get_examples(self) -> list:
        return list(type(self)._shared_examples.values())[:20]

    def __init__(
        self,
        language: str,
        action: str,
        model: str | None = None
    ):
        self.language = language
        self.action = action
        self.model = model
        self.file_changed = True
        self._progress_callback = None
        self.strategy = StrategyFactory.get_strategy(language, action, model)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        logging.captureWarnings(True)

    def set_progress_callback(self, callback) -> None:
        """Register callback(current, total) to receive progress updates."""
        self._progress_callback = callback

    def _attach_session_handler(self, session_path: str) -> logging.FileHandler:
        """Attach file and rich console handlers for one processing session."""
        log_dir = (
            session_path
            if os.path.isdir(session_path)
            else str(Path(session_path).parent)
        )
        log_path = os.path.join(log_dir, f"{self.__class__.__name__}.log")

        self.logger.handlers.clear()

        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
        )
        self.logger.addHandler(file_handler)

        console_handler = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_level=True,
            show_path=False,
        )
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(console_handler)

        self.logger.info(
            "[bold green]Started processing session[/bold green] %s",
            session_path,
        )

        return file_handler

    def _detach_session_handler(self, file_handler: logging.FileHandler) -> None:
        """Detach and close handlers created for the current session."""
        self.logger.info("Detaching session handler")

        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)
            handler.close()

    def process(self, folder: str) -> None:
        """Process a single folder: find → read → transform → write."""
        file_handler = self._attach_session_handler(folder)

        try:
            files = self._find_files(folder)
            self.logger.info("Found %d file(s) to process", len(files))

            for file in files:
                self.file_changed = True

                try:
                    self.logger.info("[cyan]Processing[/cyan] %s", file)

                    df = self._read_file(file)
                    self.logger.info("Loaded %d row(s)", len(df))

                    df = self._process_dataframe(df)

                    if self.file_changed:
                        self._write_file(file, df)
                    else:
                        self.logger.info("[yellow]No changes; skipping write[/yellow] %s", file)

                except FileNotFoundError as e:
                    self.logger.warning("[yellow]Skipping missing file:[/yellow] %s", e)

                except Exception:
                    self.logger.exception("[bold red]Failed processing file[/bold red] %s", file)

        finally:
            self.logger.info("[bold green]Finished session[/bold green] %s", folder)
            self._detach_session_handler(file_handler)

    def _read_file(self, path: str) -> pd.DataFrame:
        """Load the Excel workbook at path into a DataFrame."""
        return pd.read_excel(path)

    @abstractmethod
    def _find_files(self, base_dir: str) -> list[str]:
        """Return ordered list of file paths to process under base_dir."""
        raise NotImplementedError

    @abstractmethod
    def _write_file(self, path: str, df: pd.DataFrame) -> None:
        """Persist df to path."""
        raise NotImplementedError

    @abstractmethod
    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply processor transformation."""
        raise NotImplementedError