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
logic, and progress callbacks — lives here and is shared by all subclasses.

Few-shot examples are managed by subclasses via self.example_store, an
ExampleStore instance created once per processor (i.e. per job).  No class-level
state is needed because the processor itself already has job lifetime.

Strategy construction
---------------------
The concrete strategy (which model to use) is resolved by StrategyFactory
inside __init__, so subclasses never import individual strategy classes or
factories directly.
"""

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from rich.logging import RichHandler


class AbstractProcessor(ABC):
    """Pipeline skeleton and shared infrastructure for all inference processors.

    Concrete subclasses pick a format (labvanced, plain) and a task
    (translate, gloss, …) and only implement _find_files, _write_file, and
    _process_dataframe.  The outer loop, logging, and error handling are
    inherited from here.
    """

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
        self._put_callback = None

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        logging.captureWarnings(True)

    def _emit(self, msg: str, level: str = "info") -> None:
        if self._put_callback:
            self._put_callback(msg, level)
        self.logger.log(getattr(logging, level.upper(), logging.INFO), msg)

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

    def process(self, folder: str, put=None, progress=None) -> None:
        """Process a single folder: find → read → transform → write."""
        self._put_callback = put
        self._progress_callback = progress
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
                    self._emit(f"Skipping missing file: {e}", level="warning")

                except Exception as e:
                    self.logger.exception("[bold red]Failed processing file[/bold red] %s", file)
                    self._emit(f"Failed processing file {os.path.basename(file)}: {e}", level="warning")

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
