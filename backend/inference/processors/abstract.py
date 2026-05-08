"""
Abstract base for all inference data processors.

AbstractProcessor owns the pipeline loop and logging infrastructure.
Format-specific I/O (_find_files, _read_file, _write_file) and the core
transformation (_process_dataframe) are left abstract.
"""

from abc import ABC, abstractmethod
import logging
import os
from pathlib import Path

import pandas as pd
from rich.logging import RichHandler


class AbstractProcessor(ABC):
    """Pipeline skeleton and logging infrastructure shared by all processors."""

    def __init__(self, language: str, instruction: str, device: str | None = None):
        self.language = language
        self.instruction = instruction
        self.file_changed = True
        self._progress_callback = None

        try:
            import torch
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:
            self.device = device or "cpu"

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