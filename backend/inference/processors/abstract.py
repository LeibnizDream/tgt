"""
Truly abstract base for all inference data processors.

:class:`AbstractProcessor` owns the pipeline loop and logging infrastructure.
Format-specific I/O (_find_files, _read_file, _write_file) and the core
transformation (_process_dataframe) are left abstract.
"""
from abc import ABC, abstractmethod
import logging
import sys
import os
from pathlib import Path
import pandas as pd


class StreamToLogger:
    """File-like object that forwards writes to a :class:`logging.Logger`."""

    def __init__(self, logger: logging.Logger, level: int = logging.INFO):
        self.logger = logger
        self.level = level

    def write(self, message: str) -> None:
        message = message.rstrip()
        if not message:
            return
        for line in message.splitlines():
            self.logger.log(self.level, line)

    def flush(self) -> None:
        pass


class Tee:
    """Writes to multiple streams simultaneously, swallowing per-stream errors."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, msg: str) -> None:
        for s in self.streams:
            try:
                s.write(msg)
            except Exception:
                pass

    def flush(self) -> None:
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


class AbstractProcessor(ABC):
    """Pipeline skeleton and logging infrastructure shared by all processors.

    Implements the find → read → process → write loop.  Subclasses provide
    the four abstract hooks; format-specific bases (LabvancedBaseProcessor,
    etc.) implement the I/O hooks, while concrete processors implement
    _process_dataframe.

    Attributes:
        language: Language code passed down to strategies.
        instruction: Sub-mode hint passed down to strategies.
        file_changed: Set to ``False`` in _process_dataframe to skip writing.
    """

    def __init__(self, language: str, instruction: str, device: str | None = None):
        self.language = language
        self.instruction = instruction
        self.file_changed = True

        try:
            import torch
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:
            self.device = device or "cpu"

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        logging.captureWarnings(True)
        warnings_logger = logging.getLogger("py.warnings")
        warnings_logger.setLevel(logging.INFO)
        warnings_logger.propagate = True

        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr

    def set_progress_callback(self, callback) -> None:
        """Register *callback(current, total)* to receive progress updates."""
        self._progress_callback = callback

    def _attach_session_handler(self, session_path: str) -> logging.FileHandler:
        """Open a per-session log file next to *session_path* and tee stdout/stderr into it."""
        log_dir = session_path if os.path.isdir(session_path) else str(Path(session_path).parent)
        log_path = os.path.join(log_dir, f"{self.__class__.__name__}.log")

        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")

        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        ch = logging.StreamHandler(self._orig_stdout)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        sys.stdout = Tee(self._orig_stdout, StreamToLogger(self.logger, logging.INFO))
        sys.stderr = Tee(self._orig_stderr, StreamToLogger(self.logger, logging.ERROR))

        self.logger.info(f"Started processing session at {session_path}")
        return fh

    def _detach_session_handler(self, fh: logging.FileHandler) -> None:
        """Restore stdout/stderr and close *fh*. Always call in a ``finally`` block."""
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr
        self.logger.info("Detaching session handler")
        self.logger.removeHandler(fh)
        fh.close()

    def process(self, folder: str) -> None:
        """Process a single folder: read → transform → write."""
        self.file_changed = True
        fh = self._attach_session_handler(folder)
        try:
            self.logger.info(f"Processing {folder}")
            df = self._read_file(folder)
            self.logger.info(f"Loaded {len(df)} rows")
            df = self._process_dataframe(df)
            if self.file_changed:
                self._write_file(folder, df)
        except FileNotFoundError as e:
            self.logger.warning(f"Skipping {folder}: {e}")
        finally:
            self.logger.info(f"Finished {folder}")
            self._detach_session_handler(fh)
    
    def _read_file(self, path: str) -> pd.DataFrame:
        """Load the Excel workbook at *path* into a DataFrame."""
        return pd.read_excel(path)
    
    @abstractmethod
    def _find_files(self, base_dir: str) -> list[str]:
        """Return ordered list of file paths to process under *base_dir*."""
        raise NotImplementedError

    @abstractmethod
    def _write_file(self, path: str, df: pd.DataFrame) -> None:
        """Persist *df* to *path*."""
        raise NotImplementedError
    
    @abstractmethod
    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply this processor's transformation. Set self.file_changed=False to skip write."""
        raise NotImplementedError
