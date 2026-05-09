"""
Abstract base for all inference data processors.

AbstractProcessor owns the pipeline loop and logging infrastructure.
Format-specific I/O (_find_files, _read_file, _write_file) and the core
transformation (_process_dataframe) are left abstract.
"""

from abc import ABC, abstractmethod
import json
import logging
import torch
import os
from pathlib import Path

import pandas as pd
from rich.logging import RichHandler
from inference.strategies.strategy_factory import StrategyFactory


class AbstractProcessor(ABC):
    """Pipeline skeleton and logging infrastructure shared by all processors."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._shared_examples: dict = {}
        cls._shared_index: int = 0

    @classmethod
    def reset_examples(cls) -> None:
        cls._shared_examples = {}
        cls._shared_index = 0

    def _separate_examples_and_todo(
        self,
        df: pd.DataFrame,
        source_col: str,
        target_col: str,
        example_target_key: str,
    ) -> tuple[bool, list]:
        """Split rows into few-shot examples (already processed) and todo items (pending)."""
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

    def _call_with_llm(self, strategy_fn, todo_items: list, result_key: str, progress_cb=None) -> dict:
        if not todo_items:
            return {}
        examples = list(type(self)._shared_examples.values())[:10]
        payload = json.dumps({"examples": examples, "items": todo_items}, ensure_ascii=False)
        response = json.loads(strategy_fn(payload))
        result = {item["id"]: item[result_key] for item in response["items"]}
        if progress_cb:
            progress_cb(len(todo_items), len(todo_items))
        return result

    def __init__(
        self,
        language: str,
        instruction: str,
        action: str | None = None,
        translationModel: str | None = None,
        glossingModel: str | None = None,
        device: str | None = None,
    ):
        self.language = language
        self.instruction = instruction
        self.file_changed = True
        self._progress_callback = None
        self.strategy = StrategyFactory.get_strategy(action, language, translationModel, glossingModel) if action else None

        try:
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