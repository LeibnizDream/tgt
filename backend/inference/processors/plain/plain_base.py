"""
Plain-format base processor.

Provides file discovery and write conventions for the plain (non-Labvanced)
pipeline.  Rather than structured experiment directories, plain processors work
on flat folders of audio files and produce a single ``transcribed.xlsx`` per
folder.
"""

import os
import pandas as pd
from inference.processors.abstract_processor import AbstractProcessor


class BasePlainProcessor(AbstractProcessor):
    """Base class for plain-format processors.

    Discovers ``transcribed.xlsx`` files under a root directory and delegates
    transformation to :meth:`_process_dataframe` in each concrete subclass.
    """

    def __init__(
        self,
        language: str,
        instruction: str,
        action: str | None = None,
        translationModel: str | None = None,
        glossingModel: str | None = None,
        transliterationModel: str | None = None,
        device: str | None = None,
    ):
        super().__init__(language, instruction, action, translationModel, glossingModel, transliterationModel, device)

    def _find_files(self, base_dir: str) -> list[str]:
        """Return all transcribed.xlsx files under *base_dir*."""
        matches = []

        for root, _, files in os.walk(base_dir):
            for f in files:
                if f.lower() == "transcribed.xlsx":
                    matches.append(os.path.join(root, f))

        self.logger.info(f"Found {len(matches)} matching files in {base_dir}")
        return sorted(matches)

    def _write_file(self, path: str, df: pd.DataFrame) -> None:
        """Write *df* to the Excel file at *path*."""
        df.to_excel(path, index=False)
        self.logger.info(f"Wrote output to {path}")