"""
Default-format base processor.

Treats the input directory as a single session: _find_files returns [base_dir]
so the pipeline runs once per directory.  Concrete subclasses implement
_read_file (to build the initial DataFrame) and _process_dataframe.
"""

import os
import pandas as pd
from inference.processors.abstract import AbstractProcessor

class BasePlainProcessor(AbstractProcessor):

    def __init__(self, language: str, instruction: str, device: str | None = None):
        super().__init__(language, instruction, device)

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