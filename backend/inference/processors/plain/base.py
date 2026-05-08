"""
Default-format base processor.

Treats the input directory as a single session: _find_files returns [base_dir]
so the pipeline runs once per directory.  Concrete subclasses implement
_read_file (to build the initial DataFrame) and _process_dataframe.
"""
import fnmatch
import os
import pandas as pd
from inference.processors.abstract import AbstractProcessor


class BasePlainProcessor(AbstractProcessor):
    """Base for all default-format processors.

    Provides directory-level find/write so subclasses only need to implement
    _read_file and _process_dataframe.
    """

    def __init__(self, language: str, instruction: str, device: str | None = None):
        super().__init__(language, instruction, device)

    def _find_files(self, base_dir: str) -> list[str]:
        """Return [base_dir] — the whole directory is one session."""
        matches = []
        for root, _, files in os.walk(base_dir):
            for f in files:
                if fnmatch.fnmatch(f, 'transcribed.xlsx'):
                    matches.append(os.path.join(root, f))
        self.logger.info(f"Found {len(matches)} matching files in {base_dir}")
        return sorted(matches)

    def _write_file(self, _path: str, df: pd.DataFrame) -> None:
        """Write *df* to transcription.xlsx in the session directory."""
        df.to_excel(_path, index=False)
        self.logger.info(f"Wrote output to {_path}")
