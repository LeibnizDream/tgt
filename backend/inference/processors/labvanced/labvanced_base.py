"""
Labvanced-specific base processor.

Implements the three I/O hooks (_find_files, _read_file, _write_file) for the
Labvanced annotation pipeline.  _process_dataframe remains abstract — each
concrete processor (transcription, translation, etc.) implements it.
"""
import os
import fnmatch
import pandas as pd
from utils.functions import set_global_variables, format_excel_output
from inference.processors.abstract_processor import AbstractProcessor

LANGUAGES, NO_LATIN, OBLIGATORY_COLUMNS = set_global_variables()


class LabvancedBaseProcessor(AbstractProcessor):
    """Base class for all Labvanced processors.

    Adds columns_to_highlight for conditional Excel formatting and implements
    the Labvanced file-discovery and write conventions.
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
        self.columns_to_highlight = None

    def _find_files(self, base_dir: str) -> list[str]:
        """Return sorted paths to every ``trials_and_sessions_annotated.xlsx`` under *base_dir*."""
        matches = []
        for root, _, files in os.walk(base_dir):
            for f in files:
                if fnmatch.fnmatch(f, 'trials_and_sessions_annotated.xlsx'):
                    matches.append(os.path.join(root, f))
        self.logger.info(f"Found {len(matches)} matching files in {base_dir}")
        return sorted(matches)

    def _write_file(self, path: str, df: pd.DataFrame) -> None:
        """Write *df* back to *path*, placing obligatory columns last, with optional formatting."""
        extra_cols = [c for c in df.columns if c not in OBLIGATORY_COLUMNS]
        df = df[extra_cols + [c for c in OBLIGATORY_COLUMNS if c in df.columns]]
        df.to_excel(path, index=False)
        self.logger.info(f"Wrote output to {path}")
        if self.columns_to_highlight:
            format_excel_output(path, self.columns_to_highlight)
