"""
Labvanced column-creation processor.

Reads raw ``trials_and_sessions.csv`` files, passes each DataFrame through
:func:`utils.reorder_columns.create_columns` to derive the obligatory
annotation columns, then writes the result as
``trials_and_sessions_annotated.xlsx`` alongside the original CSV.

This processor runs before any annotation step; it is the entry point that
transforms a raw Labvanced export into the annotated-sheet format expected by
all other processors.
"""
import os
import pandas as pd
from inference.processors.labvanced.labvanced_base import LabvancedBaseProcessor
from utils.reorder_columns import create_columns
from utils.functions import (
    set_global_variables,
    format_excel_output
)
from pathlib import Path

LANGUAGES, NO_LATIN, OBLIGATORY_COLUMNS = set_global_variables()

class ColumnCreationProcessor(LabvancedBaseProcessor):
    """Seeds the annotated Excel sheet from a raw Labvanced CSV export."""

    def _find_files(self, base_dir: str) -> list[str]:
        """Return paths to every ``trials_and_sessions.csv`` under *base_dir*."""
        matches = []
        for root, _, files in os.walk(base_dir):
            for f in files:
                if f.endswith("trials_and_sessions.csv"):
                    matches.append(os.path.join(root, f))
        self.logger.info(f"Found {len(matches)} matching files in {base_dir}")
        return sorted(matches)

    def _read_file(self, path: str) -> pd.DataFrame:
        """Read the raw CSV (not Excel, unlike the base class)."""
        return pd.read_csv(path)

    def _write_file(self, path: str, df: pd.DataFrame) -> None:
        """Write *df* to ``<stem>_annotated.xlsx`` alongside the source CSV."""
        extra_cols = [c for c in df.columns if c not in OBLIGATORY_COLUMNS]
        df = df[extra_cols + [c for c in OBLIGATORY_COLUMNS if c in df.columns]]

        in_path = Path(path)
        out_path = in_path.with_name(in_path.stem + "_annotated.xlsx")

        df.to_excel(out_path, index=False)
        self.logger.info(f"Wrote output to {out_path}")

        if self.columns_to_highlight:
            format_excel_output(str(out_path), self.columns_to_highlight)

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive and insert all obligatory annotation columns for the target language."""
        return create_columns(df, self.language)