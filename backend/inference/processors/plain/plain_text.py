"""
Plain-format base processor.

Provides file discovery and write conventions for the plain (non-Labvanced)
pipeline.  Rather than structured experiment directories, plain processors work
on flat folders of audio files and produce a single transcribed.xlsx per folder.
"""
import os
import pandas as pd
from inference.processors.abstract_processor import AbstractProcessor

_SOURCE_COLS = {
    "translate": "transcription",
    "gloss":     "to_gloss",
}

_TARGET_COLS = {
    "translate": ["translation"],
    "gloss":     ["glossing"],
}


class PlainTextProcessor(AbstractProcessor):
    """Base class for plain-format processors.

    Discovers transcribed.xlsx files under a root directory and applies
    the shared _process_dataframe skeleton for translation and glossing.
    PlainTranscriber overrides _process_dataframe for its audio-file pattern.
    """

    def __init__(self, language, action, model=None):
        super().__init__(language, action, model)

    def _find_files(self, base_dir: str) -> list[str]:
        """Return all transcribed.xlsx files under base_dir."""
        matches = []
        for root, _, files in os.walk(base_dir):
            for f in files:
                if f.lower() == "transcribed.xlsx":
                    matches.append(os.path.join(root, f))
        self.logger.info(f"Found {len(matches)} matching files in {base_dir}")
        return sorted(matches)

    def _write_file(self, path: str, df: pd.DataFrame) -> None:
        """Write df to the Excel file at path."""
        df.to_excel(path, index=False)
        self.logger.info(f"Wrote output to {path}")

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Translate/gloss pending rows and write results to target columns."""
        source_col = _SOURCE_COLS.get(self.action)
        target_cols = _TARGET_COLS.get(self.action, [])

        if not source_col or source_col not in df.columns:
            self.logger.warning(f"Source column '{source_col}' not found, skipping.")
            self.file_changed = False
            return df

        for col in target_cols:
            if col not in df.columns:
                df[col] = pd.NA
            df[col] = df[col].astype(object)

        progress_cb = getattr(self, "_progress_callback", None)
        had_examples, todo_items = self._separate_examples_and_todo(
            df, source_col, target_cols[-1], self.action
        )

        if had_examples or not todo_items:
            self.file_changed = False
            return df

        id_to_result = self.strategy.run_strategy(todo_items, self._get_examples(), progress_cb)
        for i, result in id_to_result.items():
            for col in target_cols:
                df.at[i, col] = result

        return df
