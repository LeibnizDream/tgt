"""
Plain transcription processor.

Walks a directory tree for audio files, creates a ``transcribed.xlsx`` output
sheet per audio folder, then transcribes each file and fills in the
``transcription`` and ``to_gloss`` columns.  Folders that already have a
``transcribed.xlsx`` are skipped to avoid overwriting prior work.
"""
import os

import pandas as pd
from inference.processors.abstract_processor import AbstractProcessor
from inference.strategies.strategy_factory import StrategyFactory
from tqdm import tqdm
from utils.functions import find_ffmpeg


ffmpeg_path = find_ffmpeg()


class PlainTranscriptionProcessor(AbstractProcessor):
    """Transcribes audio files in a flat folder and writes a ``transcribed.xlsx`` sheet."""

    def __init__(self, language, action, model=None):
        super().__init__(language, action, model)
        self.strategy = StrategyFactory.get_strategy(language, "transcribe", model)
        self.logger.info(f"Initialized transcription strategy: {self.strategy.__class__.__name__}")

    def _find_files(self, base_dir: str) -> list[str]:
        """Return output Excel paths for every directory under base_dir containing audio files."""
        output_files = []

        for root, _, files in os.walk(base_dir):
            if any(f.lower().endswith((".mp3", ".mp4", ".m4a")) for f in files):

                out_path = os.path.join(root, "transcribed.xlsx")

                if os.path.exists(out_path):
                    self._emit(f"Output already exists , skipping transcription for this folder.", level="error")
                    self.logger.warning(f"[bold red]Skipping existing output: {out_path}")
                    continue

                output_files.append(out_path)

        self.logger.info(f"Found {len(output_files)} audio folder(s) under {base_dir}")
        return sorted(output_files)

    def _read_file(self, path: str) -> pd.DataFrame:
        """Build dataframe from audio files in the directory containing path."""
        directory = os.path.dirname(path)

        audio_files = sorted(
            f for f in os.listdir(directory)
            if f.lower().endswith((".mp3", ".mp4", ".m4a"))
        )

        self._current_dir = directory
        self.logger.info(f"Found {len(audio_files)} audio files in {directory}")

        n = len(audio_files)
        return pd.DataFrame({
            "file_name": audio_files,
            "transcription": [""] * n,
            "transliteration": [""] * n,
            "to_gloss": [""] * n,
            "translation": [""] * n,
            "glossing": [""] * n,
        })

    def _write_file(self, path: str, df: pd.DataFrame) -> None:
        """Write *df* to *path* (the pre-computed ``transcribed.xlsx`` location)."""
        df.to_excel(path, index=False)
        self.logger.info(f"Wrote output to {path}")

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transcribe each audio file and fill the transcription column."""
        total = len(df)
        progress_cb = self._progress_callback

        for i, row in tqdm(df.iterrows(), total=total, desc="Transcribing"):
            path = os.path.join(self._current_dir, row["file_name"])
            try:
                text = self.strategy.run_strategy(path)
                df.at[i, "transcription"] = text
            except Exception as e:
                self.logger.exception(f"Error transcribing '{row['file_name']}': {e}")
                self._emit(f"Error transcribing '{row['file_name']}'", level="warning")
            if progress_cb:
                progress_cb(i + 1, total)

        return df
