import os
import pandas as pd
from tqdm import tqdm
from inference.processors.plain.base import BasePlainProcessor
from inference.strategies.pii_identifier.factory import PIIIdentifierFactory
from inference.strategies.transcription.factory import TranscriptionStrategyFactory


class PlainTranscriber(BasePlainProcessor):

    def __init__(self, language: str, instruction: str, device: str | None = None):
        super().__init__(language, instruction, device)
        self.pii_identifier = PIIIdentifierFactory.get_strategy(self.language)
        self.strategy = TranscriptionStrategyFactory.get_strategy(self.language)
        self.logger.info(f"Initialized transcription strategy: {self.strategy.__class__.__name__}")

    def _find_files(self, base_dir: str) -> list[str]:
        """Return every directory under *base_dir* (inclusive) that contains audio files."""
        dirs = [
            root for root, _, files in os.walk(base_dir)
            if any(f.lower().endswith(('.mp3', '.mp4', '.m4a')) for f in files)
        ]
        self.logger.info(f"Found {len(dirs)} folder(s) with audio under {base_dir}")
        return sorted(dirs)

    def _read_file(self, directory: str) -> pd.DataFrame:
        """Scan *directory* (non-recursively) for audio files."""
        audio_files = sorted(
            f for f in os.listdir(directory)
            if f.lower().endswith(('.mp3', '.mp4', '.m4a'))
        )
        self._current_dir = directory
        self.logger.info(f"Found {len(audio_files)} audio files in {directory}")
        return pd.DataFrame({"file_name": audio_files, "transcription": [""] * len(audio_files)})

    def _write_file(self, directory: str, df: pd.DataFrame) -> None:
        """Write *df* to ``transcribed.xlsx`` inside *directory*."""
        out_path = os.path.join(directory, "transcribed.xlsx")
        df.to_excel(out_path, index=False)
        self.logger.info(f"Wrote output to {out_path}")

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transcribe each audio file and fill the transcription column."""
        total = len(df)
        progress_cb = getattr(self, '_progress_callback', None)

        for i, row in tqdm(df.iterrows(), total=total, desc="Transcribing"):
            path = os.path.join(self._current_dir, row["file_name"])
            try:
                text = self.strategy.transcribe(path)
                if self.pii_identifier:
                    _, text = self.pii_identifier.identify_and_annotate(text)
                df.at[i, "transcription"] = text
            except Exception as e:
                self.logger.error(f"Error transcribing '{row['file_name']}': {e}")
            if progress_cb:
                progress_cb(i + 1, total)

        return df
