import os
import pandas as pd
from tqdm import tqdm
from inference.processors.plain.plain_base import BasePlainProcessor


class PlainTranscriber(BasePlainProcessor):

    def __init__(self, language: str, instruction: str, device: str | None = None):
        super().__init__(language, instruction, action="transcribe", device=device)
        self.logger.info(f"Initialized transcription strategy: {self.strategy.__class__.__name__}")

    def _find_files(self, base_dir: str) -> list[str]:
        """Return output Excel paths for every directory under base_dir containing audio files."""
        output_files = []

        for root, _, files in os.walk(base_dir):
            if any(f.lower().endswith((".mp3", ".mp4", ".m4a")) for f in files):

                out_path = os.path.join(root, "transcribed.xlsx")

                if os.path.exists(out_path):
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

        return pd.DataFrame({
            "file_name": audio_files,
            "transcription": [""] * len(audio_files),
        })

    def _write_file(self, path: str, df: pd.DataFrame) -> None:
        df.to_excel(path, index=False)
        self.logger.info(f"Wrote output to {path}")

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transcribe each audio file and fill the transcription column."""
        total = len(df)
        progress_cb = getattr(self, '_progress_callback', None)

        for i, row in tqdm(df.iterrows(), total=total, desc="Transcribing"):
            path = os.path.join(self._current_dir, row["file_name"])
            try:
                text = self.strategy.transcribe(path)
                df.at[i, "transcription"] = text
            except Exception as e:
                self.logger.error(f"Error transcribing '{row['file_name']}': {e}")
            if progress_cb:
                progress_cb(i + 1, total)

        return df
