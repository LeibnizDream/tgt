"""
Plain translation processor.

Reads ``transcribed.xlsx`` files produced by :class:`PlainTranscriber` and
fills the ``translation`` column.  Skips files that already have any human
translation to avoid overwriting corrections.
"""
import pandas as pd
from tqdm import tqdm

from inference.processors.plain.plain_base import BasePlainProcessor
from inference.strategies.translation.llm import LLMTranslationStrategy


class PlainTranslator(BasePlainProcessor):
    """Translates the ``transcription`` column and writes results to ``translation``."""

    def __init__(self, language: str, instruction: str, translationModel: str = None, device: str | None = None):
        super().__init__(language, instruction, action="translate", translationModel=translationModel, device=device)
        self.logger.info(f"Initialized translation strategy: {self.strategy.__class__.__name__}")

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Translate pending rows from ``transcription`` into ``translation``.

        Returns *df* unchanged (with ``file_changed=False``) when the
        ``transcription`` column is absent or any row already has a translation.
        """
        if "transcription" not in df.columns:
            self.logger.warning("No 'transcription' column found, skipping translation.")
            self.file_changed = False
            return df

        if "translation" not in df.columns:
            df["translation"] = ""
        df["translation"] = df["translation"].astype(object)

        progress_cb = getattr(self, "_progress_callback", None)
        had_examples, todo_items = self._separate_examples_and_todo(df, "transcription", "translation", "translation")

        if had_examples or not todo_items:
            self.file_changed = False
            return df

        if isinstance(self.strategy, LLMTranslationStrategy):
            id_to_translation = self._call_with_llm(self.strategy.translate, todo_items, 'translation', progress_cb)
            for i, translation in id_to_translation.items():
                df.at[i, "translation"] = translation
        else:
            total = len(todo_items)
            for done, item in enumerate(tqdm(todo_items, desc="Translating"), 1):
                try:
                    df.at[item["id"], "translation"] = self.strategy.translate(item["text"])
                except Exception as e:
                    self.logger.error(f"Error translating row {item['id']}: {e}")
                if progress_cb:
                    progress_cb(done, total)

        return df
