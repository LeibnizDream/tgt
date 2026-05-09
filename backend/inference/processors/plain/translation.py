
import pandas as pd
from tqdm import tqdm
from typing import Dict, List

from inference.processors.plain.plain_base import BasePlainProcessor
from utils.llm_functions import call_llm_batch
from inference.strategies.translation.llm import LLMTranslationStrategy


class PlainTranslator(BasePlainProcessor):

    _shared_examples: Dict[int, Dict] = {}
    _shared_index: int = 0

    def __init__(self, language: str, instruction: str, translationModel: str = None, device: str | None = None):
        super().__init__(language, instruction, action="translate", translationModel=translationModel, device=device)
        self.logger.info(f"Initialized translation strategy: {self.strategy.__class__.__name__}")

    @classmethod
    def reset_examples(cls):
        cls._shared_examples = {}
        cls._shared_index = 0

    def _separate_examples_and_todo(self, df: pd.DataFrame) -> tuple[bool, List[Dict]]:
        """Accumulate already-translated rows as examples; return (had_examples, todo_items).

        If any row already has a translation, the whole file is skipped — those rows
        are harvested as few-shot examples for other files."""
        had_examples = False
        todo = []
        for i, row in df.iterrows():
            source = row.get("transcription", "")
            existing = row.get("translation", "")
            if not isinstance(source, str) or not source.strip():
                continue
            if isinstance(existing, str) and existing.strip():
                PlainTranslator._shared_examples[PlainTranslator._shared_index] = {
                    "source": source,
                    "translation": existing,
                }
                PlainTranslator._shared_index += 1
                had_examples = True
            else:
                todo.append({"id": i, "text": source})
        return had_examples, todo

    def _translate_with_llm(self, todo_items: List[Dict], progress_cb=None) -> Dict[int, str]:
        return call_llm_batch(self.strategy.translate, todo_items, PlainTranslator._shared_examples, 'translation', progress_cb)

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if "transcription" not in df.columns:
            self.logger.warning("No 'transcription' column found, skipping translation.")
            self.file_changed = False
            return df

        if "translation" not in df.columns:
            df["translation"] = ""
        df["translation"] = df["translation"].astype(object)

        progress_cb = getattr(self, "_progress_callback", None)
        had_examples, todo_items = self._separate_examples_and_todo(df)

        # If any row was already translated, harvest those as examples and skip the file
        if had_examples or not todo_items:
            self.file_changed = False
            return df

        if isinstance(self.strategy, LLMTranslationStrategy):
            id_to_translation = self._translate_with_llm(todo_items, progress_cb)
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
