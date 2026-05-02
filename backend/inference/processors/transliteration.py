import os
import pandas as pd
from tqdm import tqdm
import json

from utils.functions import (
    set_global_variables,
)
from inference.transliteration.abstract import TransliterationStrategy
from inference.transliteration.factory import TransliterationStrategyFactory
from inference.transliteration.llm import LLMTransliterationStrategy

from inference.processors.abstract import DataProcessor

LANGUAGES, NO_LATIN, OBLIGATORY_COLUMNS = set_global_variables()


class TransliteratorProcessor(DataProcessor):
    """
    Processes all '*.annotated.xlsx' files in a directory,
    applying a TransliterationStrategy to each DataFrame.
    """
    _shared_examples = {}
    _shared_index = 0

    def __init__(self, language: str, instruction: str, transliterate_model: str = None, device: str = "cpu"):
        # initialize base with language & instruction
        super().__init__(language, instruction)
        self.device = device
        # pick strategy based on resolved language code
        print(f"Getting transliteration strategy for language: {self.language}")
        self.strategy: TransliterationStrategy = TransliterationStrategyFactory.get_strategy(
            self.language, transliterate_model
        )
        print('initialized transliteration strategy:', self.strategy.__class__.__name__)
        self.columns_to_highlight = (
            "latin_transcription_utterance_used"
            if self.instruction == "sentences"
            else "latin_transcription_everything"
        )

    @classmethod
    def reset_examples(cls):
        cls._shared_examples = {}
        cls._shared_index = 0

    def _separate_examples_and_todo(
            self,
            df: pd.DataFrame,
            source_col: str,
            target_col: str,
    ) -> list[dict]:
        todo_items = []
        for i in range(len(df)):
            source_text = df.at[i, source_col]
            existing = df.at[i, target_col]
            if not isinstance(source_text, str) or not source_text.strip():
                continue
            if isinstance(existing, str) and existing.strip():
                TransliteratorProcessor._shared_examples[TransliteratorProcessor._shared_index] = {
                    "source": source_text,
                    "transliteration": existing,
                }
                TransliteratorProcessor._shared_index += 1
            else:
                todo_items.append({"id": i, "text": source_text})
        return todo_items

    def _transliterate_with_llm(self, todo_items: list[dict]) -> dict[int, str]:
        if not todo_items:
            return {}
        examples = list(TransliteratorProcessor._shared_examples.values())[:10]
        payload = {"examples": examples, "items": todo_items}
        response_text = self.strategy.transliterate(json.dumps(payload, ensure_ascii=False))
        response_json = json.loads(response_text)
        return {item["id"]: item["transliteration"] for item in response_json["items"]}

    def _transliterate_with_standard_strategy(
            self,
            df: pd.DataFrame,
            source_col: str,
            target_col: str,
    ) -> pd.Series:
        results = []
        for i in range(len(df)):
            source_text = df.at[i, source_col]
            existing = df.at[i, target_col]
            if isinstance(existing, str) and existing.strip():
                results.append(existing)
            elif isinstance(source_text, str) and source_text.strip():
                translit = self.strategy.transliterate(source_text)
                if isinstance(translit, bytes):
                    translit = translit.decode("utf-8")
                results.append(translit)
            else:
                results.append("")
        return pd.Series(results, index=df.index)

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        # determine source/target columns by instruction
        if self.instruction == "sentences":
            source_col = "transcription_original_script_utterance_used"
            target_col = "latin_transcription_utterance_used"
        elif self.instruction == "corrected":
            source_col = "transcription_original_script"
            target_col = "latin_transcription_everything"
        else:
            raise ValueError(f"Unsupported instruction: {self.instruction}")

        # ensure target column exists with proper string dtype
        if target_col not in df.columns:
            df[target_col] = ""
        df[target_col] = df[target_col].astype(str).replace("nan", "")

        if isinstance(self.strategy, LLMTransliterationStrategy):
            todo_items = self._separate_examples_and_todo(df, source_col, target_col)
            if not todo_items:
                self.file_changed = False
                return df
            id_to_translit = self._transliterate_with_llm(todo_items)
            transliterated = []
            for i in range(len(df)):
                if i in id_to_translit:
                    transliterated.append(id_to_translit[i])
                else:
                    existing = df.at[i, target_col]
                    transliterated.append(existing if isinstance(existing, str) else "")
            df[target_col] = transliterated
        else:
            df[target_col] = self._transliterate_with_standard_strategy(df, source_col, target_col)

        return df
