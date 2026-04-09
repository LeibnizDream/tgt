import warnings
import json
import pandas as pd
from tqdm import tqdm
from typing import Dict, List

from inference.translation.factory import TranslationStrategyFactory
from inference.translation.abstract import TranslationStrategy
from inference.translation.llm import LLMTranslationStrategy  # add this import
from utils.functions import set_global_variables, find_ffmpeg
from inference.processors.abstract import DataProcessor

LANGUAGES, NO_LATIN, OBLIGATORY_COLUMNS = set_global_variables()
warnings.filterwarnings("ignore")
ffmpeg_path = find_ffmpeg()


class TranslationProcessor(DataProcessor):
    """
    Processes annotated Excel files by translating text in specified columns.
    """

    _shared_examples: Dict[int, Dict] = {}
    _shared_index: int = 0

    def __init__(
        self,
        language: str,
        instruction: str,
        translationModel: str = None,
        device: str = "cpu",
    ):
        super().__init__(language, instruction)
        self.device = device
        self.strategy: TranslationStrategy = TranslationStrategyFactory.get_strategy(self.language, translationModel)
        print(f"Using translation strategy: {self.strategy.__class__.__name__} for language: {self.language}")
        self.columns_to_highlight = {
            "automatic": "automatic_translation_automatic_transcription",
            "corrected": "translation_everything",
            "sentences": "translation_utterance_used",
        }.get(self.instruction)

    @classmethod
    def reset_examples(cls):
        cls._shared_examples = {}
        cls._shared_index = 0

    def _get_source_column(self) -> str:
        auto_col = "automatic_transcription"
        corr_col = "latin_transcription_everything"
        sent_col = "latin_transcription_utterance_used"
        if self.language in NO_LATIN:
            corr_col = "transcription_original_script"
            sent_col = "transcription_original_script_utterance_used"

        return {
            "corrected": corr_col,
            "automatic": auto_col,
            "sentences": sent_col,
        }.get(self.instruction, sent_col)

    def _get_target_columns(self) -> List[str]:
        return {
            "corrected": ["automatic_translation_corrected_transcription", "translation_everything"],
            "automatic": ["automatic_translation_automatic_transcription"],
            "sentences": ["automatic_translation_utterance_used", "translation_utterance_used"],
        }[self.instruction]

    def _get_primary_target_column(self) -> str:
        """The column used as the example reference (first human-edited target)."""
        return self._get_target_columns()[-1]  # last col is the canonical/human one

    def _separate_examples_and_todo(
        self,
        df: pd.DataFrame,
        source_col: str,
    ) -> List[Dict]:
        """
        Rows with an existing translation in the primary target column go into
        _shared_examples for few-shot context. Rows without go into todo_items.
        """
        primary_col = self._get_primary_target_column()
        todo_items = []

        for i in range(len(df)):
            source_text = df.at[i, source_col]
            existing_translation = df.at[i, primary_col] if primary_col in df.columns else None

            if not isinstance(source_text, str) or not source_text.strip():
                continue

            if isinstance(existing_translation, str) and existing_translation.strip():
                TranslationProcessor._shared_examples[TranslationProcessor._shared_index] = {
                    "source": source_text,
                    "translation": existing_translation,
                }
                TranslationProcessor._shared_index += 1
            else:
                todo_items.append({"id": i, "text": source_text})

        return todo_items

    def _translate_with_llm(self, todo_items: List[Dict]) -> Dict[int, str]:
        """
        Send batch payload with accumulated few-shot examples to Gemini strategy.
        Returns {row_index: translation_string}.
        """
        if not todo_items:
            return {}

        examples = list(TranslationProcessor._shared_examples.values())[:10]

        payload = {
            "examples": examples,
            "items": todo_items,
        }

        response_text = self.strategy.translate(json.dumps(payload, ensure_ascii=False))
        response_json = json.loads(response_text)

        return {item["id"]: item["translation"] for item in response_json["items"]}

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        source_col = self._get_source_column()
        target_cols = self._get_target_columns()

        if source_col not in df.columns:
            return df

        # Ensure target columns exist with object dtype
        for col in target_cols:
            if col not in df.columns:
                df[col] = pd.NA
            df[col] = df[col].astype(object)

        print(f"Source column for {self.instruction}: {source_col}")

        progress_cb = getattr(self, '_progress_callback', None)

        if isinstance(self.strategy, GeminiTranslationStrategy):
            todo_items = self._separate_examples_and_todo(df, source_col)

            if not todo_items:
                self.file_changed = False
                return df

            if progress_cb:
                progress_cb(0, len(todo_items))

            id_to_translation = self._translate_with_llm(todo_items)

            if progress_cb:
                progress_cb(len(todo_items), len(todo_items))

            for i in range(len(df)):
                if i in id_to_translation:
                    for col in target_cols:
                        df.at[i, col] = id_to_translation[i]

        else:
            # Fallback: row-by-row for non-Gemini strategies
            eligible = [
                (idx, row) for idx, row in df.iterrows()
                if idx < 100 and not pd.isna(row.get(source_col)) and str(row.get(source_col, "")).strip()
            ]
            total = len(eligible)
            done = 0
            for idx, row in tqdm(df.iterrows(), desc="Translating rows"):
                if idx >= 100:
                    self.logger.info(f"Reached max rows at {idx}")
                    break
                text = row.get(source_col)
                if pd.isna(text) or not str(text).strip():
                    continue

                translation = self.strategy.translate(str(text))
                for col in target_cols:
                    df.at[idx, col] = translation
                done += 1
                if progress_cb:
                    progress_cb(done, total)

        return df