import warnings

import pandas as pd
from tqdm import tqdm

from inference.processors.labvanced.labvanced_base import LabvancedBaseProcessor
from inference.strategies.translation.llm import LLMTranslationStrategy
from utils.functions import set_global_variables, find_ffmpeg


LANGUAGES, NO_LATIN, OBLIGATORY_COLUMNS = set_global_variables()
warnings.filterwarnings("ignore")
ffmpeg_path = find_ffmpeg()


class TranslationProcessor(LabvancedBaseProcessor):
    """
    Processes annotated Excel files by translating text in specified columns.
    """

    def __init__(
        self,
        language: str,
        instruction: str,
        translationModel: str = None,
        device: str = "cpu",
    ):
        super().__init__(language, instruction, action="translate", translationModel=translationModel, device=device)
        print(f"Using translation strategy: {self.strategy.__class__.__name__} for language: {self.language}")
        self.columns_to_highlight = {
            "automatic": "automatic_translation_automatic_transcription",
            "corrected": "translation_everything",
            "sentences": "translation_utterance_used",
        }.get(self.instruction)

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

    def _get_target_columns(self) -> list[str]:
        return {
            "corrected": ["automatic_translation_corrected_transcription", "translation_everything"],
            "automatic": ["automatic_translation_automatic_transcription"],
            "sentences": ["automatic_translation_utterance_used", "translation_utterance_used"],
        }[self.instruction]

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

        had_examples, todo_items = self._separate_examples_and_todo(df, source_col, self._get_target_columns()[-1], "translation")

        if had_examples or not todo_items:
            self.file_changed = False
            return df

        if isinstance(self.strategy, LLMTranslationStrategy):
            id_to_translation = self._call_with_llm(self.strategy.translate, todo_items, 'translation', progress_cb)
            for i in range(len(df)):
                if i in id_to_translation:
                    for col in target_cols:
                        df.at[i, col] = id_to_translation[i]

        else:
            total = len(todo_items)
            for done, item in enumerate(tqdm(todo_items, desc="Translating rows"), 1):
                try:
                    translation = self.strategy.translate(item["text"])
                    for col in target_cols:
                        df.at[item["id"], col] = translation
                except Exception as e:
                    self.logger.error(f"Error translating row {item['id']}: {e}")
                if progress_cb:
                    progress_cb(done, total)

        return df