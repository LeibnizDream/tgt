"""
Labvanced translation processor.

Translates a source transcription column into one or two target columns
depending on the *instruction* mode:

- ``"sentences"``  — utterance-level source → automatic + human-editable target.
- ``"corrected"``  — hand-corrected full transcription → two translation targets.
- ``"automatic"``  — raw automatic transcription → one translation target.

LLM strategies process all rows in a single batch call via
:meth:`AbstractProcessor._call_with_llm`.  Non-LLM strategies translate
row-by-row with tqdm progress reporting.
"""
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
    """Translates a source transcription column and writes results into the Labvanced annotated sheet."""

    def __init__(
        self,
        language: str,
        instruction: str,
        translationModel: str = None,
        device: str = "cpu",
    ):
        super().__init__(language, instruction, action="translate", translationModel=translationModel, device=device)
        self.logger.info(f"Using translation strategy: {self.strategy.__class__.__name__} for language: {self.language}")
        self.columns_to_highlight = {
            "automatic": "automatic_translation_automatic_transcription",
            "corrected": "translation_everything",
            "sentences": "translation_utterance_used",
        }.get(self.instruction)

    def _get_source_column(self) -> str:
        """Return the column to translate from, accounting for non-Latin scripts."""
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
        """Return the column(s) to write translations into for the current instruction mode."""
        return {
            "corrected": ["automatic_translation_corrected_transcription", "translation_everything"],
            "automatic": ["automatic_translation_automatic_transcription"],
            "sentences": ["automatic_translation_utterance_used", "translation_utterance_used"],
        }[self.instruction]

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Translate pending rows and write results to all target columns.

        Skips the file entirely when any row already has a human translation
        (to avoid overwriting corrections).  The last target column is the
        canonical one checked for existing translations.
        """
        source_col = self._get_source_column()
        target_cols = self._get_target_columns()

        if source_col not in df.columns:
            return df

        for col in target_cols:
            if col not in df.columns:
                df[col] = pd.NA
            df[col] = df[col].astype(object)

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