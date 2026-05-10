"""
Labvanced transliteration processor.

Converts non-Latin script transcriptions into their romanised equivalents.
Only applies to languages in the ``NO_LATIN`` set; calling this processor
for a Latin-script language is a configuration error that raises ``ValueError``.

Instruction modes
-----------------
- ``"sentences"``  — romanise utterance-level column only.
- ``"corrected"``  — romanise the full corrected transcription column.
"""
import pandas as pd
from tqdm import tqdm

from utils.functions import (
    set_global_variables,
)
from inference.processors.labvanced.labvanced_base import LabvancedBaseProcessor

LANGUAGES, NO_LATIN, OBLIGATORY_COLUMNS = set_global_variables()


class TransliteratorProcessor(LabvancedBaseProcessor):
    """Romanises non-Latin transcription columns in Labvanced annotated sheets."""

    def __init__(self, language: str, instruction: str, transliterationModel: str = None, device: str = "cpu"):
        super().__init__(language, instruction, action="transliterate", transliterationModel=transliterationModel, device=device)
        self.logger.info(f"Initialized transliteration strategy: {self.strategy.__class__.__name__} for language: {self.language}")
        self.columns_to_highlight = (
            "latin_transcription_utterance_used"
            if self.instruction == "sentences"
            else "latin_transcription_everything"
        )

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transliterate every unique non-null sentence in the source column.

        Unique sentences are grouped so that each distinct string is transliterated
        once and the result is written to all rows sharing that sentence, avoiding
        redundant API/model calls.  Bytes are decoded to UTF-8 before and after
        transliteration to handle binary-typed cells from certain Excel parsers.
        """
        if self.instruction == "sentences":
            source_col = "transcription_original_script_utterance_used"
            target_col = "latin_transcription_utterance_used"
        elif self.instruction == "corrected":
            source_col = "transcription_original_script"
            target_col = "latin_transcription_everything"
        else:
            raise ValueError(f"Unsupported instruction: {self.instruction}")

        if target_col not in df.columns:
            df[target_col] = ""
        df[target_col] = df[target_col].astype(str).replace('nan', '')

        progress_cb = getattr(self, '_progress_callback', None)

        had_examples, todo_items = self._separate_examples_and_todo(df, source_col, target_col, "transliteration")

        if had_examples or not todo_items:
            self.file_changed = False
            return df

        id_to_translation = self.strategy.transliterate(todo_items, self._get_examples(), progress_cb)
        for i in range(len(df)):
            if i in id_to_translation:
                df.at[i, target_col] = id_to_translation[i]

        return df