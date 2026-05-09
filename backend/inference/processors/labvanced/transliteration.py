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

    def __init__(self, language: str, instruction: str, device: str = "cpu"):
        super().__init__(language, instruction, action="transliterate", device=device)
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
            source = "transcription_original_script_utterance_used"
            target = "latin_transcription_utterance_used"
        elif self.instruction == "corrected":
            source = "transcription_original_script"
            target = "latin_transcription_everything"
        else:
            raise ValueError(f"Unsupported instruction: {self.instruction}")

        if target not in df.columns:
            df[target] = ""
        df[target] = df[target].astype(str).replace('nan', '')

        for sentence in tqdm(
            df[source].dropna(), desc="Transliterating sentences", leave=False
        ):
            if isinstance(sentence, bytes):
                sentence = sentence.decode('utf-8')

            hits = df[df[source] == sentence].index
            translit = self.strategy.transliterate(sentence)
            if isinstance(translit, bytes):
                translit = translit.decode('utf-8')

            for idx in hits:
                df.at[idx, target] = translit

        return df