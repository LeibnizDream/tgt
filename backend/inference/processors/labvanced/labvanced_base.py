"""
Labvanced-specific base processor.

Implements the three I/O hooks (_find_files, _read_file, _write_file) for the
Labvanced annotation pipeline, plus the shared _process_dataframe skeleton for
translation, glossing, and transliteration.

Column mappings (source, target) are defined here by (action, instruction)
so that concrete subclasses only need a minimal __init__ naming the action.
TranscriptionProcessor overrides _process_dataframe for its audio-file pattern.
"""
import os
import fnmatch
import pandas as pd
from utils.functions import set_global_variables, format_excel_output
from inference.processors.abstract_processor import AbstractProcessor

LANGUAGES, NO_LATIN, OBLIGATORY_COLUMNS = set_global_variables()


class LabvancedBaseProcessor(AbstractProcessor):
    """Base class for all Labvanced processors.

    Concrete subclasses only override __init__ to supply the action name.
    TranscriptionProcessor also overrides _process_dataframe.
    """

    def __init__(self, language, action, instruction, model=None):
        super().__init__(language, action, model)
        self.instruction = instruction
        self.columns_to_highlight = None

    def _find_files(self, base_dir: str) -> list[str]:
        """Return sorted paths to every trials_and_sessions_annotated.xlsx under base_dir."""
        matches = []
        for root, _, files in os.walk(base_dir):
            for f in files:
                if fnmatch.fnmatch(f, 'trials_and_sessions_annotated.xlsx'):
                    matches.append(os.path.join(root, f))
        self.logger.info(f"Found {len(matches)} matching files in {base_dir}")
        return sorted(matches)

    def _write_file(self, path: str, df: pd.DataFrame) -> None:
        """Write df back to path, placing obligatory columns last, with optional formatting."""
        extra_cols = [c for c in df.columns if c not in OBLIGATORY_COLUMNS]
        df = df[extra_cols + [c for c in OBLIGATORY_COLUMNS if c in df.columns]]
        df.to_excel(path, index=False)
        self.logger.info(f"Wrote output to {path}")
        if self.columns_to_highlight:
            format_excel_output(path, self.columns_to_highlight)

    def _get_source_column(self) -> str:
        no_latin = self.language in NO_LATIN
        if self.action in ("translate", "gloss"):
            return {
                "corrected": "transcription_original_script" if no_latin else "latin_transcription_everything",
                "automatic": "automatic_transcription",
                "sentences": "transcription_original_script_utterance_used" if no_latin else "latin_transcription_utterance_used",
            }[self.instruction]
        if self.action == "transliterate":
            return {
                "sentences": "transcription_original_script_utterance_used",
                "corrected":  "transcription_original_script",
            }[self.instruction]
        raise ValueError(f"No source column for action={self.action!r}, instruction={self.instruction!r}")

    def _get_target_columns(self) -> list[str]:
        if self.action == "translate":
            return {
                "corrected": ["automatic_translation_corrected_transcription", "translation_everything"],
                "automatic": ["automatic_translation_automatic_transcription"],
                "sentences": ["automatic_translation_utterance_used", "translation_utterance_used"],
            }[self.instruction]
        if self.action == "gloss":
            return ["automatic_glossing", "glossing_utterance_used"]
        if self.action == "transliterate":
            return {
                "sentences": ["latin_transcription_utterance_used"],
                "corrected":  ["latin_transcription_everything"],
            }[self.instruction]
        raise ValueError(f"No target columns for action={self.action!r}, instruction={self.instruction!r}")

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Translate/gloss/transliterate pending rows and write results to target columns."""
        source_col = self._get_source_column()
        target_cols = self._get_target_columns()
        self.columns_to_highlight = target_cols[-1]

        if source_col not in df.columns:
            return df

        for col in target_cols:
            if col not in df.columns:
                df[col] = pd.NA
            df[col] = df[col].astype(object)

        progress_cb = getattr(self, '_progress_callback', None)
        had_examples, todo_items = self._separate_examples_and_todo(
            df, source_col, target_cols[-1], self.action
        )

        if had_examples or not todo_items:
            self.file_changed = False
            return df

        id_to_result = self.strategy.run_strategy(todo_items, self._get_examples(), progress_cb)
        for i in range(len(df)):
            if i in id_to_result:
                for col in target_cols:
                    df.at[i, col] = id_to_result[i]

        return df
