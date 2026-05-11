"""
Labvanced-specific base processor.

Implements the three I/O hooks (_find_files, _read_file, _write_file) for the
Labvanced annotation pipeline, plus the shared _process_dataframe skeleton for
translation, glossing, and transliteration.

Column mappings (source, target) are defined here by (action, instruction)
so that concrete subclasses only need a minimal __init__ naming the action.
TranscriptionProcessor overrides _process_dataframe for its audio-file pattern.
"""
import fnmatch
import os

import pandas as pd
from inference.processors.abstract_processor import AbstractProcessor
from inference.strategies.strategy_factory import StrategyFactory
from utils.functions import format_excel_output, set_global_variables

LANGUAGES, NO_LATIN, OBLIGATORY_COLUMNS = set_global_variables()


class LabvancedTextProcessor(AbstractProcessor):
    """Base class for all Labvanced processors.

    Concrete subclasses only override __init__ to supply the action name.
    TranscriptionProcessor also overrides _process_dataframe.
    """

    def __init__(self, language, action, instruction, model=None):
        super().__init__(language, action, model)
        self.instruction = instruction
        self.columns_to_highlight = None
        self.strategy = StrategyFactory.get_strategy(language, action, model)

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
            format_excel_output(path, self.columns_to_highlight, getattr(self, '_todo_row_ids', None))

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

        progress_cb = self._progress_callback
        todo_items = self._get_todo(
            df, source_col, target_cols[-1], self.action
        )
        print('to do items:', todo_items)

        if not todo_items:
            self.file_changed = False
            self._todo_row_ids = set()
            return df

        self._todo_row_ids = {item["id"] for item in todo_items}
        id_to_result = {}
        if self.model in ["gemini", "qwen"]:
            id_to_result = self.strategy.run_strategy(todo_items, self._get_examples())
        else:
            total = len(todo_items)
            remaining = total
            for todo in todo_items:
                id_to_result[todo["id"]] = self.strategy.run_strategy(todo["text"])
                remaining -= 1
                if progress_cb:
                    progress_cb(remaining, total)
        for i in range(len(df)):
            if i in id_to_result:
                for col in target_cols:
                    df.at[i, col] = id_to_result[i]

        return df
