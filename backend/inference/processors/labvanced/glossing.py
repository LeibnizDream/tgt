"""
Glossing processor: applies a glossing strategy to annotated Excel files.

:class:`GlossingProcessor` is a :class:`~inference.processors.abstract.DataProcessor`
that uses a :class:`~inference.glossing.abstract.GlossingStrategy` (selected
by :class:`~inference.glossing.factory.GlossingStrategyFactory`) to assign
interlinear Leipzig glosses to transcription utterances.

Few-shot learning (LLM strategies)
-----------------------------------
When an LLM-backed strategy is used (e.g. Gemini or Qwen), rows that already
contain a gloss serve as few-shot examples.  These examples are accumulated in
the class-level :attr:`_shared_examples` dict so that examples from earlier
files in the same job are available when later files are processed.
Call :meth:`GlossingProcessor.reset_examples` between jobs to clear the cache.

Standard strategies (spaCy / Stanza)
--------------------------------------
Non-LLM strategies process rows one-by-one, skipping rows that already have a
gloss, and report progress via the optional progress callback.
"""
import pandas as pd
from tqdm import tqdm



from utils.functions import set_global_variables
from inference.processors.labvanced.labvanced_base import LabvancedBaseProcessor
from inference.strategies.glossing.llm import LLMGlossingStrategy


LANGUAGES, NO_LATIN, OBLIGATORY_COLUMNS = set_global_variables()


class GlossingProcessor(LabvancedBaseProcessor):
    """
    Concrete DataProcessor that applies a GlossingStrategy
    to each '*.annotated.xlsx' file under input_dir.
    """
    def __init__(
        self,
        language: str,
        instruction: str,
        translation_model: str | None = None,
        glossing_model: str | None = None
    ):
        super().__init__(language=language, instruction=instruction, action="gloss",
                         translationModel=translation_model, glossingModel=glossing_model)
        self.columns_to_highlight = ["glossing_utterance_used"]

    def _get_source_column(self) -> str:
        """Determine which column to gloss based on instruction."""
        if self.instruction == "sentences":
            return ("transcription_original_script_utterance_used" 
                    if self.language in NO_LATIN 
                    else "latin_transcription_utterance_used")
        elif self.instruction == "corrected":
            return ("transcription_original_script" 
                    if self.language in NO_LATIN 
                    else "latin_transcription_everything")
        elif self.instruction == "automatic":
            return "automatic_transcription"
        else:
            raise ValueError(f"Unsupported instruction: {self.instruction!r}")

    def _gloss_with_standard_strategy(
        self,
        df: pd.DataFrame,
        source_col: str,
        progress_cb=None,
    ) -> pd.Series:
        """
        Apply the strategy row-by-row for non-LLM strategies (e.g. Stanza, SpaCy).
        Rows with existing glosses or empty source are left as-is.
        """
        todo_indices = [
            i for i in range(len(df))
            if not (isinstance(df.at[i, "glossing_utterance_used"], str) and df.at[i, "glossing_utterance_used"].strip())
            and isinstance(df.at[i, source_col], str) and df.at[i, source_col].strip()
        ]
        total = len(todo_indices)
        done = 0

        results = []
        for i in range(len(df)):
            source_text = df.at[i, source_col]
            existing_gloss = df.at[i, "glossing_utterance_used"]

            if isinstance(existing_gloss, str) and existing_gloss.strip():
                results.append(existing_gloss)
            elif isinstance(source_text, str) and source_text.strip():
                results.append(self.strategy.gloss(source_text))
                done += 1
                if progress_cb:
                    progress_cb(done, total)
            else:
                results.append("")

        return pd.Series(results, index=df.index)

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        source_col = self._get_source_column()

        if source_col not in df.columns:
            return df

        had_examples, todo_items = self._separate_examples_and_todo(df, source_col, "glossing_utterance_used", "gloss")
        if had_examples or not todo_items:
            self.file_changed = False
            return df

        progress_cb = getattr(self, '_progress_callback', None)

        if isinstance(self.strategy, LLMGlossingStrategy):
            id_to_gloss = self._call_with_llm(self.strategy.gloss, todo_items, 'gloss', progress_cb)

            glossed = []
            for i in range(len(df)):
                if i in id_to_gloss:
                    glossed.append(id_to_gloss[i])
                else:
                    existing = df.at[i, "glossing_utterance_used"]
                    glossed.append(existing if isinstance(existing, str) else "")

            df["automatic_glossing"] = glossed
            df["glossing_utterance_used"] = glossed
        else:
            glossed_series = self._gloss_with_standard_strategy(df, source_col, progress_cb)
            if glossed_series.empty:
                self.file_changed = False
            df["automatic_glossing"] = glossed_series
            df["glossing_utterance_used"] = glossed_series

        return df