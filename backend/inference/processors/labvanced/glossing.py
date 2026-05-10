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



from utils.functions import set_global_variables
from inference.processors.labvanced.labvanced_base import LabvancedBaseProcessor


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

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        source_col = self._get_source_column()

        if source_col not in df.columns:
            return df

        had_examples, todo_items = self._separate_examples_and_todo(df, source_col, "glossing_utterance_used", "gloss")
        if had_examples or not todo_items:
            self.file_changed = False
            return df

        progress_cb = getattr(self, '_progress_callback', None)
        id_to_gloss = self.strategy.gloss(todo_items, self._get_examples(), progress_cb)

        glossed = []
        for i in range(len(df)):
            if i in id_to_gloss:
                glossed.append(id_to_gloss[i])
            else:
                existing = df.at[i, "glossing_utterance_used"]
                glossed.append(existing if isinstance(existing, str) else "")

        df["automatic_glossing"] = glossed
        df["glossing_utterance_used"] = glossed
        return df