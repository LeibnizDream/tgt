"""
Plain glossing processor.

Reads ``transcribed.xlsx`` files and fills the ``glossing`` column with
interlinear Leipzig glosses derived from the ``to_gloss`` column.  Skips
files that already contain any human gloss to avoid overwriting corrections.
"""
import pandas as pd
from tqdm import tqdm

from inference.processors.plain.plain_base import BasePlainProcessor
from inference.strategies.glossing.llm import LLMGlossingStrategy


class PlainGlosser(BasePlainProcessor):
    """Glosses the ``to_gloss`` column and writes results to ``glossing``."""

    def __init__(self, language: str, instruction: str, glossingModel: str = None, translationModel: str = None, device: str | None = None):
        super().__init__(language, instruction, action="gloss", glossingModel=glossingModel, translationModel=translationModel, device=device)
        self.logger.info(f"Initialized glossing strategy: {self.strategy.__class__.__name__}")

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gloss pending rows from ``to_gloss`` into ``glossing``.

        Returns *df* unchanged (with ``file_changed=False``) when the
        ``to_gloss`` column is absent or any row already has a gloss.
        """
        if "to_gloss" not in df.columns:
            self.logger.warning("No 'to_gloss' column found, skipping glossing.")
            self.file_changed = False
            return df

        if "glossing" not in df.columns:
            df["glossing"] = ""
        df["glossing"] = df["glossing"].astype(object)

        progress_cb = getattr(self, "_progress_callback", None)
        had_examples, todo_items = self._separate_examples_and_todo(df, "to_gloss", "glossing", "gloss")

        if had_examples or not todo_items:
            self.file_changed = False
            return df

        if isinstance(self.strategy, LLMGlossingStrategy):
            id_to_gloss = self._call_with_llm(self.strategy.gloss, todo_items, 'gloss', progress_cb)
            for i, gloss in id_to_gloss.items():
                df.at[i, "glossing"] = gloss
        else:
            total = len(todo_items)
            for done, item in enumerate(tqdm(todo_items, desc="Glossing"), 1):
                try:
                    df.at[item["id"], "glossing"] = self.strategy.gloss(item["text"])
                except Exception as e:
                    self.logger.error(f"Error glossing row {item['id']}: {e}")
                if progress_cb:
                    progress_cb(done, total)

        return df
