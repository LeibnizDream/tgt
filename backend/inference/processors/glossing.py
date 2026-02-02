import pandas as pd
from tqdm import tqdm
import json
from typing import Dict, List, Tuple
from collections import defaultdict

from utils.functions import set_global_variables
from inference.glossing.factory import GlossingStrategyFactory
from inference.glossing.abstract import GlossingStrategy
from inference.glossing.gemini import GeminiGlossingStrategy
from inference.glossing.qwen import QwenGlossingStrategy
from inference.processors.abstract import DataProcessor

LANGUAGES, NO_LATIN, OBLIGATORY_COLUMNS = set_global_variables()


class GlossingProcessor(DataProcessor):
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
        super().__init__(language=language, instruction=instruction)
        self.glossing_model = glossing_model
        self.translation_model = translation_model
        self.strategy: GlossingStrategy = GlossingStrategyFactory.get_strategy(
            self.language, self.glossing_model, self.translation_model
        )
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

    def _separate_examples_and_todo(
        self, 
        df: pd.DataFrame, 
        source_col: str
    ) -> Tuple[Dict[int, str], List[Dict[int, str]]]:
        """
        Separate rows into:
        - examples: rows that already have glosses (for few-shot prompting)
        - todo_items: rows that need glossing
        
        Returns:
            Tuple of (examples_dict, todo_items_list)
        """
        examples = {}
        todo_items = []
        
        for i in range(len(df)):
            source_text = df.at[i, source_col]
            existing_gloss = df.at[i, "glossing_utterance_used"]
            
            # If source text is empty, skip
            if not isinstance(source_text, str) or not source_text.strip():
                continue
            
            # If existing gloss is present, use as example
            if isinstance(existing_gloss, str) and existing_gloss.strip():
                examples[i] = {
                    'source': source_text,
                    'gloss': existing_gloss
                }
            else:
                # Needs glossing
                todo_items.append({
                    'id': i,
                    'text': source_text
                })
        
        return examples, todo_items

    def _gloss_with_llm(
        self, 
        examples: Dict[int, Dict[str, str]],  # Fixed type hint
        todo_items: List[Dict[int, str]]
    ) -> Dict[int, str]:
        """
        Send batch request to Gemini with examples for few-shot learning.
        
        Returns:
            Dict mapping row indices to glossed text
        """
        if not todo_items:
            return {}
        
        # Prepare payload with examples for few-shot prompting
        payload = {
            'examples': list(examples.values()),
            'items': todo_items  # Items to gloss
        }
        
        # Single Gemini API call
        response_text = self.strategy.gloss(json.dumps(payload, ensure_ascii=False))
        response_json = json.loads(response_text)
        
        # Map results back to row indices
        return {item['id']: item['gloss'] for item in response_json['items']}

    def _gloss_with_standard_strategy(
        self, 
        df: pd.DataFrame, 
        source_col: str
    ) -> pd.Series:
        """
        Process rows one-by-one with standard (non-Gemini) strategy.
        """
        glossed = []
        
        for i in tqdm(range(len(df)), desc="Glossing rows", unit="row"):
            source_text = df.at[i, source_col]
            existing_gloss = df.at[i, "glossing_utterance_used"]
            
            # Keep existing gloss if present
            if isinstance(existing_gloss, str) and existing_gloss.strip():
                glossed.append(existing_gloss)
            # Gloss each line separately if source exists
            elif isinstance(source_text, str) and source_text.strip():
                lines = source_text.split("\n")
                glossed_lines = [self.strategy.gloss(line) for line in lines]
                glossed.append("\n".join(glossed_lines))
            else:
                glossed.append("")
        
        return pd.Series(glossed, index=df.index)

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main processing logic: gloss the dataframe using appropriate strategy.
        """
        source_col = self._get_source_column()
        
        if source_col not in df.columns:
            return df
        
        # Separate into examples and items needing glossing
        examples, todo_items = self._separate_examples_and_todo(df, source_col)
        if not todo_items:
            self.file_changed = False
            return df
        
        # Process based on strategy type
        if isinstance(self.strategy, GeminiGlossingStrategy) or isinstance(self.strategy, QwenGlossingStrategy):
            id_to_gloss = self._gloss_with_llm(examples, todo_items)
            
            # Merge results back into dataframe
            glossed = []
            for i in range(len(df)):
                if i in id_to_gloss:
                    glossed.append(id_to_gloss[i])
                else:
                    # Keep existing gloss or empty string
                    existing = df.at[i, "glossing_utterance_used"]
                    glossed.append(existing if isinstance(existing, str) else "")
            
            df["automatic_glossing"] = glossed
            df["glossing_utterance_used"] = glossed
        else:
            # Row-by-row processing with standard strategy
            glossed_series = self._gloss_with_standard_strategy(df, source_col)
            if glossed_series.empty:
                self.file_changed = False
            df["automatic_glossing"] = glossed_series
            df["glossing_utterance_used"] = glossed_series
        
        return df