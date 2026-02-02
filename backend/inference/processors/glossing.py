import pandas as pd
from tqdm import tqdm
import json
from typing import Dict, List, Tuple
from collections import deque

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
    _shared_examples = {}  # Class variable
    _shared_index = 0
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
    ) -> List[Dict[int, str]]:
        """
        Separate rows into examples (stored in class-level _shared_examples) and todo_items.
        Only returns items that need glossing.
        """
        todo_items = []
        
        for i in range(len(df)):
            source_text = df.at[i, source_col]
            existing_gloss = df.at[i, "glossing_utterance_used"]
            
            if not isinstance(source_text, str) or not source_text.strip():
                continue
            
            if isinstance(existing_gloss, str) and existing_gloss.strip():
                GlossingProcessor._shared_examples[GlossingProcessor._shared_index] = {
                    'source': source_text,
                    'gloss': existing_gloss
                }
                GlossingProcessor._shared_index += 1
            else:
                todo_items.append({
                    'id': i,
                    'text': source_text
                })
        
        return todo_items

    def _gloss_with_llm(
        self, 
        todo_items: List[Dict[int, str]]
    ) -> Dict[int, str]:
        """
        Send batch request with accumulated examples for few-shot learning.
        """
        if not todo_items:
            return {}
        
        examples = list(GlossingProcessor._shared_examples.values())[-10:]
        
        payload = {
            'examples': examples,
            'items': todo_items
        }
        
        response_text = self.strategy.gloss(json.dumps(payload, ensure_ascii=False))
        response_json = json.loads(response_text)
        
        return {item['id']: item['gloss'] for item in response_json['items']}

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main processing logic: gloss the dataframe using appropriate strategy.
        """
        source_col = self._get_source_column()
        
        if source_col not in df.columns:
            return df
        
        todo_items = self._separate_examples_and_todo(df, source_col)
        if not todo_items:
            self.file_changed = False
            return df
        
        if isinstance(self.strategy, (GeminiGlossingStrategy, QwenGlossingStrategy)):
            id_to_gloss = self._gloss_with_llm(todo_items)
            
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
            glossed_series = self._gloss_with_standard_strategy(df, source_col)
            if glossed_series.empty:
                self.file_changed = False
            df["automatic_glossing"] = glossed_series
            df["glossing_utterance_used"] = glossed_series
        
        return df