import json
import sys
from typing import List

from pydantic import BaseModel

from inference.strategies.transliteration.abstract import TransliterationStrategy
from inference.strategies.llm_base import LLMStrategy


class TransliterationItem(BaseModel):
    id: int
    transliteration: str


class TransliterationResponse(BaseModel):
    items: List[TransliterationItem]


class LLMTransliterationStrategy(TransliterationStrategy, LLMStrategy):

    @property
    def _model_hint(self) -> str | None:
        return self.transliterationModel

    def load_model(self) -> None:
        self._init_llm()
        print(f"Loaded model for transliteration: {getattr(self, 'model_name', 'gemini')}", file=sys.stderr)

    def transliterate(self, items: list, examples: list = None, progress_cb=None) -> dict:
        examples = examples or []
        if 0 < len(examples) < 10:
            raise ValueError(
                f"Only {len(examples)} few-shot example(s) available — "
                "at least 10 are required for LLM processing."
            )
        response_json = self._call(items, examples)
        parsed = json.loads(response_json)
        result = {item["id"]: item["transliteration"] for item in parsed["items"]}
        if progress_cb:
            progress_cb(len(items), len(items))
        return result

    def _transliterate_one(self, text: str) -> str:
        raise Exception("Transliteration with LLM does not allow single transliteration")

    def _result_key(self) -> str:
        return "transliteration"

    def _response_model(self):
        return TransliterationResponse

    def _normalize_examples(self, examples: list) -> list:
        return [
            {"text": ex["source"], "transliteration": ex["transliteration"]}
            for ex in examples
            if ex.get("source") and ex.get("transliteration")
        ]

    def _build_system_prompt(self, include_schema_hint: bool = False) -> str:
        prompt = (
            f"You are a transliteration engine for {self.language_code}.\n"
            "Romanize each input text from its native script into Latin script.\n"
            "Preserve the phonetic content as accurately as possible.\n"
            "Return only JSON.\n"
            "The output must contain exactly the same ids as the input items.\n"
            "Do not include explanations, markdown, or code fences."
        )
        if include_schema_hint:
            prompt += (
                "\n\nYou MUST return JSON with exactly this structure:"
                '\n{"items": [{"id": <integer>, "transliteration": "<romanized string>"}, ...]}'
                "\n\nExample input:"
                '\n{"items": [{"id": 1, "text": "東京"}, {"id": 2, "text": "大阪"}]}'
                "\nExample output:"
                '\n{"items": [{"id": 1, "transliteration": "toukyou"}, {"id": 2, "transliteration": "oosaka"}]}'
                "\n\nDo NOT return a flat dictionary like {\"1\": \"toukyou\", \"2\": \"oosaka\"}."
                "\nDo NOT use string keys for IDs. Each entry must be an object with an integer 'id' and a string 'transliteration'."
                "\nThe 'items' array must contain exactly the same IDs as the input, no more, no less."
            )
        return prompt
