from typing import List

from pydantic import BaseModel

from inference.strategies.translation.abstract import TranslationStrategy
from inference.strategies.llm_base import LLMStrategy


class TranslationItem(BaseModel):
    id: int
    translation: str


class TranslationResponse(BaseModel):
    items: List[TranslationItem]


class LLMTranslationStrategy(TranslationStrategy, LLMStrategy):

    @property
    def _model_hint(self) -> str | None:
        return self.translationModel

    def load_model(self) -> None:
        self._init_llm()

    def translate(self, payload: str) -> str:
        return self._call(payload)

    def _result_key(self) -> str:
        return "translation"

    def _response_model(self):
        return TranslationResponse

    def _normalize_examples(self, examples: list) -> list:
        return [
            {"text": ex["source"], "translation": ex["translation"]}
            for ex in examples
            if ex.get("source") and ex.get("translation")
        ]

    def _build_system_prompt(self, include_schema_hint: bool = False) -> str:
        prompt = (
            "You are a translation engine.\n"
            "Translate the input texts into natural English.\n"
            "Return only JSON.\n"
            "The output must contain exactly the same ids as the input items.\n"
            "Do not include explanations, markdown, or code fences."
        )
        if include_schema_hint:
            prompt += (
                "\n\nYou MUST return JSON with exactly this structure:"
                '\n{"items": [{"id": <integer>, "translation": "<translated string>"}, ...]}'
                "\n\nExample input:"
                '\n{"items": [{"id": 4, "text": "der Löwe"}, {"id": 5, "text": "die Schokolade"}]}'
                "\nExample output:"
                '\n{"items": [{"id": 4, "translation": "the lion"}, {"id": 5, "translation": "the chocolate"}]}'
                "\n\nDo NOT return a flat dictionary like {\"4\": \"translation\", \"5\": \"translation\"}."
                "\nDo NOT use string keys for IDs. Each entry must be an object with an integer 'id' and a string 'translation'."
                "\nThe 'items' array must contain exactly the same IDs as the input, no more, no less."
            )
        return prompt
