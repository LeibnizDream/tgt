import sys
from typing import List

from pydantic import BaseModel

from inference.strategies.glossing.abstract import GlossingStrategy
from inference.strategies.llm_base import LLMStrategy


class GlossItem(BaseModel):
    id: int
    gloss: str


class GlossResponse(BaseModel):
    items: List[GlossItem]


class LLMGlossingStrategy(GlossingStrategy, LLMStrategy):

    @property
    def _model_hint(self) -> str | None:
        return self.glossing_model

    def load_model(self) -> None:
        self._init_llm()
        if (self.glossing_model or "qwen") == "qwen":
            self._warmup()
        print(f"Loaded model for glossing: {getattr(self, 'model_name', 'gemini')}", file=sys.stderr)

    def gloss(self, payload: str) -> str:
        return self._call(payload)

    def _result_key(self) -> str:
        return "gloss"

    def _response_model(self):
        return GlossResponse

    def _ollama_num_predict(self) -> int:
        return 10000

    def _normalize_examples(self, examples: list) -> list:
        return [
            {"text": ex["source"], "gloss": ex["gloss"]}
            for ex in examples
            if ex.get("source") and ex.get("gloss")
        ]

    def _build_system_prompt(self, include_schema_hint: bool = False) -> str:
        prompt = (
            "You are a linguistic glossing engine following Leipzig Glossing Rules.\n"
            "You receive examples and items to gloss.\n"
            "Return only JSON.\n"
            "The output must contain exactly the same ids as the input items.\n"
            "Do not include explanations, markdown, or code fences."
        )
        if include_schema_hint:
            prompt += (
                "\n\nYou MUST return JSON with exactly this structure:"
                '\n{"items": [{"id": <integer>, "gloss": "<gloss string>"}, ...]}'
                "\n\nExample input:"
                '\n{"items": [{"id": 4, "text": "der Löwe"}, {"id": 5, "text": "die Schokolade"}]}'
                "\nExample output:"
                '\n{"items": [{"id": 4, "gloss": "DET.DEF.M.SG.NOM lion"}, {"id": 5, "gloss": "DET.DEF.F.SG.NOM chocolate"}]}'
                "\n\nDo NOT return a flat dictionary like {\"4\": \"gloss\", \"5\": \"gloss\"}."
                "\nDo NOT use string keys for IDs. Each entry must be an object with an integer 'id' and a string 'gloss'."
                "\nThe 'items' array must contain exactly the same IDs as the input, no more, no less."
            )
        return prompt

    def _warmup(self) -> None:
        print(f"[Ollama] Warming up model {self.model_name} (loading into GPU)...", file=sys.stderr)
        try:
            resp = self.nlp.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": '{"items": [{"id": 0, "text": "test"}]}'}],
                format=GlossResponse.model_json_schema(),
                stream=False,
                think=False,
                keep_alive="10m",
                options={"temperature": 0, "num_predict": 50, "num_ctx": 512},
            )
            load_ms = resp.get("load_duration", 0) / 1e6
            eval_ms = resp.get("eval_duration", 0) / 1e6
            print(f"[Ollama] Warmup done — model load: {load_ms:.0f}ms, inference: {eval_ms:.0f}ms", file=sys.stderr)
        except Exception as e:
            print(f"[Ollama] Warmup failed (non-fatal): {e}", file=sys.stderr)
