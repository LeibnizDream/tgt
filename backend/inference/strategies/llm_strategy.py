import json
import sys

from inference.strategies.abstract_strategy import AbstractStrategy
from langchain_google_genai import ChatGoogleGenerativeAI
from ollama import Client
from pydantic import ValidationError, create_model
from utils.functions import ensure_ollama_running

_RESULT_KEYS = {
    "translate":    "translation",
    "gloss":        "gloss",
    "transliterate": "transliteration",
}

_NUM_PREDICT = {
    "gloss": 10000,
}

_SYSTEM_PROMPTS = {
    "translate": (
        "You are a translation engine.\n"
        "Translate each input text into the target language.\n"
        "Return only JSON.\n"
        "The output must contain exactly the same ids as the input items.\n"
        "Do not include explanations, markdown, or code fences."
    ),
    "gloss": (
        "You are a linguistic glossing engine following Leipzig Glossing Rules.\n"
        "You receive examples and items to gloss.\n"
        "Return only JSON.\n"
        "The output must contain exactly the same ids as the input items.\n"
        "Do not include explanations, markdown, or code fences."
    ),
    "transliterate": (
        "You are a transliteration engine.\n"
        "Romanize each input text from its native script into Latin script.\n"
        "Preserve the phonetic content as accurately as possible.\n"
        "Return only JSON.\n"
        "The output must contain exactly the same ids as the input items.\n"
        "Do not include explanations, markdown, or code fences."
    ),
}


def _make_response_model(result_key: str):
    Item = create_model("Item", id=(int, ...), **{result_key: (str, ...)})
    Response = create_model("Response", items=(list[Item], ...))
    return Response


def is_llm(model: str | None) -> bool:
    return (model or "").lower() in {"gemini", "qwen"}


class LLMStrategy(AbstractStrategy):
    """Single LLM strategy for translation, glossing, and transliteration.

    The action ("translate", "gloss", "transliterate") determines the prompt,
    result key, and response schema.  Gemini vs Ollama dispatch is driven by
    the model hint passed at construction time.
    """

    def __init__(self, language: str, action: str, model: str = None):
        self.action = action
        self.result_key = _RESULT_KEYS[action]
        self._model = (model or "qwen").lower()
        self._response_model = _make_response_model(self.result_key)
        super().__init__(language)
        if self._model == "qwen":
            self._warmup()

    # ------------------------------------------------------------------ setup

    def load_model(self) -> None:
        if self._model == "gemini":
            self.nlp = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite",
                temperature=0.0,
                max_tokens=None,
                timeout=120,
                max_retries=2,
            )
        elif self._model == "qwen":
            ensure_ollama_running()
            self.nlp = Client(host="http://127.0.0.1:11434")
            self.model_name = "qwen3.5:9b"
        else:
            raise ValueError(f"Unsupported LLM model: {self._model!r}")
        
    def _run_one(self, text: str):
        raise NotImplementedError("LLMStrategy only supports batch inference")

    def _warmup(self) -> None:
        print(f"[Ollama] Warming up {self.model_name}...", file=sys.stderr)
        try:
            resp = self.nlp.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": '{"items": [{"id": 0, "text": "test"}]}'}],
                format=self._response_model.model_json_schema(),
                stream=False, think=False, keep_alive="10m",
                options={"temperature": 0, "num_predict": 50, "num_ctx": 512},
            )
            load_ms = resp.get("load_duration", 0) / 1e6
            eval_ms  = resp.get("eval_duration",  0) / 1e6
            print(f"[Ollama] Warmup done — load: {load_ms:.0f}ms, eval: {eval_ms:.0f}ms", file=sys.stderr)
        except Exception as e:
            print(f"[Ollama] Warmup failed (non-fatal): {e}", file=sys.stderr)

    # --------------------------------------------------------------- interface

    def run_strategy(self, todo_items: list, examples: list = None, progress_cb=None) -> dict:
        examples = examples or []
        if 0 < len(examples) < 10:
            raise ValueError(
                f"Only {len(examples)} few-shot example(s) — at least 10 required for LLM processing."
            )
        if not todo_items:
            raise ValueError("No items provided")
        self._validate_input_items(todo_items)
        if self._model == "qwen":
            response_json = self._call_with_ollama(todo_items, examples)
        elif self._model == "gemini":
            response_json = self._call_with_gemini(todo_items, examples)
        parsed = json.loads(response_json)
        result = {item["id"]: item[self.result_key] for item in parsed["items"]}
        if progress_cb:
            progress_cb(len(todo_items), len(todo_items))
        return result

    # ---------------------------------------------------------------- dispatch

    def _call_with_gemini(self, items: list, examples: list) -> str:
        system = self._build_system_prompt(include_schema_hint=True)
        human = json.dumps(
            {"examples": self._normalize_examples(examples), "items": items},
            ensure_ascii=False,
        )
        response = self.nlp.invoke([("system", system), ("human", human)])
        text = self._strip_code_fences(response.content.strip())
        return self._validate_output_text(text, items).model_dump_json()

    def _call_with_ollama(self, items: list, examples: list) -> str:
        system = self._build_system_prompt(include_schema_hint=True)
        payload = {"examples": self._normalize_examples(examples), "items": items}
        response = self.nlp.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": json.dumps(payload, ensure_ascii=False)},
            ],
            format=self._response_model.model_json_schema(),
            stream=False, think=False, keep_alive="10m",
            options={
                "temperature": 0,
                "num_predict": _NUM_PREDICT.get(self.action, 8000),
                "num_ctx": 4096,
            },
        )
        content = response["message"]["content"].strip()
        self._log_ollama_timings(response, len(items), content)
        return self._validate_output_text(content, items).model_dump_json()

    # ----------------------------------------------------------------- helpers

    def _build_system_prompt(self, include_schema_hint: bool = False) -> str:
        prompt = _SYSTEM_PROMPTS[self.action]
        if include_schema_hint:
            key = self.result_key
            prompt += (
                f"\n\nYou MUST return JSON with exactly this structure:"
                f'\n{{"items": [{{"id": <integer>, "{key}": "<string>"}}, ...]}}'
                f"\n\nDo NOT return a flat dictionary."
                f"\nDo NOT use string keys for IDs."
                f"\nThe 'items' array must contain exactly the same IDs as the input."
            )
        return prompt

    def _normalize_examples(self, examples: list) -> list:
        key = self.result_key
        return [
            {"text": ex["source"], key: ex[key]}
            for ex in examples
            if ex.get("source") and ex.get(key)
        ]

    def _validate_input_items(self, items: list) -> None:
        seen_ids = set()
        for item in items:
            if "id" not in item or "text" not in item:
                raise ValueError(f"Each item must contain 'id' and 'text'. Bad item: {item}")
            if item["id"] in seen_ids:
                raise ValueError(f"Duplicate item id: {item['id']}")
            seen_ids.add(item["id"])

    def _validate_output_text(self, text: str, input_items: list):
        text = self._strip_code_fences(text)
        try:
            parsed = self._response_model.model_validate_json(text)
        except ValidationError as e:
            raise ValueError(f"Invalid {self.result_key} JSON:\n{text}") from e
        input_ids  = {item["id"] for item in input_items}
        output_ids = {item.id for item in parsed.items}
        if input_ids != output_ids:
            raise ValueError(f"ID mismatch. Input: {input_ids}, Output: {output_ids}")
        return parsed

    def _strip_code_fences(self, text: str) -> str:
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            return "\n".join(lines).strip()
        return text

    def _log_ollama_timings(self, response: dict, num_items: int, content: str) -> None:
        done_reason = response.get("done_reason")
        print(
            f"Ollama {self.result_key} timings | "
            f"total={response.get('total_duration')} "
            f"load={response.get('load_duration')} "
            f"prompt_eval={response.get('prompt_eval_duration')} "
            f"eval={response.get('eval_duration')} "
            f"prompt_tokens={response.get('prompt_eval_count')} "
            f"output_tokens={response.get('eval_count')} "
            f"done_reason={done_reason}",
            file=sys.stderr,
        )
        print(
            f"[DEBUG] num_items={num_items} output_chars={len(content)}",
            file=sys.stderr,
        )
        if done_reason == "length":
            print("[DEBUG] WARNING: output cut off — num_predict limit reached", file=sys.stderr)
        print(f"[DEBUG] raw ollama content: {content}", file=sys.stderr)
