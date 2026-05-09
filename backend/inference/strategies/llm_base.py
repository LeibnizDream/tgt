import json
import sys
from abc import ABC, abstractmethod

from pydantic import ValidationError
from langchain_google_genai import ChatGoogleGenerativeAI
from ollama import Client

from utils.functions import ensure_ollama_running


class LLMStrategy(ABC):
    """Mixin providing shared Gemini/Ollama calling infrastructure for LLM strategies.

    Subclasses must implement:
        _model_hint       — returns the model name ('gemini', 'qwen', None=default)
        _build_system_prompt — returns the task-specific system prompt
        _response_model   — returns the Pydantic response class
        _result_key       — returns the response item field name ('translation', 'gloss', …)
        _normalize_examples — converts raw examples to prompt-ready dicts
    """

    @property
    @abstractmethod
    def _model_hint(self) -> str | None: ...

    @abstractmethod
    def _build_system_prompt(self, include_schema_hint: bool = False) -> str: ...

    @abstractmethod
    def _response_model(self): ...

    @abstractmethod
    def _result_key(self) -> str: ...

    @abstractmethod
    def _normalize_examples(self, examples: list) -> list: ...

    def _ollama_num_predict(self) -> int:
        return 8000

    # ------------------------------------------------------------------ setup

    def _init_llm(self) -> None:
        """Initialize self.nlp (and self.model_name for Ollama). Call from load_model()."""
        hint = self._model_hint or "qwen"
        if hint == "gemini":
            self.nlp = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite",
                temperature=0.0,
                max_tokens=None,
                timeout=120,
                max_retries=2,
            )
        elif hint == "qwen":
            ensure_ollama_running()
            self.nlp = Client(host="http://127.0.0.1:11434")
            self.model_name = "qwen3.5:9b"
        else:
            raise ValueError(f"Unsupported LLM model: {hint!r}")

    # ---------------------------------------------------------------- dispatch

    def _call(self, payload: str) -> str:
        payload_data = json.loads(payload)
        examples = payload_data.get("examples", []) or []
        items = payload_data.get("items", []) or []

        if not items:
            raise ValueError("No items found in payload")

        self._validate_input_items(items)

        hint = self._model_hint or "qwen"
        if hint == "gemini":
            return self._call_with_gemini(items, examples)
        return self._call_with_ollama(items, examples)

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
        user_payload = {"examples": self._normalize_examples(examples), "items": items}

        response = self.nlp.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            format=self._response_model().model_json_schema(),
            stream=False,
            think=False,
            keep_alive="10m",
            options={"temperature": 0, "num_predict": self._ollama_num_predict(), "num_ctx": 4096},
        )

        content = response["message"]["content"].strip()
        self._log_ollama_timings(response, len(items), content)
        return self._validate_output_text(content, items).model_dump_json()

    # ----------------------------------------------------------------- helpers

    def _log_ollama_timings(self, response: dict, num_items: int, content: str) -> None:
        eval_count = response.get("eval_count")
        done_reason = response.get("done_reason")
        print(
            f"Ollama {self._result_key()} timings | "
            f"total={response.get('total_duration')} "
            f"load={response.get('load_duration')} "
            f"prompt_eval={response.get('prompt_eval_duration')} "
            f"eval={response.get('eval_duration')} "
            f"prompt_tokens={response.get('prompt_eval_count')} "
            f"output_tokens={eval_count} "
            f"done_reason={done_reason}",
            file=sys.stderr,
        )
        print(
            f"[DEBUG] num_items={num_items} num_predict={self._ollama_num_predict()} "
            f"output_chars={len(content)}",
            file=sys.stderr,
        )
        if done_reason == "length":
            print("[DEBUG] WARNING: output was cut off because num_predict limit was reached", file=sys.stderr)
        print(f"[DEBUG] raw ollama content: {content}", file=sys.stderr)

    def _validate_input_items(self, items: list) -> None:
        seen_ids = set()
        for item in items:
            if "id" not in item or "text" not in item:
                raise ValueError(f"Each item must contain 'id' and 'text'. Bad item: {item}")
            if item["id"] in seen_ids:
                raise ValueError(f"Duplicate item id found: {item['id']}")
            seen_ids.add(item["id"])

    def _strip_code_fences(self, text: str) -> str:
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            return "\n".join(lines).strip()
        return text

    def _validate_output_text(self, text: str, input_items: list):
        text = self._strip_code_fences(text)
        ResponseModel = self._response_model()

        try:
            parsed = ResponseModel.model_validate_json(text)
        except ValidationError as e:
            raise ValueError(f"Invalid {self._result_key()} JSON:\n{text}") from e

        input_ids = {item["id"] for item in input_items}
        output_ids = {item.id for item in parsed.items}

        if input_ids != output_ids:
            raise ValueError(f"ID mismatch. Input: {input_ids}, Output: {output_ids}")

        return parsed
