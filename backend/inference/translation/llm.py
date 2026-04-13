import json
import sys
from typing import List, Optional

from pydantic import BaseModel, Field, ValidationError
from langchain_google_genai import ChatGoogleGenerativeAI
from ollama import Client

from inference.translation.abstract import TranslationStrategy


class TranslationItem(BaseModel):
    id: int
    translation: str


class TranslationResponse(BaseModel):
    items: List[TranslationItem]


class LLMTranslationStrategy(TranslationStrategy):
    def load_model(self):
        if self.translationModel == "gemini":
            self.nlp = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.0,
                max_tokens=None,
                timeout=120,
                max_retries=2,
            )
        elif self.translationModel == "qwen":
            self.nlp = Client(host="http://127.0.0.1:11434")
            self.model_name = "qwen3.5:9b"
        else:
            raise ValueError(f"Unsupported translation model: {self.translationModel}")

    def translate(self, payload: str) -> str:
        """
        Payload shape:
        {
            "examples": [{"source": "...", "translation": "..."}],
            "items": [{"id": 0, "text": "..."}]
        }
        """
        payload_data = json.loads(payload)
        examples = payload_data.get("examples", []) or []
        items = payload_data.get("items", []) or []

        if not items:
            raise ValueError("No items found in payload")

        self._validate_input_items(items)

        if self.translationModel == "gemini":
            return self._translate_with_gemini(items, examples)

        if self.translationModel == "qwen":
            return self._translate_with_ollama(items, examples)

        raise ValueError(f"Unsupported translation model: {self.translationModel}")

    def _translate_with_gemini(self, items: list, examples: list) -> str:
        system = self._build_system_prompt()
        human_payload = json.dumps(
            {
                "examples": self._normalize_examples(examples),
                "items": items,
            },
            ensure_ascii=False,
        )

        messages = [
            ("system", system),
            ("human", human_payload),
        ]

        response = self.nlp.invoke(messages)
        text = self._strip_code_fences(response.content.strip())
        parsed = self._validate_output_text(text, items)
        return parsed.model_dump_json(ensure_ascii=False)

    def _translate_with_ollama(self, items: list, examples: list) -> str:
        system = self._build_system_prompt(include_schema_hint=True)

        user_payload = {
            "examples": self._normalize_examples(examples),
            "items": items,
        }

        response = self.nlp.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            format=TranslationResponse.model_json_schema(),
            stream=False,
            think=False,
            keep_alive="10m",
            options={
                "temperature": 0,
                "num_predict": 8000,
                "num_ctx": 4096,
            },
        )

        content = response["message"]["content"].strip()

        eval_count = response.get('eval_count')
        done_reason = response.get('done_reason')
        print(
            "Ollama translation timings | "
            f"total={response.get('total_duration')} "
            f"load={response.get('load_duration')} "
            f"prompt_eval={response.get('prompt_eval_duration')} "
            f"eval={response.get('eval_duration')} "
            f"prompt_tokens={response.get('prompt_eval_count')} "
            f"output_tokens={eval_count} "
            f"done_reason={done_reason}",
            file=sys.stderr,
        )
        print(f"[DEBUG] num_items={len(items)} num_predict=2000 output_chars={len(content)}", file=sys.stderr)
        if done_reason == "length":
            print(f"[DEBUG] WARNING: output was cut off because num_predict limit was reached", file=sys.stderr)
        print(f"[DEBUG] raw ollama content: {content}", file=sys.stderr)

        parsed = self._validate_output_text(content, items)
        return parsed.model_dump_json(ensure_ascii=False)

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
                "\nReturn JSON with exactly this structure:\n"
                '{"items": [{"id": 0, "translation": "..." }]}'
            )

        return prompt

    def _normalize_examples(self, examples: list) -> list:
        normalized = []
        for ex in examples:
            source = ex.get("source")
            translation = ex.get("translation")
            if source is None or translation is None:
                continue
            normalized.append(
                {
                    "text": source,
                    "translation": translation,
                }
            )
        return normalized

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

    def _validate_output_text(self, text: str, input_items: list) -> TranslationResponse:
        text = self._strip_code_fences(text)

        try:
            parsed = TranslationResponse.model_validate_json(text)
        except ValidationError as e:
            raise ValueError(f"Invalid translation JSON:\n{text}") from e

        input_ids = {item["id"] for item in input_items}
        output_ids = {item.id for item in parsed.items}

        if input_ids != output_ids:
            raise ValueError(f"ID mismatch. Input: {input_ids}, Output: {output_ids}")

        return parsed