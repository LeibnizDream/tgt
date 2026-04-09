import json
import sys
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from inference.translation.abstract import TranslationStrategy
from langchain_ollama import ChatOllama


class LLMTranslationStrategy(TranslationStrategy):

    def load_model(self):
        if self.translationModel == 'gemini':
            self.nlp = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.0,
                max_tokens=None,
                timeout=120,
                max_retries=2,
            )
        elif self.translationModel == 'qwen':
            self.nlp = ChatOllama(
                model="qwen3.5:9b",
                temperature=0.0,
                base_url="http://127.0.0.1:11434",
                think=False,
                keep_alive=-1,
            )
            try:
                health = self.nlp.invoke([("human", "ping")])
                print(f"Qwen translation health check OK: {health.content[:80]}", file=sys.stderr)
            except Exception as e:
                raise RuntimeError(f"Qwen is not responding before translation: {e}") from e

    def translate(self, payload: str) -> str:
        """
        Payload is a JSON string with:
        {
            "examples": [{"source": "...", "translation": "..."}],
            "items": [{"id": 0, "text": "..."}]
        }
        """
        payload_data = json.loads(payload)
        examples = payload_data.get("examples", [])
        items = payload_data.get("items", [])

        if not examples:
            raise Exception('No examples found or provided')

        system = (
            "You are a translation engine.\n"
            "You will receive JSON with optional examples (showing source text and expected translation) "
            "and items to translate into English.\n"
            "Return ONLY valid JSON: {\"items\": [{\"id\": <int>, \"translation\": \"...\"}]}\n"
            "No explanations, markdown, or code blocks.\n"
            "Output must contain exactly the same ids as input items."
        )

        formatted_examples = [
            {"text": ex["source"], "translation": ex["translation"]}
            for ex in examples
        ]

        if self.translationModel == 'qwen':
            result_items = []
            for item in items:
                human_payload = json.dumps({
                    "examples": formatted_examples,
                    "items": [item],
                }, ensure_ascii=False)
                messages = [("system", system), ("human", human_payload)]
                print(f"Sending translation request for item id={item['id']}", file=sys.stderr)
                t0 = time.time()
                response = self.nlp.invoke(messages)
                print(f"Qwen translation response in {time.time() - t0:.1f}s: {response.content[:80]}", file=sys.stderr)
                text = self._clean_and_validate(response.content.strip(), [item])
                result_items.extend(json.loads(text)["items"])
            return json.dumps({"items": result_items}, ensure_ascii=False)
        else:
            human_payload = json.dumps({
                "examples": formatted_examples,
                "items": items,
            }, ensure_ascii=False)
            messages = [("system", system), ("human", human_payload)]
            response = self.nlp.invoke(messages)
            return self._clean_and_validate(response.content.strip(), items)

    def _clean_and_validate(self, text: str, items: list) -> str:
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            parsed = json.loads(text)
            if "items" not in parsed:
                raise ValueError("Response missing 'items' key")
            input_ids = {item["id"] for item in items}
            output_ids = {item["id"] for item in parsed["items"]}
            if input_ids != output_ids:
                raise ValueError(f"ID mismatch. Input: {input_ids}, Output: {output_ids}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON:\n{text}") from e

        return text