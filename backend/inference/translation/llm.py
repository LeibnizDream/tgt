import json
import sys
from langchain_google_genai import ChatGoogleGenerativeAI
from inference.translation.abstract import TranslationStrategy


class LLMTranslationStrategy(TranslationStrategy):

    def load_model(self):
        if self.glossing_model == 'gemini':
            self.nlp = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.0,
                max_tokens=None,
                timeout=120,
                max_retries=2,
            )
        if self.glossing_model == 'qwen':
            self.nlp = ChatOllama(
                model="qwen3:latest",
                temperature=0.0,
                think=False,
            )

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

        system = (
            "You are a translation engine.\n"
            "You will receive JSON with optional examples (showing source text and expected translation) "
            "and items to translate into English.\n"
            "Return ONLY valid JSON: {\"items\": [{\"id\": <int>, \"translation\": \"...\"}]}\n"
            "No explanations, markdown, or code blocks.\n"
            "Output must contain exactly the same ids as input items."
        )

        if examples:
            human_payload = json.dumps({
                "examples": [
                    {"text": ex["source"], "translation": ex["translation"]}
                    for ex in examples
                ],
                "items": items,
            }, ensure_ascii=False)
        else:
            raise Exception('No examples found or provided')

        messages = [
            ("system", system),
            ("human", human_payload),
        ]

        print("Sending Gemini translation request:", messages, file=sys.stderr)

        response = self.nlp.invoke(messages)

        print("Received Gemini translation response:", response)

        text = response.content.strip()

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