import json
import sys
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from inference.glossing.abstract import GlossingStrategy


class LLMGlossingStrategy(GlossingStrategy):

    def load_model(self):
        if self.glossing_model == 'gemini':
            self.nlp = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.0,
                max_tokens=None,
                timeout=120,
                max_retries=2,
            )
        elif self.glossing_model == 'qwen' or self.glossing_model is None:
            self.nlp = ChatOllama(
                model="qwen3.5:9b",
                temperature=0.0,
                think=False,
            )
        print(f"Loaded model for glossing: {self.glossing_model or 'gemini'}", file=sys.stderr)

    def gloss(self, payload: str) -> str:
        """
        Payload is a JSON string with:
        {
            "examples": [{"source": "...", "gloss": "..."}],
            "items": [{"id": 0, "text": "..."}]
        }
        """
        payload_data = json.loads(payload)
        examples = payload_data.get('examples', [])
        items = payload_data.get('items', [])

        system = (
            "You are a linguistic glossing engine following Leipzig Glossing Rules.\n"
            "You will receive JSON with examples (showing input text and expected gloss output) "
            "and items to process.\n"
            "Return ONLY valid JSON: {\"items\": [{\"id\": <int>, \"gloss\": \"...\"}]}\n"
            "No explanations, markdown, or code blocks.\n"
            "Output must contain exactly the same ids as input items."
        )

        # Build payload with examples separate from items
        if examples:
            human_payload = json.dumps({
                "examples": [
                    {"text": ex['source'], "gloss": ex['gloss']}
                    for ex in examples
                ],
                "items": items
            }, ensure_ascii=False)
        else:
            raise Exception('No examples provided or recognized')

        messages = [
            ("system", system),
            ("human", human_payload),
        ]

        print("Sending request with messages:", messages, file=sys.stderr)

        response = self.nlp.invoke(messages)

        print("Received response:", response)

        text = response.content.strip()

        # Clean markdown
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        # Validate
        try:
            parsed = json.loads(text)
            if 'items' not in parsed:
                raise ValueError("Response missing 'items' key")
            
            input_ids = {item['id'] for item in items}
            output_ids = {item['id'] for item in parsed['items']}
            if input_ids != output_ids:
                raise ValueError(f"ID mismatch. Input: {input_ids}, Output: {output_ids}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON:\n{text}") from e

        return text