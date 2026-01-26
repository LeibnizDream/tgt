import json
import sys
from langchain_google_genai import ChatGoogleGenerativeAI
from inference.glossing.abstract import GlossingStrategy


class GeminiGlossingStrategy(GlossingStrategy):

    def load_model(self):
        # Check this initialization....
        self.nlp = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.0,
            max_tokens=None,
            timeout=120,
            max_retries=2,
        )

    def gloss(self, payload: str) -> str:
        """
        Payload is a JSON string describing the batch with format:
        {
            "examples": [{"source": "...", "gloss": "..."}],  # Few-shot examples
            "items": [{"id": 0, "text": "..."}]              # Items to gloss
        }
        
        Return value MUST be a raw JSON string with format:
        {
            "items": [{"id": 0, "gloss": "..."}]
        }
        """

        # Parse payload to include examples in prompt if provided
        payload_data = json.loads(payload)
        examples = payload_data.get('examples', [])
        items = payload_data.get('items', [])

        # Build system message with examples
        system_parts = [
            "You are a linguistic glossing engine following Leipzig Glossing Rules.",
            "You will receive JSON with items: [{id, text}].",
            "Return ONLY valid JSON with format: {\"items\": [{\"id\": 0, \"gloss\": \"...\"}]}",
            "Do not add explanations, markdown, code blocks, or extra keys.",
            "Output must contain exactly the same ids as input.",
            "One output item per input item.",
        ]

        # Add few-shot examples if available
        if examples:
            system_parts.append("\nHere are example glosses to follow:")
            for i, ex in enumerate(examples[:10], 1):  # Limit to 10 examples
                system_parts.append(f"\nExample {i}:")
                system_parts.append(f"Input: {ex['source']}")
                system_parts.append(f"Output gloss: {ex['gloss']}")

        system = "\n".join(system_parts)

        # Prepare human message (only the items to gloss)
        human_payload = json.dumps({"items": items}, ensure_ascii=False)

        messages = [
            ("system", system),
            ("human", human_payload),
        ]

        response = self.nlp.invoke(messages)
        print('Full response:', response, file=sys.stderr)

        text = response.content.strip()
        print("Gemini response text:", text, file=sys.stderr)

        # Clean up markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json or ```)
            lines = lines[1:]
            # Remove last line if it's closing ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        # Remove "json" language identifier if present
        if text.startswith("json"):
            text = text[4:].strip()

        # Validate JSON
        try:
            parsed = json.loads(text)
            # Ensure it has the expected structure
            if 'items' not in parsed:
                raise ValueError("Response missing 'items' key")
            if not isinstance(parsed['items'], list):
                raise ValueError("'items' must be a list")
            # Validate all input ids are present
            input_ids = {item['id'] for item in items}
            output_ids = {item['id'] for item in parsed['items']}
            if input_ids != output_ids:
                raise ValueError(f"ID mismatch. Input: {input_ids}, Output: {output_ids}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Gemini returned invalid JSON:\n{text}") from e

        return text