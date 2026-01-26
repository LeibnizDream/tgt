import json
from langchain_google_genai import ChatGoogleGenerativeAI
from inference.glossing.abstract import GlossingStrategy


class GeminiGlossingStrategy(GlossingStrategy):

    def load_model(self):
        self.nlp = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",   # use stable
            temperature=0.0,         # determinism
            max_tokens=None,
            timeout=120,
            max_retries=2,
        )

    def gloss(self, payload: str) -> str:
        """
        Payload is a JSON string describing the batch.
        Return value MUST be a raw JSON string.
        """

        system = (
            "You are a linguistic glossing engine with leipzig glossing rules.\n"
            "You will receive JSON with items: [{id, text}].\n"
            "Return ONLY valid JSON with items: [{id, gloss}].\n"
            "Do not add explanations, markdown, or extra keys.\n"
            "Output must contain exactly the same ids as input.\n"
            "One output item per input item."
        )

        human = payload

        messages = [
            ("system", system),
            ("human", human),
        ]

        response = self.nlp.invoke(messages)

        text = response.content.strip()

        if text.startswith("```"):
            text = text.split("```", 2)[1].strip()

        try:
            json.loads(text)
        except Exception as e:
            raise ValueError(f"Gemini returned invalid JSON:\n{text}") from e

        return text
