import json
from inference.transliteration.abstract import TransliterationStrategy

class LLMTransliterationStrategy(TransliterationStrategy):
    def __init__(self, language_code: str):
        super().__init__(language_code)

    def load_model(self):
        # same LLM client as translation/glossing strategies
        pass

    def transliterate(self, text: str) -> str:
        try:
            payload = json.loads(text)
            # batch mode — few-shot prompt from payload["examples"] + payload["items"]
            ...
        except (json.JSONDecodeError, KeyError):
            # single string mode
            prompt = (
                f"Transliterate the following {self.language_code} text into Latin script. "
                f"Return only the transliteration, nothing else.\n\n{text}"
            )
