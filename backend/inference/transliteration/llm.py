from inference.transliteration.abstract import TransliterationStrategy
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from ollama import Client
from utils.functions import ensure_ollama_running


class LLMTransliterationStrategy(TransliterationStrategy):
    def __init__(self, language_code: str):
        super().__init__(language_code)

    def load_model(self):
        if self.transliterate_model == "gemini":
            self.nlp = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite",
                temperature=0.0,
                max_tokens=None,
                timeout=120,
                max_retries=2,
            )
        elif self.transliterate_model == "qwen":
            ensure_ollama_running()
            self.nlp = Client(host="http://127.0.0.1:11434")
            self.model_name = "qwen3.5:9b"
        else:
            raise ValueError(f"Unsupported transliteration model: {self.transliterate_model}")

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
