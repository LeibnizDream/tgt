import os
import deepl
from inference.strategies.translation.abstract import TranslationStrategy


class DeeplStrategy(TranslationStrategy):
    def load_model(self):
        api_key = os.getenv("DEEPL_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPL_API_KEY not set — load secrets before starting the worker")

        self._deepl_client = deepl.DeepLClient(api_key)
        code = self.language_code.upper()
        if code.lower() == "pt":
            code = "PT-BR"
        self._deepl_source_lang = code

    def translate(self, text: str) -> str | None:
        try:
            result = self._deepl_client.translate_text(
                text,
                source_lang=self._deepl_source_lang,
                target_lang="EN-US",
            )
        except deepl.QuotaExceededException as e:
            raise RuntimeError("DeepL character quota exceeded.") from e

        return result.text
