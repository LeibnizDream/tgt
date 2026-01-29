from inference.translation.abstract import TranslationStrategy
from inference.translation.marian import MarianStrategy
import os
import sys
import deepl
from pathlib import Path
from dotenv import load_dotenv

_this_file = Path(__file__).resolve()
parent_dir = _this_file.parent.parent.parent

class DeeplStrategy(TranslationStrategy):
    def load_model(self):
            """
            Attempt to create a DeepL client. If the API key is missing
            or DeepL is unreachable, leave them as None.
            """
            secrets_path = os.path.join(parent_dir, 'materials', 'secrets.env')
            load_dotenv(secrets_path, override=True)
            api_key = os.getenv("DEEPL_API_KEY")
            if not api_key:
                raise RuntimeError("DeepL API_KEY missing or invalid")

            self._deepl_client = deepl.DeepLClient(api_key)
            # Normalize “PT → PT-BR” for DeepL
            code = self.language_code.upper()
            if code.lower() == "pt":
                code = "PT-BR"
            self._deepl_source_lang = code

            try:
                fallback = MarianStrategy(self.language_code)
            except Exception as e:
                print(f"Error initializing fallback MarianStrategy: {e}")
                fallback = None

    def translate(self, text: str) -> str | None:
            print("Using DeepL Strategy")
            """
            If the DeepL client was successfully created, call it.
            Otherwise return None.
            """
            if not self._deepl_client:
                raise RuntimeError(
                    "DeepL client not initialized. "
                    "Call _init_deepl_client() before translating."
                )

            result = self._deepl_client.translate_text(
                text,
                source_lang=self._deepl_source_lang,
                target_lang="EN-US"
                )
            print(f"Inside strategy {result}")

            if not result.text:
                result = self.fallback.translate(text)
                print(f"DeepL translation failed, using fallback: {result}")
                return result

            else:
                print(f"DeepL translation successful: {result.text}")
                return result.text