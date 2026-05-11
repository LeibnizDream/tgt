from bntrans import Translator
from inference.strategies.abstract_strategy import AbstractStrategy


class BengaliTranslationStrategy(AbstractStrategy):

    def load_model(self):
        """Load the Bengali translator model"""
        self.translator = Translator(src="bn", dest="en")

    def run_strategy(self, text: str) -> str | None:
        """Translate Bengali text to English"""
        if not hasattr(self, 'translator') or self.translator is None:
            self.load_model()
        
        # Ensure input is properly encoded as UTF-8 string
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='replace')
        elif not isinstance(text, str):
            text = str(text)
        
        # Strip whitespace and check for empty input
        text = text.strip()
        if not text:
            return ""
        
        result = self.translator.translate(text)

        if isinstance(result, bytes):
            result = result.decode('utf-8', errors='replace')

        return result