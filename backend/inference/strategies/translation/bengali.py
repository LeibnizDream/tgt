from inference.strategies.abstract_strategy import AbstractStrategy
from bntrans import Translator

class BengaliTranslationStrategy(AbstractStrategy):

    def load_model(self):
        """Load the Bengali translator model"""
        self.translator = Translator(src="bn", dest="en")

    def _run_one(self, text: str) -> str | None:
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
        
        try:
            result = self.translator.translate(text)
            
            # Ensure output is UTF-8 string
            if isinstance(result, bytes):
                result = result.decode('utf-8', errors='replace')
            
            return result
        except Exception as e:
            print(f"Translation error for text: {repr(text)[:100]}... Error: {e}")
            return None