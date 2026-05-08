from inference.strategies.transliteration.abstract import TransliterationStrategy 
import bengali_romanizer


class BengaliStrategy(TransliterationStrategy):
    def __init__(self):
        pass

    def transliterate(self, text: str) -> str:
        if isinstance(text, bytes):
            text = text.decode('utf-8')
        
        if not isinstance(text, str):
            text = str(text)
        
        result = bengali_romanizer.romanize(text)
        
        if isinstance(result, bytes):
            result = result.decode('utf-8')
        
        return result