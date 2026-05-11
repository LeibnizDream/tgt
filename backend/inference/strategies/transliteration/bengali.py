import bengali_romanizer
from inference.strategies.abstract_strategy import AbstractStrategy


class BengaliTransliterationStrategy(AbstractStrategy):
    
    def load_model(self):
        pass
    
    def run_strategy(self, text: str) -> str:
        if isinstance(text, bytes):
            text = text.decode('utf-8')
        
        if not isinstance(text, str):
            text = str(text)
        
        result = bengali_romanizer.romanize(text)
        
        if isinstance(result, bytes):
            result = result.decode('utf-8')
        
        return result