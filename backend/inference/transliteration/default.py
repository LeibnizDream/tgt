from inference.transliteration.abstract import TransliterationStrategy 
from transliterate import translit

class DefaultStrategy(TransliterationStrategy):
    def transliterate(self, sentence: str) -> str:
        text = translit(sentence, self.language_code, reversed=True)
        if self.language_code == 'el':
            text = text.replace('x', 'ks').replace('y', 'u')
        
        return text
            