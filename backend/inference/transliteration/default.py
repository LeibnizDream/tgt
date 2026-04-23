from inference.transliteration.abstract import TransliterationStrategy 
from transliterate import translit

class DefaultStrategy(TransliterationStrategy):
    def transliterate(self, sentence: str) -> str:
        return translit(sentence, self.language_code, reversed=True)