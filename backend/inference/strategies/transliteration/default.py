import re

from inference.strategies.transliteration.abstract import TransliterationStrategy
from transliterate import translit

class DefaultStrategy(TransliterationStrategy):
    def _transliterate_one(self, sentence: str) -> str:
        text = translit(sentence, self.language_code, reversed=True)
        if self.language_code == 'el':
            text = re.sub(r'\w+@pii|x', lambda m: m.group() if '@pii' in m.group() else 'ks', text)
            text = text.replace('y', 'u')

        return text
            