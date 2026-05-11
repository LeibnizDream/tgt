import re

from inference.strategies.abstract_strategy import AbstractStrategy
from transliterate import translit


class DefaultStrategy(AbstractStrategy):
    def load_model(self) -> None:
        pass

    def run_strategy(self, sentence: str) -> str:
        text = translit(sentence, self.language_code, reversed=True)
        if self.language_code == 'el':
            text = re.sub(r'\w+@pii|x', lambda m: m.group() if '@pii' in m.group() else 'ks', text)
            text = text.replace('y', 'u')

        return text
            