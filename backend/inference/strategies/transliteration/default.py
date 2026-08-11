# Copyright (C) 2026 Leibniz-Zentrum Allgemeine Sprachwissenschaft
# Developed as part of the ERC-funded LeibnizDream project.
# Licensed under the GNU General Public License v3.0 or later.
# See LICENSE for details.

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
            