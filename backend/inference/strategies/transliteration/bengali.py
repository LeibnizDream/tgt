# Copyright (C) 2026 Leibniz-Zentrum Allgemeine Sprachwissenschaft
# Developed as part of the ERC-funded LeibnizDream project.
# Licensed under the GNU General Public License v3.0 or later.
# See LICENSE for details.

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