from inference.transliteration.abstract import TransliterationStrategy 
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate


class BanglaStrategy(TransliterationStrategy):
    def __init__(self):
        self.script_from = sanscript.BENGALI
        self.script_to = sanscript.IAST  # THIS is the academic Latin output you want

    def transliterate(self, text: str) -> str:
        return transliterate(text, self.script_from, self.script_to)