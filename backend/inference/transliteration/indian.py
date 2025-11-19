from ai4bharat.transliteration import XlitEngine


class IndianStrategy(TransliterationStrategy):
    def __init__(self, language_code):
        self.engine = XlitEngine(language_code, beam_width=10)

    def transliterate(self, sentence: str) -> str:
        output = self.engine.translit_sentence(sentence)
        return output