from inference.strategies.abstract_strategy import AbstractStrategy
from pypinyin import lazy_pinyin


class ChineseTransliterationStrategy(AbstractStrategy):

    def load_model(self) -> None:
        # Choose pinyin style: TONE2 (Tone after syllable), TONE3 (tone after word), etc.
        self.style = None

    def run_strategy(self, sentence: str) -> str:
        # Use lazy_pinyin for simple local transliteration
        # It handles non-Chinese characters by returning them unchanged

        pinyin_list = lazy_pinyin(
            sentence,
            style=self.style,
            errors='default',
            strict=False
        )
        text = ' '.join(pinyin_list)
        text = text.replace('  ', ' ')
        return text
