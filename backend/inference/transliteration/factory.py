from inference.transliteration.abstract import TransliterationStrategy
from inference.transliteration.japanese import JapaneseStrategy
from inference.transliteration.chinese import ChineseStrategy
from inference.transliteration.bengali import BengaliStrategy
from inference.transliteration.default import DefaultStrategy

class TransliterationStrategyFactory:
    @staticmethod
    def get_strategy(language_code: str) -> TransliterationStrategy:
        if language_code == "zh":
            return ChineseStrategy()
        elif language_code == "ja":
            return JapaneseStrategy()
        elif language_code == "bn":
            return BengaliStrategy()
        elif language_code in ['el', 'ru']:
            return DefaultStrategy(language_code=language_code)
        else:
            raise ValueError(f"No transliteration strategy available for language code: {language_code}")
            
