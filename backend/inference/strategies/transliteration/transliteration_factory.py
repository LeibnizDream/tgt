from inference.strategies.transliteration.abstract import TransliterationStrategy
from inference.strategies.transliteration.japanese import JapaneseStrategy
from inference.strategies.transliteration.chinese import ChineseStrategy
from inference.strategies.transliteration.bengali import BengaliStrategy
from inference.strategies.transliteration.default import DefaultStrategy
from inference.strategies.transliteration.llm import LLMTransliterationStrategy


class TransliterationStrategyFactory:
    @staticmethod
    def get_strategy(language_code: str, transliterationModel: str = None) -> TransliterationStrategy:
        if transliterationModel and transliterationModel.lower() in ["gemini", "qwen"]:
            return LLMTransliterationStrategy(language_code, transliterationModel)
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
            
