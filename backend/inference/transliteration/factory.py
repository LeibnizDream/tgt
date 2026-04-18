import logging
from inference.transliteration.abstract import TransliterationStrategy
from inference.transliteration.japanese import JapaneseStrategy
from inference.transliteration.chinese import ChineseStrategy
from inference.transliteration.bengali import BengaliStrategy
from inference.transliteration.llm import LLMTransliterationStrategy

logger = logging.getLogger(__name__)

class TransliterationStrategyFactory:
    @staticmethod
    def get_strategy(language_code: str) -> TransliterationStrategy:
        if language_code == "zh":
            return ChineseStrategy()
        elif language_code == "ja":
            return JapaneseStrategy()
        elif language_code == "bn":
            return BengaliStrategy()
        else:
            logger.warning(f"No rule-based transliteration strategy available for language {language_code},"
                           f" falling back to LLM.")
            return LLMTransliterationStrategy(language_code)
            
