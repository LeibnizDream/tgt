from inference.translation.abstract import TranslationStrategy
from inference.translation.custom import CustomTranslationStrategy
from inference.translation.deepl import DeeplStrategy
from inference.translation.marian import MarianStrategy
from inference.translation.M2M100 import M2M100Strategy
from inference.translation.bengali import BengaliTranslationStrategy
from inference.translation.llm import LLMTranslationStrategy


class TranslationStrategyFactory:
    @staticmethod
    def get_strategy(language_code: str, translationModel: str = None) -> TranslationStrategy:
        # Explicit model selection from frontend
        if translationModel and translationModel.lower() == "deepl":
            return DeeplStrategy(language_code)
        elif translationModel and translationModel.lower() in ["gemini", "qwen"]:
            return LLMTranslationStrategy(language_code, translationModel)
        elif translationModel and translationModel.lower() == "marian":
            return MarianStrategy(language_code)
        elif translationModel and translationModel.lower() == "m2m100":
            return M2M100Strategy(language_code)
        # Custom trained model
        elif translationModel and translationModel.lower() not in ("default", ""):
            return CustomTranslationStrategy(language_code, translationModel)
        # Default: pick by language
        if language_code in ['ar','de', 'el', 'es', 'et', 'it', 'ja',
                             'pt', 'ro', 'ru', 'th', 'tr', 'uk', 'vi', 'zh', 'hu']:
            return DeeplStrategy(language_code)
        elif language_code == 'bn':
            return BengaliTranslationStrategy(language_code)
        else:
            raise ValueError(f"No pretrained translation strategy available for language code: {language_code}")