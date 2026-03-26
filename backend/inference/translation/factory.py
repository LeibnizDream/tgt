from inference.translation.abstract import TranslationStrategy
from inference.translation.custom import CustomTranslationStrategy
from inference.translation.deepl import DeeplStrategy
from inference.translation.marian import MarianStrategy
from inference.translation.M2M100 import M2M100Strategy
from inference.translation.bengali import BengaliTranslationStrategy
from inference.translation.gemini import GeminiTranslationStrategy


class TranslationStrategyFactory:
    @staticmethod
    def get_strategy(language_code: str, translationModel: str = None) -> TranslationStrategy:
        if translationModel:
            return CustomTranslationStrategy(language_code, translationModel)
        if language_code in ['ar','de', 'el', 'es', 'et', 'it', 'ja', 
                             'pt', 'ro', 'ru', 'th', 'tr', 'uk', 'vi', 'zh', 'hu']:
            return DeeplStrategy(language_code)
        elif language_code == 'bn':
            return BengaliTranslationStrategy(language_code)
        elif language_code in []:
            return MarianStrategy(language_code)
        elif language_code in []:
            return M2M100Strategy(language_code)
        elif language_code in []:
            return GeminiTranslationStrategy(language_code)
        else:
            raise ValueError(f"No pretrained translation strategy available for language code: {language_code}")