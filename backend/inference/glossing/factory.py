from inference.glossing.abstract import GlossingStrategy
from inference.glossing.spacy import SpaCyGlossingStrategy
from inference.glossing.portuguese import PortugueseGlossingStrategy
from inference.glossing.stanza import StanzaGlossingStrategy
from inference.glossing.japanese import JapaneseGlossingStrategy
from inference.glossing.gemini import GeminiGlossingStrategy
from inference.glossing.qwen import QwenGlossingStrategy


class GlossingStrategyFactory:
    @staticmethod
    def get_strategy(language_code: str, glossingModel = None, translationModel = None) -> GlossingStrategy:
        if glossingModel:
            return SpaCyGlossingStrategy(language_code, 
                                         glossingModel=glossingModel, 
                                         translationModel=translationModel)
        elif language_code in []:
            return SpaCyGlossingStrategy(language_code)
        elif language_code in ['de', 'et', 'fi', 'fr', 'it', 'ro', 'ru', 'uk',
                               "tr", "vi", "uk", "ru", "en", "it", "pt", 'ja']:
            return GeminiGlossingStrategy(language_code)
        elif language_code in []:
            return QwenGlossingStrategy(language_code)
        elif language_code in []:
            return StanzaGlossingStrategy(language_code)
        elif language_code in []:
            return PortugueseGlossingStrategy(language_code)
        elif language_code in []:
            return JapaneseGlossingStrategy(language_code)
        else:
            raise ValueError(f"No glossing strategy available for language code: {language_code}")