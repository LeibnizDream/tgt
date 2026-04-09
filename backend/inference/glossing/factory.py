from inference.glossing.abstract import GlossingStrategy
from inference.glossing.spacy import SpaCyGlossingStrategy
from inference.glossing.portuguese import PortugueseGlossingStrategy
from inference.glossing.stanza import StanzaGlossingStrategy
from inference.glossing.japanese import JapaneseGlossingStrategy
from inference.glossing.llm import LLMGlossingStrategy
from inference.glossing.qwen import QwenGlossingStrategy


class GlossingStrategyFactory:
    @staticmethod
    def get_strategy(language_code: str, glossingModel = None, translationModel = None) -> GlossingStrategy:
        # Explicit model selection from frontend
        if glossingModel and glossingModel.lower() == "spacy":
            return SpaCyGlossingStrategy(language_code, translationModel=translationModel)
        elif glossingModel and glossingModel.lower() == "stanza":
            return StanzaGlossingStrategy(language_code)
        elif glossingModel and glossingModel.lower() in ['gemini', 'qwen']:
            return LLMGlossingStrategy(language_code, glossingModel=glossingModel)
        # Custom trained model
        elif glossingModel and glossingModel.lower() not in ("default", ""):
            return SpaCyGlossingStrategy(language_code,
                                         glossingModel=glossingModel,
                                         translationModel=translationModel)
        # Default: pick by language
        elif language_code in ['de', 'et', 'fi', 'fr', 'it', 'ro', 'ru', 'uk',
                               'tr', 'vi', 'en', 'pt', 'ja', 'hu', 'el']:
            return GeminiGlossingStrategy(language_code)
        else:
            raise ValueError(f'No glossing strategy available for language code: {language_code}')