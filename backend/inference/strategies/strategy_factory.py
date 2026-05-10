from inference.strategies.transcription.transcription_factory import TranscriptionStrategyFactory
from inference.strategies.translation.translation_factory import TranslationStrategyFactory
from inference.strategies.transliteration.transliteration_factory import TransliterationStrategyFactory
from inference.strategies.glossing.glossing_factory import GlossingStrategyFactory


class StrategyFactory:
    @staticmethod
    def get_strategy(
        action: str,
        language: str,
        translationModel: str = None,
        glossingModel: str = None,
        transliterationModel: str = None,
    ):
        if action == "transcribe":
            return TranscriptionStrategyFactory.get_strategy(language)
        elif action == "translate":
            return TranslationStrategyFactory.get_strategy(language, translationModel)
        elif action == "transliterate":
            return TransliterationStrategyFactory.get_strategy(language, transliterationModel)
        elif action == "gloss":
            return GlossingStrategyFactory.get_strategy(language, glossingModel, translationModel)
        elif action == "create columns":
            return None
        else:
            raise ValueError(f"No strategy for action: {action!r}")
