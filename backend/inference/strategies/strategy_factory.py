from inference.strategies.llm_strategy import is_llm
from inference.strategies.transliteration.default import DefaultStrategy

_TRANSLATION_DEEPL_LANGUAGES = {
    'ar', 'de', 'el', 'es', 'et', 'it', 'ja', 'pt', 'ro', 'ru', 'th', 'tr', 'uk', 'vi', 'zh', 'hu',
}


class StrategyFactory:

    @staticmethod
    def get_strategy(language: str, action: str, model: str = None):
        if action == "transcribe":
            return StrategyFactory.get_transcribe(language)
        elif action == "translate":
            return StrategyFactory.get_translate(language, model)
        elif action == "transliterate":
            return StrategyFactory.get_transliterate(language, model)
        elif action == "gloss":
            return StrategyFactory.get_gloss(language, model)
        elif action == "create columns":
            return None
        else:
            raise ValueError(f"No strategy for action: {action!r}")

    @staticmethod
    def get_transcribe(language: str):
        if language == 'th':
            from inference.strategies.transcription.thai import ThaiTranscriptionStrategy
            return ThaiTranscriptionStrategy(language)
        if language == 'vi':
            from inference.strategies.transcription.vietnamese import (
                VietnameseTranscriptionStrategy,
            )
            return VietnameseTranscriptionStrategy(language)
        if language == 'bn':
            from inference.strategies.transcription.bengali import BengaliTranscriptionStrategy
            return BengaliTranscriptionStrategy(language)
        if language == 'et':
            from inference.strategies.transcription.whisper import WhisperStrategy
            return WhisperStrategy(language)
        from inference.strategies.transcription.whisperx import WhisperxStrategy
        return WhisperxStrategy(language)

    @staticmethod
    def get_translate(language: str, model: str = None):
        if is_llm(model):
            from inference.strategies.llm_strategy import LLMStrategy
            return LLMStrategy(language, "translate", model)
        m = (model or "").lower()
        if m == "deepl":
            from inference.strategies.translation.deepl import DeeplStrategy
            return DeeplStrategy(language)
        if m == "marian":
            from inference.strategies.translation.marian import MarianStrategy
            return MarianStrategy(language)
        if m == "m2m100":
            from inference.strategies.translation.M2M100 import M2M100Strategy
            return M2M100Strategy(language)
        if m and m != "default":
            from inference.strategies.translation.custom import CustomTranslationStrategy
            return CustomTranslationStrategy(language, model)
        if language in _TRANSLATION_DEEPL_LANGUAGES:
            from inference.strategies.translation.deepl import DeeplStrategy
            return DeeplStrategy(language)
        if language == "bn":
            from inference.strategies.translation.bengali import BengaliTranslationStrategy
            return BengaliTranslationStrategy(language)
        raise ValueError(f"No translation strategy for language: {language!r}")

    @staticmethod
    def get_transliterate(language: str, model: str = None):
        if is_llm(model):
            from inference.strategies.llm_strategy import LLMStrategy
            return LLMStrategy(language, "transliterate", model)
        if language == "zh":
            from inference.strategies.transliteration.chinese import ChineseTransliterationStrategy
            return ChineseTransliterationStrategy(language)
        if language == "ja":
            from inference.strategies.transliteration.japanese import (
                JapaneseTransliterationStrategy,
            )
            return JapaneseTransliterationStrategy(language)
        if language == "bn":
            from inference.strategies.transliteration.bengali import BengaliTransliterationStrategy
            return BengaliTransliterationStrategy(language)
        if language in ("el", "ru"):
            return DefaultStrategy(language)
        raise ValueError(f"No transliteration strategy for language: {language!r}")

    @staticmethod
    def get_gloss(language: str, model: str = None):
        if is_llm(model):
            from inference.strategies.llm_strategy import LLMStrategy
            return LLMStrategy(language, "gloss", model)
        m = (model or "").lower()
        if m == "spacy":
            from inference.strategies.glossing.spacy import SpaCyGlossingStrategy
            return SpaCyGlossingStrategy(language)
        if m == "stanza":
            from inference.strategies.glossing.stanza import StanzaGlossingStrategy
            return StanzaGlossingStrategy(language)
        if m and m != "default":
            from inference.strategies.glossing.spacy import SpaCyGlossingStrategy
            return SpaCyGlossingStrategy(language, model)
        raise ValueError(f"No glossing strategy for language: {language!r}")
