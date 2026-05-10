from inference.strategies.transcription.whisperx import WhisperxStrategy
from inference.strategies.transcription.whisper import WhisperStrategy
from inference.strategies.transcription.bengali import BengaliStrategy as BengaliTranscriptionStrategy
from inference.strategies.transcription.vietnamese import VietnameseStrategy
from inference.strategies.transcription.thai import ThaiTranscriptionStrategy

from inference.strategies.translation.custom import CustomTranslationStrategy
from inference.strategies.translation.deepl import DeeplStrategy
from inference.strategies.translation.marian import MarianStrategy
from inference.strategies.translation.M2M100 import M2M100Strategy
from inference.strategies.translation.bengali import BengaliTranslationStrategy

from inference.strategies.transliteration.japanese import JapaneseStrategy as JapaneseTransliterationStrategy
from inference.strategies.transliteration.chinese import ChineseStrategy
from inference.strategies.transliteration.bengali import BengaliStrategy as BengaliTransliterationStrategy
from inference.strategies.transliteration.default import DefaultStrategy

from inference.strategies.glossing.spacy import SpaCyGlossingStrategy
from inference.strategies.glossing.stanza import StanzaGlossingStrategy

from inference.strategies.llm_strategy import LLMStrategy, is_llm


_TRANSLATION_DEEPL_LANGUAGES = {
    'ar', 'de', 'el', 'es', 'et', 'it', 'ja', 'pt', 'ro', 'ru', 'th', 'tr', 'uk', 'vi', 'zh', 'hu',
}

_GLOSSING_LLM_LANGUAGES = {
    'de', 'et', 'fi', 'fr', 'it', 'ro', 'ru', 'uk', 'tr', 'vi', 'en', 'pt', 'ja', 'hu', 'el',
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
            return ThaiTranscriptionStrategy(language)
        if language == 'vi':
            return VietnameseStrategy(language)
        if language == 'bn':
            return BengaliTranscriptionStrategy(language)
        if language == 'et':
            return WhisperStrategy(language)
        return WhisperxStrategy(language)

    @staticmethod
    def get_translate(language: str, model: str = None):
        if is_llm(model):
            return LLMStrategy(language, "translate", model)
        m = (model or "").lower()
        if m == "deepl":
            return DeeplStrategy(language)
        if m == "marian":
            return MarianStrategy(language)
        if m == "m2m100":
            return M2M100Strategy(language)
        if m and m != "default":
            return CustomTranslationStrategy(language, model)
        if language in _TRANSLATION_DEEPL_LANGUAGES:
            return DeeplStrategy(language)
        if language == "bn":
            return BengaliTranslationStrategy(language)
        raise ValueError(f"No translation strategy for language: {language!r}")

    @staticmethod
    def get_transliterate(language: str, model: str = None):
        if is_llm(model):
            return LLMStrategy(language, "transliterate", model)
        if language == "zh":
            return ChineseStrategy()
        if language == "ja":
            return JapaneseTransliterationStrategy()
        if language == "bn":
            return BengaliTransliterationStrategy()
        if language in ("el", "ru"):
            return DefaultStrategy(language_code=language)
        raise ValueError(f"No transliteration strategy for language: {language!r}")

    @staticmethod
    def get_gloss(language: str, model: str = None):
        if is_llm(model):
            return LLMStrategy(language, "gloss", model)
        m = (model or "").lower()
        if m == "spacy":
            return SpaCyGlossingStrategy(language)
        if m == "stanza":
            return StanzaGlossingStrategy(language)
        if m and m != "default":
            return SpaCyGlossingStrategy(language, model)
        if language in _GLOSSING_LLM_LANGUAGES:
            return LLMStrategy(language, "gloss")
        raise ValueError(f"No glossing strategy for language: {language!r}")
