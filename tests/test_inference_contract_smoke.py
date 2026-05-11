import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from inference.processing_options import ProcessingOptions
from inference.processors.processor_factory import ProcessorFactory
from inference.strategies.strategy_factory import StrategyFactory


class FakeStrategy:
    def run_strategy(self, text):
        return f"ok:{text}"


@pytest.fixture
def no_model_loading(monkeypatch):
    """Keep factory tests focused on Python contracts, not model downloads."""
    from inference.strategies.glossing.spacy import SpaCyGlossingStrategy
    from inference.strategies.glossing.stanza import StanzaGlossingStrategy
    from inference.strategies.llm_strategy import LLMStrategy
    from inference.strategies.translation.M2M100 import M2M100Strategy
    from inference.strategies.translation.bengali import BengaliTranslationStrategy
    from inference.strategies.translation.deepl import DeeplStrategy
    from inference.strategies.translation.marian import MarianStrategy

    for cls in (
        DeeplStrategy,
        MarianStrategy,
        M2M100Strategy,
        BengaliTranslationStrategy,
        SpaCyGlossingStrategy,
        StanzaGlossingStrategy,
        LLMStrategy,
    ):
        monkeypatch.setattr(cls, "load_model", lambda self: None)

    monkeypatch.setattr(LLMStrategy, "_warmup", lambda self: None)


@pytest.mark.parametrize(
    ("language", "model"),
    [
        ("de", None),
        ("de", "deepl"),
        ("de", "marian"),
        ("de", "m2m100"),
        ("bn", None),
        ("de", "gemini"),
        ("de", "qwen"),
    ],
)
def test_translation_strategy_factory_contracts(no_model_loading, language, model):
    strategy = StrategyFactory.get_translate(language, model)

    assert callable(strategy.run_strategy)


@pytest.mark.parametrize(
    ("language", "model"),
    [
        ("de", "spacy"),
        ("de", "stanza"),
        ("de", "gemini"),
        ("de", "qwen"),
    ],
)
def test_gloss_strategy_factory_contracts(no_model_loading, language, model):
    strategy = StrategyFactory.get_gloss(language, model)

    assert callable(strategy.run_strategy)


@pytest.mark.parametrize(
    ("language", "model"),
    [
        ("zh", None),
        ("ja", None),
        ("bn", None),
        ("el", None),
        ("de", "gemini"),
        ("de", "qwen"),
    ],
)
def test_transliteration_strategy_factory_contracts(no_model_loading, language, model):
    strategy = StrategyFactory.get_transliterate(language, model)

    assert callable(strategy.run_strategy)


@pytest.mark.parametrize(
    ("format_name", "action", "language", "instruction"),
    [
        ("labvanced", "transcribe", "de", "sentences"),
        ("labvanced", "translate", "de", "sentences"),
        ("labvanced", "gloss", "de", "sentences"),
        ("labvanced", "transliterate", "ja", "sentences"),
        ("labvanced", "create columns", "de", "sentences"),
        ("plain", "transcribe", "de", None),
        ("plain", "translate", "de", None),
        ("plain", "gloss", "de", None),
        ("plain", "transliterate", "ja", None),
    ],
)
def test_processor_factory_accepts_processing_options(
    monkeypatch,
    format_name,
    action,
    language,
    instruction,
):
    monkeypatch.setattr(
        StrategyFactory,
        "get_strategy",
        staticmethod(lambda *args, **kwargs: FakeStrategy()),
    )

    processor = ProcessorFactory.get_processor(
        ProcessingOptions(
            language=language,
            action=action,
            format=format_name,
            instruction=instruction,
            model=None,
        )
    )

    assert callable(processor.process)


def test_plain_text_processor_has_all_text_action_column_mappings():
    from inference.processors.plain import plain_text

    for action in ("translate", "gloss", "transliterate"):
        assert action in plain_text._SOURCE_COLS
        assert action in plain_text._TARGET_COLS
        assert plain_text._TARGET_COLS[action]


@pytest.mark.parametrize(
    ("language", "action", "instruction", "source_col", "target_cols"),
    [
        (
            "de",
            "translate",
            "corrected",
            "latin_transcription_everything",
            ["automatic_translation_corrected_transcription", "translation_everything"],
        ),
        (
            "de",
            "translate",
            "automatic",
            "automatic_transcription",
            ["automatic_translation_automatic_transcription"],
        ),
        (
            "de",
            "translate",
            "sentences",
            "latin_transcription_utterance_used",
            ["automatic_translation_utterance_used", "translation_utterance_used"],
        ),
        (
            "ja",
            "translate",
            "corrected",
            "transcription_original_script",
            ["automatic_translation_corrected_transcription", "translation_everything"],
        ),
        (
            "de",
            "gloss",
            "sentences",
            "latin_transcription_utterance_used",
            ["automatic_glossing", "glossing_utterance_used"],
        ),
        (
            "ja",
            "transliterate",
            "sentences",
            "transcription_original_script_utterance_used",
            ["latin_transcription_utterance_used"],
        ),
        (
            "ja",
            "transliterate",
            "corrected",
            "transcription_original_script",
            ["latin_transcription_everything"],
        ),
    ],
)
def test_labvanced_text_processor_column_contracts(
    monkeypatch,
    language,
    action,
    instruction,
    source_col,
    target_cols,
):
    from inference.processors.labvanced.labvanced_text import LabvancedTextProcessor

    monkeypatch.setattr(
        StrategyFactory,
        "get_strategy",
        staticmethod(lambda *args, **kwargs: FakeStrategy()),
    )

    processor = LabvancedTextProcessor(language, action, instruction)

    assert processor._get_source_column() == source_col
    assert processor._get_target_columns() == target_cols


@pytest.mark.parametrize(
    ("language", "action", "instruction", "source_col", "target_col"),
    [
        ("de", "translate", "sentences", "latin_transcription_utterance_used", "translation_utterance_used"),
        ("de", "gloss", "sentences", "latin_transcription_utterance_used", "glossing_utterance_used"),
        ("ja", "transliterate", "sentences", "transcription_original_script_utterance_used", "latin_transcription_utterance_used"),
    ],
)
def test_labvanced_text_processor_writes_expected_target_column(
    monkeypatch,
    language,
    action,
    instruction,
    source_col,
    target_col,
):
    from inference.processors.labvanced.labvanced_text import LabvancedTextProcessor

    monkeypatch.setattr(
        StrategyFactory,
        "get_strategy",
        staticmethod(lambda *args, **kwargs: FakeStrategy()),
    )

    processor = LabvancedTextProcessor(language, action, instruction)
    df = pd.DataFrame({source_col: ["source text"]})

    result = processor._process_dataframe(df)

    assert result.at[0, target_col] == "ok:source text"


def test_local_worker_accepts_language_codes():
    from inference.local_worker import LocalWorker

    LocalWorker(
        base_dir="/tmp",
        options=ProcessingOptions(
            language="german",
            action="translate",
            format="plain",
            instruction=None,
            model=None,
        ),
    )


def test_local_worker_initial_message_uses_available_state(capsys):
    from inference.local_worker import LocalWorker

    worker = LocalWorker(
        base_dir="/tmp",
        options=ProcessingOptions(
            language="german",
            action="translate",
            format="plain",
            instruction=None,
            model=None,
        ),
    )

    worker._initial_message()

    assert "translate" in capsys.readouterr().out
