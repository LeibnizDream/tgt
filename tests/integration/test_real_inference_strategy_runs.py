"""
Opt-in integration checks that instantiate real strategies and call run_strategy().

Run from the repo root with:

    RUN_REAL_INFERENCE_MODELS=1 uv run pytest tests/integration/test_real_inference_strategy_runs.py -q

These tests are intentionally separate from smoke tests because real strategies
may load local models, use heavier libraries, or require environment setup.
"""
import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from inference.strategies.strategy_factory import StrategyFactory


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_REAL_INFERENCE_MODELS") != "1",
    reason="Set RUN_REAL_INFERENCE_MODELS=1 to run real strategy integration tests.",
)


SECRETS_PATH = Path(__file__).resolve().parents[2] / "backend" / "materials" / "secrets.env"


@pytest.fixture(autouse=True)
def load_test_secrets():
    load_dotenv(SECRETS_PATH, override=True)


@pytest.mark.parametrize(
    ("language", "text"),
    [
        ("zh", "你好世界"),
        ("ja", "こんにちは世界"),
        ("bn", "আমি বাংলা বলি"),
        ("el", "Καλημέρα κόσμε"),
    ],
)
def test_real_transliteration_strategy_runs(language, text):
    strategy = StrategyFactory.get_transliterate(language)

    result = strategy.run_strategy(text)

    assert isinstance(result, str)
    assert result.strip()
    assert result != text


@pytest.mark.parametrize(
    ("language", "model", "text"),
    [
        ("de", "deepl", "Guten Morgen"),
        ("de", "marian", "Guten Morgen"),
        ("de", "m2m100", "Guten Morgen"),
        ("bn", None, "আমি বাংলা বলি"),
    ],
)
def test_real_translation_strategy_runs(language, model, text):
    strategy = StrategyFactory.get_translate(language, model)

    result = strategy.run_strategy(text)

    assert isinstance(result, str)
    assert result.strip()


@pytest.mark.parametrize(
    ("language", "model", "text"),
    [
        ("de", "spacy", "Guten Morgen"),
        ("de", "stanza", "Guten Morgen"),
    ],
)
def test_real_gloss_strategy_runs(language, model, text):
    strategy = StrategyFactory.get_gloss(language, model)

    result = strategy.run_strategy(text)

    assert isinstance(result, str)
    assert result.strip()
