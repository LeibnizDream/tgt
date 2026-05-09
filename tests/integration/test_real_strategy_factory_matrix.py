import ast
import importlib
import json
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from inference.strategies.strategy_factory import StrategyFactory

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
SECRETS_PATH = BACKEND_ROOT / "materials" / "secrets.env"

FACTORY_SOURCE_BY_ACTION = {
    "transcribe":    BACKEND_ROOT / "inference" / "strategies" / "transcription"   / "transcription_factory.py",
    "translate":     BACKEND_ROOT / "inference" / "strategies" / "translation"     / "translation_factory.py",
    "gloss":         BACKEND_ROOT / "inference" / "strategies" / "glossing"        / "glossing_factory.py",
    "transliterate": BACKEND_ROOT / "inference" / "strategies" / "transliteration" / "transliteration_factory.py",
}

FACTORY_MODULE_BY_ACTION = {
    "transcribe":    "inference.strategies.transcription.transcription_factory",
    "translate":     "inference.strategies.translation.translation_factory",
    "gloss":         "inference.strategies.glossing.glossing_factory",
    "transliterate": "inference.strategies.transliteration.transliteration_factory",
}

METHOD_BY_ACTION = {
    "transcribe":    "transcribe",
    "translate":     "translate",
    "gloss":         "gloss",
    "transliterate": "transliterate",
}

MODEL_CANDIDATES_BY_ACTION = {
    "transcribe": [(None, None)],
    "translate": [
        (None, None),
        ("deepl", None),
        ("marian", None),
        ("m2m100", None),
        ("gemini", None),
        ("qwen", None),
    ],
    "gloss": [
        (None, None),
        (None, "gemini"),
        (None, "qwen"),
        (None, "spacy"),
        (None, "stanza"),
    ],
    "transliterate": [(None, None)],
}

UNCOVERED_BY_DESIGN_BY_ACTION = {
    "translate": {"CustomTranslationStrategy"},
}

def _get_class_names_from_factory(factory_path: Path) -> set[str]:
    tree = ast.parse(factory_path.read_text(encoding="utf-8"))
    class_names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Return):
            continue
        if not isinstance(node.value, ast.Call):
            continue
        if isinstance(node.value.func, ast.Name):
            class_names.add(node.value.func.id)
        elif isinstance(node.value.func, ast.Attribute):
            class_names.add(node.value.func.attr)
    return class_names

def _expected_strategy_classes_by_action() -> dict[str, set[str]]:
    expected = {}
    for action, factory_path in FACTORY_SOURCE_BY_ACTION.items():
        classes = _get_class_names_from_factory(factory_path)
        classes -= UNCOVERED_BY_DESIGN_BY_ACTION.get(action, set())
        expected[action] = classes
    return expected

def _covered_strategy_classes_by_action() -> dict[str, set[str]]:
    covered = {action: set() for action in FACTORY_SOURCE_BY_ACTION}
    for _, action, _, _, _, expected_class_name in STRATEGY_CASES:
        covered[action].add(expected_class_name)
    return covered


def _get_all_languages() -> list[str]:
    languages_path = BACKEND_ROOT / "materials" / "global_variables" / "LANGUAGES"
    languages = json.loads(languages_path.read_text(encoding="utf-8"))
    preferred = ["de", "et", "bn", "vi", "th", "zh", "ja", "el", "ru", "ar"]
    remaining = sorted(code for code in languages if code not in preferred and code != "en")
    return preferred + remaining


def _make_probe_class(class_name: str):
    """Clase falsa que solo recuerda su nombre. Se usa en lugar de la clase real."""
    class ProbeStrategy:
        def __init__(self, *args, **kwargs):
            self.__strategy_class_name__ = class_name
    ProbeStrategy.__name__ = class_name
    return ProbeStrategy


def _patch_factories():
    """Reemplaza las clases reales en los factories con ProbeStrategy."""
    originals = []
    for action, module_name in FACTORY_MODULE_BY_ACTION.items():
        module = importlib.import_module(module_name)
        for class_name in _get_class_names_from_factory(FACTORY_SOURCE_BY_ACTION[action]):
            if hasattr(module, class_name):
                originals.append((module, class_name, getattr(module, class_name)))
                setattr(module, class_name, _make_probe_class(class_name))
    return originals


def _restore_factories(originals) -> None:
    """Restaura las clases reales después del parcheo."""
    for module, class_name, original in originals:
        setattr(module, class_name, original)


def _discover_all_cases() -> list[tuple]:
    all_languages = _get_all_languages()
    cases = []
    seen = set()
    covered_per_action = {action: set() for action in FACTORY_SOURCE_BY_ACTION}

    originals = _patch_factories()
    try:
        for action in FACTORY_SOURCE_BY_ACTION:
            languages = [l for l in all_languages if not (action == "translate" and l == "en")]

            for translation_model, glossing_model in MODEL_CANDIDATES_BY_ACTION[action]:
                for language in languages:
                    try:
                        strategy = StrategyFactory.get_strategy(
                            action=action,
                            language=language,
                            translationModel=translation_model,
                            glossingModel=glossing_model,
                        )
                    except Exception:
                        continue

                    class_name = strategy.__strategy_class_name__

                    if class_name in UNCOVERED_BY_DESIGN_BY_ACTION.get(action, set()):
                        continue

                    case_key = (action, class_name, translation_model, glossing_model)
                    if case_key in seen:
                        continue

                    is_new_class = class_name not in covered_per_action[action]
                    has_explicit_model = translation_model is not None or glossing_model is not None
                    if not is_new_class and not has_explicit_model:
                        continue

                    label = f"{action}-{class_name.removesuffix('Strategy').lower()}-{language}"
                    if translation_model:
                        label += f"-translation-{translation_model}"
                    if glossing_model:
                        label += f"-glossing-{glossing_model}"

                    cases.append((label, action, language, translation_model, glossing_model, class_name))
                    seen.add(case_key)
                    covered_per_action[action].add(class_name)
    finally:
        _restore_factories(originals)

    return cases


def _display(value) -> str:
    if value is None:
        return "<default>"
    return str(value)


def _print_box(title: str, rows: list[tuple[str, str]]) -> None:
    key_width = max([len(key) for key, _ in rows] + [0])
    value_width = max([len(value) for _, value in rows] + [len(title), 0])
    inner_width = key_width + value_width + 5
    border = "+" + "-" * (inner_width + 2) + "+"

    print()
    print(border)
    print("| " + title.center(inner_width) + " |")
    print(border)
    for key, value in rows:
        print(f"| {key.ljust(key_width)} : {value.ljust(value_width)} |")
    print(border)


def _print_cases_summary(cases: list[tuple]) -> None:
    rows = []
    for label, action, language, translation_model, glossing_model, expected_class_name in cases:
        rows.append(
            (
                label,
                (
                    f"action={action}, language={language}, "
                    f"translation={_display(translation_model)}, "
                    f"glossing={_display(glossing_model)}, "
                    f"expected={expected_class_name}"
                ),
            )
        )
    _print_box("Discovered Real Strategy Cases", rows)


STRATEGY_CASES = _discover_all_cases()


pytestmark = pytest.mark.integration


def test_print_discovered_real_strategy_cases():
    _print_cases_summary(STRATEGY_CASES)
    assert STRATEGY_CASES


@pytest.mark.parametrize(
    ("label", "action", "language", "translation_model", "glossing_model", "expected_class_name"),
    STRATEGY_CASES,
    ids=[case[0] for case in STRATEGY_CASES],
)
def test_strategy_factory_builds_correct_strategy(
    label, action, language, translation_model, glossing_model, expected_class_name
):
    _print_box(
        "Real Strategy Case",
        [
            ("case", label),
            ("action", action),
            ("language", language),
            ("translation_model", _display(translation_model)),
            ("glossing_model", _display(glossing_model)),
            ("expected_class", expected_class_name),
            ("expected_method", METHOD_BY_ACTION[action]),
            ("secrets_path", str(SECRETS_PATH)),
        ],
    )

    load_dotenv(SECRETS_PATH, override=True)

    print("[real-strategy-test] Building real strategy...")
    strategy = StrategyFactory.get_strategy(
        action=action,
        language=language,
        translationModel=translation_model,
        glossingModel=glossing_model,
    )
    actual_class_name = strategy.__class__.__name__
    actual_class_path = f"{strategy.__class__.__module__}.{actual_class_name}"

    _print_box(
        "Real Strategy Result",
        [
            ("actual_class", actual_class_path),
            ("expected_class", expected_class_name),
            ("method_checked", METHOD_BY_ACTION[action]),
        ],
    )

    assert actual_class_name == expected_class_name
    assert callable(getattr(strategy, METHOD_BY_ACTION[action]))
    print("[real-strategy-test] Passed")
