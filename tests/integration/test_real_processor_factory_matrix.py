import ast
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from inference.processors.processor_factory import ProcessorFactory


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
PROCESSOR_FACTORY_SOURCE = (
    BACKEND_ROOT / "inference" / "processors" / "processor_factory.py"
)

HELPER_BY_FORMAT = {
    "labvanced": "_get_labvanced",
    "plain": "_get_plain",
}

PROCESSOR_CASE_INPUTS = [
    ("labvanced-transcribe", "labvanced", "transcribe", "de", "sentences", None, None),
    ("labvanced-translate", "labvanced", "translate", "de", "sentences", None, None),
    ("labvanced-gloss", "labvanced", "gloss", "de", "sentences", None, None),
    ("labvanced-transliterate", "labvanced", "transliterate", "ja", "sentences", None, None),
    ("labvanced-create-columns", "labvanced", "create columns", "de", "sentences", None, None),
    ("plain-transcribe", "plain", "transcribe", "de", None, None, None),
    ("plain-translate", "plain", "translate", "de", None, None, None),
    ("plain-gloss", "plain", "gloss", "de", None, None, None),
]


class ProbeStrategy:
    def transcribe(self, path: str) -> str:
        return f"transcribed:{path}"

    def translate(self, text: str) -> str:
        return f"translated:{text}"

    def gloss(self, text: str) -> str:
        return f"glossed:{text}"

    def transliterate(self, text: str) -> str:
        return f"transliterated:{text}"


def _get_processor_returns_by_format() -> dict[str, set[str]]:
    tree = ast.parse(PROCESSOR_FACTORY_SOURCE.read_text(encoding="utf-8"))
    expected = {format_name: set() for format_name in HELPER_BY_FORMAT}

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue

        format_name = next(
            (
                format_name
                for format_name, helper_name in HELPER_BY_FORMAT.items()
                if node.name == helper_name
            ),
            None,
        )
        if format_name is None:
            continue

        for child in ast.walk(node):
            if not isinstance(child, ast.Return):
                continue
            if not isinstance(child.value, ast.Call):
                continue
            if isinstance(child.value.func, ast.Name):
                class_name = child.value.func.id
            elif isinstance(child.value.func, ast.Attribute):
                class_name = child.value.func.attr
            else:
                continue
            if class_name and class_name[0].isupper():
                expected[format_name].add(class_name)

    return expected


def _expected_processor_classes_by_format() -> dict[str, set[str]]:
    return _get_processor_returns_by_format()


def _covered_processor_classes_by_format() -> dict[str, set[str]]:
    covered = {format_name: set() for format_name in HELPER_BY_FORMAT}
    for _, format_name, _, _, _, _, _, expected_class_name in PROCESSOR_CASES:
        covered[format_name].add(expected_class_name)
    return covered


def _make_probe_class(class_name: str):
    class ProbeProcessor:
        def __init__(self, *args, **kwargs):
            self.__processor_class_name__ = class_name

    ProbeProcessor.__name__ = class_name
    return ProbeProcessor


def _patch_processor_factory_returns():
    import inference.processors.processor_factory as processor_factory_module

    originals = []
    for class_names in _expected_processor_classes_by_format().values():
        for class_name in class_names:
            if hasattr(processor_factory_module, class_name):
                originals.append(
                    (
                        processor_factory_module,
                        class_name,
                        getattr(processor_factory_module, class_name),
                    )
                )
                setattr(processor_factory_module, class_name, _make_probe_class(class_name))
    return originals


def _restore_processor_factory_returns(originals) -> None:
    for module, class_name, original in originals:
        setattr(module, class_name, original)


def _discover_all_cases() -> list[tuple]:
    cases = []
    originals = _patch_processor_factory_returns()
    try:
        for label, format_name, action, language, instruction, translation_model, glossing_model in PROCESSOR_CASE_INPUTS:
            processor = ProcessorFactory.get_processor(
                language=language,
                action=action,
                format=format_name,
                instruction=instruction,
                translationModel=translation_model,
                glossingModel=glossing_model,
            )
            cases.append(
                (
                    label,
                    format_name,
                    action,
                    language,
                    instruction,
                    translation_model,
                    glossing_model,
                    processor.__processor_class_name__,
                )
            )
    finally:
        _restore_processor_factory_returns(originals)

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
    for label, format_name, action, language, instruction, translation_model, glossing_model, expected_class_name in cases:
        rows.append(
            (
                label,
                (
                    f"format={format_name}, action={action}, language={language}, "
                    f"instruction={_display(instruction)}, "
                    f"translation={_display(translation_model)}, "
                    f"glossing={_display(glossing_model)}, "
                    f"expected={expected_class_name}"
                ),
            )
        )
    _print_box("Discovered Real Processor Cases", rows)


PROCESSOR_CASES = _discover_all_cases()


pytestmark = pytest.mark.integration


def test_print_discovered_real_processor_cases():
    _print_cases_summary(PROCESSOR_CASES)
    assert PROCESSOR_CASES


@pytest.mark.parametrize(
    (
        "label",
        "format_name",
        "action",
        "language",
        "instruction",
        "translation_model",
        "glossing_model",
        "expected_class_name",
    ),
    PROCESSOR_CASES,
    ids=[case[0] for case in PROCESSOR_CASES],
)
def test_processor_factory_builds_correct_processor(
    monkeypatch,
    label,
    format_name,
    action,
    language,
    instruction,
    translation_model,
    glossing_model,
    expected_class_name,
):
    from inference.processors import abstract_processor

    monkeypatch.setattr(
        abstract_processor.StrategyFactory,
        "get_strategy",
        staticmethod(lambda *args, **kwargs: ProbeStrategy()),
    )

    _print_box(
        "Real Processor Case",
        [
            ("case", label),
            ("format", format_name),
            ("action", action),
            ("language", language),
            ("instruction", _display(instruction)),
            ("translation_model", _display(translation_model)),
            ("glossing_model", _display(glossing_model)),
            ("expected_class", expected_class_name),
        ],
    )

    print("[real-processor-test] Building real processor...")
    processor = ProcessorFactory.get_processor(
        language=language,
        action=action,
        format=format_name,
        instruction=instruction,
        translationModel=translation_model,
        glossingModel=glossing_model,
    )
    actual_class_name = processor.__class__.__name__
    actual_class_path = f"{processor.__class__.__module__}.{actual_class_name}"

    _print_box(
        "Real Processor Result",
        [
            ("actual_class", actual_class_path),
            ("expected_class", expected_class_name),
        ],
    )

    assert actual_class_name == expected_class_name
    assert hasattr(processor, "process")
    print("[real-processor-test] Passed")
