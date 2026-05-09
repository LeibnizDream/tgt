import importlib.util
from pathlib import Path


MATRIX_TEST_PATH = (
    Path(__file__).resolve().parent
    / "integration"
    / "test_real_processor_factory_matrix.py"
)


def _load_matrix_module():
    spec = importlib.util.spec_from_file_location("real_processor_factory_matrix", MATRIX_TEST_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_real_processor_matrix_covers_factory_returns():
    matrix = _load_matrix_module()
    covered = matrix._covered_processor_classes_by_format()
    expected = matrix._expected_processor_classes_by_format()
    formats = sorted(set(expected) | set(covered))
    missing = {
        format_name: expected.get(format_name, set()) - covered.get(format_name, set())
        for format_name in formats
    }
    unexpected = {
        format_name: covered.get(format_name, set()) - expected.get(format_name, set())
        for format_name in formats
    }

    matrix._print_box(
        "Processor Matrix Coverage Details",
        [
            (
                format_name,
                (
                    f"expected=[{', '.join(sorted(expected.get(format_name, set()))) or '-'}] | "
                    f"covered=[{', '.join(sorted(covered.get(format_name, set()))) or '-'}] | "
                    f"missing=[{', '.join(sorted(missing[format_name])) or '-'}] | "
                    f"unexpected=[{', '.join(sorted(unexpected[format_name])) or '-'}]"
                ),
            )
            for format_name in formats
        ],
    )

    matrix._print_box(
        "Factory Returns Covered By Test",
        [
            (
                format_name,
                ", ".join(sorted(classes)),
            )
            for format_name, classes in expected.items()
        ],
    )

    failures = []
    for format_name in formats:
        if missing[format_name]:
            failures.append(
                f"{format_name}: missing coverage for {', '.join(sorted(missing[format_name]))}"
            )
        if unexpected[format_name]:
            failures.append(
                f"{format_name}: matrix contains unexpected processors {', '.join(sorted(unexpected[format_name]))}"
            )

    assert not failures, "\n".join(failures)
