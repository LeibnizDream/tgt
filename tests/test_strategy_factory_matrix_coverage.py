import importlib.util
from pathlib import Path


MATRIX_TEST_PATH = (
    Path(__file__).resolve().parent
    / "integration"
    / "test_real_strategy_factory_matrix.py"
)


def _load_matrix_module():
    spec = importlib.util.spec_from_file_location("real_strategy_factory_matrix", MATRIX_TEST_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_real_strategy_matrix_covers_factory_returns():
    matrix = _load_matrix_module()
    covered = matrix._covered_strategy_classes_by_action()
    expected = matrix._expected_strategy_classes_by_action()
    actions = sorted(set(expected) | set(covered))
    missing = {
        action: expected.get(action, set()) - covered.get(action, set())
        for action in actions
    }
    unexpected = {
        action: covered.get(action, set()) - expected.get(action, set())
        for action in actions
    }

    matrix._print_box(
        "Strategy Matrix Coverage Details",
        [
            (
                action,
                (
                    f"expected=[{', '.join(sorted(expected.get(action, set()))) or '-'}] | "
                    f"covered=[{', '.join(sorted(covered.get(action, set()))) or '-'}] | "
                    f"missing=[{', '.join(sorted(missing[action])) or '-'}] | "
                    f"unexpected=[{', '.join(sorted(unexpected[action])) or '-'}]"
                ),
            )
            for action in actions
        ],
    )

    matrix._print_box(
        "Factory Returns Covered By Test",
        [
            (
                action,
                ", ".join(sorted(classes)),
            )
            for action, classes in expected.items()
        ],
    )

    if matrix.UNCOVERED_BY_DESIGN_BY_ACTION:
        matrix._print_box(
            "Factory Returns Excluded By Design",
            [
                (
                    action,
                    ", ".join(sorted(classes)),
                )
                for action, classes in matrix.UNCOVERED_BY_DESIGN_BY_ACTION.items()
            ],
        )

    failures = []
    for action in actions:
        if missing[action]:
            failures.append(
                f"{action}: missing coverage for {', '.join(sorted(missing[action]))}"
            )
        if unexpected[action]:
            failures.append(
                f"{action}: matrix contains unexpected strategies {', '.join(sorted(unexpected[action]))}"
            )

    assert not failures, "\n".join(failures)
