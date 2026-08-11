# Copyright (C) 2026 Leibniz-Zentrum Allgemeine Sprachwissenschaft
# Developed as part of the ERC-funded LeibnizDream project.
# Licensed under the GNU General Public License v3.0 or later.
# See LICENSE for details.

class ExampleStore:
    """Accumulates and retrieves few-shot examples across files within a job.

    Keyed by (action, target_key) so translation, glossing, and transliteration
    examples never mix.  One instance lives for the lifetime of a single job and
    is discarded afterwards — no class-level state.
    """

    def __init__(self) -> None:
        self._data: dict[tuple[str, str], list[dict]] = {}

    def add(self, source: str, action: str, target_key: str, target: str) -> None:
        self._data.setdefault((action, target_key), []).append(
            {"source": source, target_key: target}
        )

    def get(self, action: str, target_key: str, limit: int = 20) -> list[dict]:
        return self._data.get((action, target_key), [])[:limit]
