# Copyright (C) 2026 Leibniz-Zentrum Allgemeine Sprachwissenschaft
# Developed as part of the ERC-funded LeibnizDream project.
# Licensed under the GNU General Public License v3.0 or later.
# See LICENSE for details.

"""Create a human- and machine-readable AI provenance trails for inference runs."""

import json
from datetime import UTC, datetime
from pathlib import Path

from inference.processing_options import ProcessingOptions

_MODEL_METADATA = Path(__file__).resolve().parents[2] / "materials" / "model_metadata.json"
AUDIT_FILENAME = "AI_provenance.md"


def _model_details(strategy: object, requested_model: str | None) -> dict[str, str]:
    """Return metadata for the strategy actually selected by the factory."""
    metadata = json.loads(_MODEL_METADATA.read_text(encoding="utf-8"))["strategies"]
    strategy_name = strategy.__class__.__name__
    key = strategy_name
    if strategy_name == "LLMStrategy":
        key = f"{strategy_name}:{getattr(strategy, '_model', requested_model or '').lower()}"

    details = metadata.get(key)
    if details:
        return details

    # Custom strategies are intentionally still auditable, even when their
    # metadata has not yet been added to the registry.
    return {
        "name": requested_model or strategy_name,
        "version": "custom / not specified",
        "provider": "User-provided",
    }


def _pipeline_details(strategy: object) -> str | None:
    """Return the NLP pipeline selected by a glossing strategy, if any."""
    pipeline = getattr(strategy, "audit_pipeline", None)
    return str(pipeline) if pipeline else None


def _append_entry(path: Path, entry: str) -> Path:
    if path.exists():
        path.write_text(path.read_text(encoding="utf-8").rstrip() + "\n\n" + entry, encoding="utf-8")
    else:
        path.write_text(
            "# AI-assisted processing:\n"
            "This content was processed using AI models. "
            "AI-generated or AI-assisted results may contain errors and should be reviewed.\n"
            "The following models were used:\n\n " + entry, encoding="utf-8"
        )
    return path


def start_processing_audit(
    input_directory: str | Path,
    options: ProcessingOptions,
    strategy: object | None,
) -> Path:
    """Create an audit entry next to the input before processing begins."""
    if strategy is None:
        details = {
            "name": "Not applicable",
            "version": "Not applicable",
            "provider": "Not applicable",
        }
    else:
        details = _model_details(strategy, options.model)

    pipeline = _pipeline_details(strategy) if strategy is not None else None

    started_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    entry = "\n".join(
        [
            "## Processing started",
            "",
            f"Started: {started_at}\n",
            f"Action: {options.action}\n",
            f"Language Code: {options.language}\n",
            "",
            "## Model",
            "",
            f"- Name: {details['name']}",
            f"- Version: {details['version']}",
            f"- Provider: {details['provider']}",
            *([f"- Pipeline/model: {pipeline}"] if pipeline else []),
            "",
        ]
    )
    path = Path(input_directory) / AUDIT_FILENAME
    return _append_entry(path, entry)


def complete_processing_audit(audit_path: str | Path) -> Path:
    """Append a completion marker only after the processor finishes successfully."""
    completed_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    return _append_entry(
        Path(audit_path),
        "\n".join(["## Processing completed", "", f"Completed: {completed_at}", ""]),
    )
