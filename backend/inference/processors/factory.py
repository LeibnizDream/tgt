"""
Factory for selecting the appropriate inference data processor.

:class:`ProcessorFactory` maps ``(format, action)`` to the correct
:class:`~inference.processors.abstract.DataProcessor` subclass.

Supported formats
-----------------
- ``"labvanced"``           – processors for the Labvanced annotation pipeline.
- ``"plain"``               – generic processors for plain data formats.

Supported actions
-----------------
- ``"transcribe"``    – transcribe audio columns.
- ``"translate"``     – translate text columns.
- ``"gloss"``         – add morphological glosses.
- ``"transliterate"`` – romanise non-Latin scripts.
- ``"create columns"``– insert derived columns.
"""
from inference.processors.abstract import AbstractProcessor
from inference.processors.labvanced.transcription import TranscriptionProcessor
from inference.processors.labvanced.translation import TranslationProcessor
from inference.processors.labvanced.glossing import GlossingProcessor
from inference.processors.labvanced.transliteration import TransliteratorProcessor
from inference.processors.labvanced.ColumnCreation import ColumnCreationProcessor
from inference.processors.plain.transcription import PlainTranscriber
from inference.processors.plain.translation import PlainTranslator


class ProcessorFactory:
    """Returns a :class:`~inference.processors.abstract.DataProcessor` for a given format and action."""

    @staticmethod
    def get_processor(
        language: str,
        action: str,
        format: str,
        instruction: str | None = None,
        translationModel: str = None,
        glossingModel: str = None,
    ) -> AbstractProcessor:
        """Return the correct processor for the given *format* and *action*.

        Args:
            language: ISO 639-1 language code.
            action: One of ``"transcribe"``, ``"translate"``, ``"gloss"``,
                ``"transliterate"``, or ``"create columns"``.
            format: Target data format — ``"labvanced"`` or ``"plain"``.
            instruction: Sub-mode required for labvanced (``"sentences"``,
                ``"corrected"``, ``"automatic"``). Not used for plain format.
            translationModel: Optional model name override for translation.
            glossingModel: Optional model name override for glossing.

        Raises:
            ValueError: When *format* or *action* is not recognised, or *instruction*
                is missing for labvanced.
        """
        if format == "labvanced":
            if not instruction:
                raise ValueError("instruction is required for labvanced format")
            return ProcessorFactory._get_labvanced(
                language, action, instruction, translationModel, glossingModel
            )
        elif format == "plain":
            return ProcessorFactory._get_plain(language, action, translationModel)
        else:
            raise ValueError(f"Unknown format: {format!r}")

    @staticmethod
    def _get_labvanced(
        language: str,
        action: str,
        instruction: str,
        translationModel: str = None,
        glossingModel: str = None,
    ) -> AbstractProcessor:
        if action == "transcribe":
            return TranscriptionProcessor(language, instruction)
        elif action == "translate":
            return TranslationProcessor(language, instruction, translationModel)
        elif action == "gloss":
            return GlossingProcessor(language, instruction, translationModel, glossingModel)
        elif action == "transliterate":
            return TransliteratorProcessor(language, instruction)
        elif action == "create columns":
            return ColumnCreationProcessor(language, instruction)
        else:
            raise ValueError(f"No labvanced processor for action: {action!r}")

    @staticmethod
    def _get_plain(
        language: str,
        action: str,
        translationModel: str = None,
    ) -> AbstractProcessor:
        if action == "transcribe":
            return PlainTranscriber(language, instruction=None)
        elif action == "translate":
            return PlainTranslator(language, instruction=None, translationModel=translationModel)
        else:
            raise ValueError(f"No plain processor for action: {action!r}")