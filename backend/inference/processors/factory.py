"""
Factory for selecting the appropriate inference data processor.

:class:`ProcessorFactory` maps an action string to the correct
:class:`~inference.processors.abstract.DataProcessor` subclass.

Supported actions
-----------------
- ``"transcribe"``    → :class:`~inference.processors.transcription.TranscriptionProcessor`
- ``"translate"``     → :class:`~inference.processors.translation.TranslationProcessor`
- ``"gloss"``         → :class:`~inference.processors.glossing.GlossingProcessor`
- ``"transliterate"`` → :class:`~inference.processors.transliteration.TransliteratorProcessor`
- ``"create columns"``→ :class:`~inference.processors.ColumnCreation.ColumnCreationProcessor`
"""
from inference.processors.abstract import DataProcessor
from inference.processors.transcription import TranscriptionProcessor
from inference.processors.translation import TranslationProcessor
from inference.processors.glossing import GlossingProcessor
from inference.processors.transliteration import TransliteratorProcessor
from inference.processors.ColumnCreation import ColumnCreationProcessor

class ProcessorFactory:
    """Factory that returns a :class:`~inference.processors.abstract.DataProcessor` for a given action."""

    @staticmethod
    def get_processor(language: str, action: str, instruction: str, translationModel = None, glossingModel = None) -> DataProcessor:
        """
        Return the correct processor for the given action.

        Args:
            language (str): ISO 639-1 language code.
            action (str): One of ``"transcribe"``, ``"translate"``,
                ``"gloss"``, ``"transliterate"``, or ``"create columns"``.
            instruction (str): Sub-mode hint (e.g. ``"sentences"``,
                ``"corrected"``, ``"automatic"``).
            translationModel (str | None): Optional model name override for
                translation.
            glossingModel (str | None): Optional model name override for
                glossing.

        Returns:
            DataProcessor: A configured processor instance ready to call
            :meth:`~inference.processors.abstract.DataProcessor.process`.

        Raises:
            ValueError: When *action* is not recognised.
        """
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
            raise ValueError(f"No data processor available for action: {action}")