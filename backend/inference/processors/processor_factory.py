"""
Factory for selecting the appropriate inference data processor.
"""
from inference.processors.abstract_processor import AbstractProcessor
from inference.processors.labvanced.transcription import TranscriptionProcessor
from inference.processors.labvanced.labvanced_base import LabvancedBaseProcessor
from inference.processors.labvanced.ColumnCreation import ColumnCreationProcessor
from inference.processors.plain.transcription import PlainTranscriber
from inference.processors.plain.plain_base import BasePlainProcessor#



class ProcessorFactory:

    @staticmethod
    def get_processor(options) -> AbstractProcessor:
        if options.format == "labvanced":
            if not options.instruction:
                raise ValueError("instruction is required for labvanced format")
            return ProcessorFactory._get_labvanced(options)
        elif options.format == "plain":
            return ProcessorFactory._get_plain(options)
        else:
            raise ValueError(f"Unknown format: {options.format!r}")

    @staticmethod
    def _get_labvanced(options) -> AbstractProcessor:
        action, language, instruction, model = (
            options.action, options.language, options.instruction, options.model
        )
        if action == "transcribe":
            return TranscriptionProcessor(language, action, instruction)
        elif action in ["translate", "gloss", "transliterate"]:
            return LabvancedBaseProcessor(language, action, instruction, model)
        elif action == "create columns":
            return ColumnCreationProcessor(language, "create columns", instruction)
        else:
            raise ValueError(f"No labvanced processor for action: {action!r}")

    @staticmethod
    def _get_plain(options) -> AbstractProcessor:
        action, language, model = options.action, options.language, options.model
        if action == "transcribe":
            return PlainTranscriber(language)
        elif action in ["translate", "gloss", "transliterate"]:
            return BasePlainProcessor(language, action, model)
        else:
            raise ValueError(f"No plain processor for action: {action!r}")
