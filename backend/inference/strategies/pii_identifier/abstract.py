"""
Abstract base for all PII (Personally Identifiable Information) identification strategies.
"""
from abc import ABC, abstractmethod


class PIIStrategy(ABC):
    """Interface for NER-based PII detection and annotation strategies.

    Subclasses must implement load_model() and identify_and_annotate(text) -> str | None.
    The annotated string should replace PII spans with placeholder tags so
    downstream processors never see raw personal data.
    """

    def __init__(self, language_code: str):
        self.lang = language_code.lower()
        self.nlp = None
        self.load_model()

    @abstractmethod
    def load_model(self) -> None:
        """Initialize the NER model used for entity detection."""
        raise NotImplementedError

    @abstractmethod
    def identify_and_annotate(self, text: str) -> str | None:
        """Detect PII in text and return an annotated copy with spans replaced.

        Returns None if annotation cannot be performed.
        """
        raise NotImplementedError
