"""
Abstract base for all transliteration strategies.
"""
from abc import ABC, abstractmethod


class TransliterationStrategy(ABC):
    """Interface for all transliteration strategies.

    load_model is a non-abstract hook rather than an abstract method because
    most transliteration strategies are rule-based and need no model loading.
    Subclasses that do load a model (e.g. a neural romanizer) override it.

    Subclasses must implement transliterate(text) -> str.
    """

    def __init__(self, language_code: str, device: str = "cpu"):
        self.language_code = language_code
        self.load_model()

    def load_model(self) -> None:
        """Optional hook for loading models.  No-op by default."""
        pass

    @abstractmethod
    def transliterate(self, text: str) -> str:
        """Convert text from the source script to Latin script."""
        raise NotImplementedError
