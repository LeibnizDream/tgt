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

    def __init__(self, language_code: str, transliterationModel: str = None, device: str = "cpu"):
        self.language_code = language_code
        self.transliterationModel = transliterationModel
        self.device = device
        self.load_model()

    def load_model(self) -> None:
        """Optional hook for loading models.  No-op by default."""
        pass

    def transliterate(self, items: list, examples: list = None, progress_cb=None) -> dict:
        """Transliterate a batch of items one-by-one. LLM subclasses override for batch calls."""
        result = {}
        total = len(items)
        for done, item in enumerate(items, 1):
            result[item["id"]] = self._transliterate_one(item["text"])
            if progress_cb:
                progress_cb(done, total)
        return result

    @abstractmethod
    def _transliterate_one(self, text: str) -> str | None:
        """Transliterate a single text. Returns None if transliteration cannot be performed."""
        raise NotImplementedError
