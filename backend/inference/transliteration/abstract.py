from abc import ABC, abstractmethod

class TransliterationStrategy(ABC):
    def __init__(self, language_code: str, transliterate_model: str = None, device: str = "cpu"):
        self.language_code = language_code
        self.transliterate_model = transliterate_model
        self.device = device
        # always safe to call—subclasses that need to can override it
        self.load_model()

    def load_model(self):
        """
        Optional hook for loading any models/resources.
        Default is a no-op; subclasses may override if needed.
        """
        pass

    @abstractmethod
    def transliterate(self, text: str) -> str:
        """
        Must be implemented by every subclass.
        """
        raise NotImplementedError("Subclasses must implement transliterate()")
