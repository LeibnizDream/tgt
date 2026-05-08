import os
from abc import ABC, abstractmethod
from huggingface_hub import login


class TranslationStrategy(ABC):
    def __init__(self, language_code: str, translationModel: str = None, device: str = "cpu"):
        self.language_code = language_code.lower()
        self.translationModel = translationModel
        self.device = device

        token = os.getenv("HUGGING_KEY", "").strip()
        if not token:
            raise ValueError("HUGGING_KEY not set — load secrets before starting the worker")
        login(token=token)

        self.load_model()

    
    @abstractmethod
    def load_model(self):
        """
        Load the translation models. 
        """
        raise NotImplementedError(
            "Subclasses must implement load_model() to initialize their translation models."
        )
        
    @abstractmethod
    def translate(self, text: str) -> str | None:
        """
        Return a non-None string on success, or None on failure.
        """
        raise NotImplementedError("Subclasses must implement translate()")
