"""
Abstract base for all translation strategies.
"""
import os
from abc import ABC, abstractmethod
from huggingface_hub import login


class TranslationStrategy(ABC):
    """Interface for all translation strategies.

    HuggingFace login is performed in __init__ so that gated-model access
    errors surface immediately at construction time rather than when the first
    batch is sent.  Subclasses that don't use HuggingFace (e.g. LLM-based
    ones using Gemini/Ollama) inherit this login harmlessly.

    Subclasses must implement load_model() and translate(text) -> str | None.
    """

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
    def load_model(self) -> None:
        """Initialize the translation model and any supporting resources."""
        raise NotImplementedError

    @abstractmethod
    def translate(self, text: str) -> str | None:
        """Translate text.  Returns None if translation cannot be performed."""
        raise NotImplementedError
