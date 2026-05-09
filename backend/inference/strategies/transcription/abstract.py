"""
Abstract base for all transcription (ASR) strategies.
"""
import os
from abc import ABC, abstractmethod
import torch
from huggingface_hub import login


class TranscriptionStrategy(ABC):
    """Interface for all ASR strategies.

    Device selection (CUDA vs CPU) and HuggingFace authentication are handled
    here so every subclass gets them for free without duplicating the logic.
    cuDNN availability is checked explicitly because CUDA can be present without
    cuDNN, which causes silent failures in some ASR libraries.

    Subclasses must implement load_model() and transcribe(path_to_audio) -> str | None.
    """

    def __init__(self, language_code: str):
        self.language_code = language_code.lower()
        cuda_available = torch.cuda.is_available()
        cudnn_available = torch.backends.cudnn.is_available() if cuda_available else False
        self.device = "cuda" if (cuda_available and cudnn_available) else "cpu"

        token = os.getenv("HUGGING_KEY", "").strip()
        if not token:
            raise ValueError("HUGGING_KEY not set — load secrets before starting the worker")
        login(token=token)

        self.load_model()

    @abstractmethod
    def load_model(self) -> None:
        """Initialize the ASR model and any supporting resources."""
        raise NotImplementedError

    @abstractmethod
    def transcribe(self, path_to_audio: str) -> str | None:
        """Transcribe the audio file at path_to_audio.  Returns None on failure."""
        raise NotImplementedError
