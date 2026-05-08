import os
from abc import ABC, abstractmethod
import torch
from huggingface_hub import login


class TranscriptionStrategy(ABC):
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
    def load_model(self):
        """
        Load the transcription models needed for the strategy.

        Subclasses must implement this method. Typical responsibilities may include
        loading specific ASR (automatic speech recognition) models.

        This ensures each subclass is responsible for preparing its required resources.
        """
        raise NotImplementedError(
            "Subclasses must implement load_model() to initialize their transcription models. "
        )
        
    @abstractmethod
    def transcribe(self, path_to_audio: str) -> str | None:
        """
        Transcribe the given text using the model implemented by the subclass.

        Arguments:
            text (str): The input string to transcribe, which could be a result of speech recognition
                        or another input source depending on the use case.

        Returns:
            str | None: The transcribed (processed) version of the input text,
                        or None if transcription cannot be performed.

        This method must be implemented by subclasses to define their core logic.
        """
        raise NotImplementedError("Subclasses must implement transcribe()")
