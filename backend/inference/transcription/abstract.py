import os
from abc import ABC, abstractmethod
from pathlib import Path
from dotenv import load_dotenv
import torch
from huggingface_hub import login

_this_file = Path(__file__).resolve()
parent_dir = _this_file.parent.parent.parent

class TranscriptionStrategy(ABC):
    def __init__(self, language_code: str):
        print("Initializing inside TranscriptionStrategy")
        self.language_code = language_code.lower()
        mps_available = torch.backends.mps.is_available()
        cuda_available = torch.cuda.is_available()
        cudnn_available = torch.backends.cudnn.is_available() if cuda_available else False

        if cuda_available and cudnn_available:
            device = "cuda"
            cudnn_version = torch.backends.cudnn.version()
            print(f"Using CUDA with cuDNN {cudnn_version}")
        elif mps_available:
            device = "mps"
            print("Using Apple Silicon GPU (MPS)")
        else:
            device = "cpu"
            print(f"Using CPU (CUDA available: {cuda_available}, cuDNN available: {cudnn_available})")

        self.device = device
        print(f"TranscriptionStrategy initialized with device: {self.device}, language_code: {self.language_code}")
        self.hugging_key = self._load_hugging_face_token()
        login(token=self.hugging_key)
        
        self.load_model()
    
    def _load_hugging_face_token(self):
        token = os.getenv("HUGGING_KEY", "").strip()
        if not token:
            secrets_path = os.path.join(parent_dir, 'materials', 'secrets.env')
            print(f"Loading Hugging Face token from {secrets_path}")
            if os.path.exists(secrets_path):
                load_dotenv(secrets_path, override=True)
                token = os.getenv("HUGGING_KEY", "").strip()
        if not token:
            raise ValueError("Hugging Face key not found. Set it in HUGGING_KEY")
        return token
    
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
