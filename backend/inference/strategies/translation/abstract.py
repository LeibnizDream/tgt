from abc import ABC, abstractmethod
from dotenv import load_dotenv
from huggingface_hub import login
import os
from pathlib import Path

_this_file = Path(__file__).resolve()
parent_dir = _this_file.parent.parent.parent.parent


class TranslationStrategy(ABC):
    def __init__(self, language_code: str, translationModel: str = None, device: str = "cpu"):
        self.language_code = language_code.lower()
        self.translationModel = translationModel
        self.device = device
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
