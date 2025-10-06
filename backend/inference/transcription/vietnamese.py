from transformers import pipeline
import os
import torch
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import login
from inference.transcription.abstract import TranscriptionStrategy

_this_file = Path(__file__).resolve()
parent_dir = _this_file.parent.parent.parent

class VietnameseStrategy(TranscriptionStrategy):
    def __init__(self, language_code: str, device: str = None):
        self.language_code = language_code.lower()
        # respect caller preference; default to cuda if available
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # pipeline expects int index for CUDA or a torch.device
        self.device = 0 if (isinstance(device, str) and device.startswith("cuda")) else device

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
    
    def load_model(self):
        self.transcriber = pipeline(
            "automatic-speech-recognition",
            model="vinai/PhoWhisper-large",
            chunk_length_s=30,
            stride_length_s=(4, 2),
            device=self.device,
        )

    def transcribe(self, audio: str) -> str | None:
        out = self.transcriber(
            audio,
            generate_kwargs={
                "language": "vi",
                "task": "transcribe",
                "temperature": 0.0,
            },
            return_timestamps=False,
        )
        text = out["text"]
        print(text)
        return text