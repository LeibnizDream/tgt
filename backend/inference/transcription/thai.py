from transformers import pipeline
from inference.transcription.abstract import TranscriptionStrategy
import torch

class ThaiTranscriptionStrategy(TranscriptionStrategy):

    def load_model(self):
        MODEL_NAME = "biodatlab/whisper-th-large-v3-combined"

        device = 0 if torch.cuda.is_available() else "cpu"
        print(f"Loading Thai transcription model on device: {device}")

        self.pipe = pipeline(
            task="automatic-speech-recognition",
            model=MODEL_NAME,
            chunk_length_s=30,
            device=device,
        )
        
    def transcribe(self, path_to_audio: str) -> str | None:
        self.pipe.model.config.forced_decoder_ids = self.pipe.tokenizer.get_decoder_prompt_ids(
        language=self.language_code,
        task="transcribe"
        )
        text = self.pipe(path_to_audio)["text"]
        return text