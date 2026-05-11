from inference.strategies.abstract_strategy import AbstractStrategy
from transformers import pipeline


class VietnameseTranscriptionStrategy(AbstractStrategy):
    
    def load_model(self):
        self.transcriber = pipeline(
            "automatic-speech-recognition",
            model="vinai/PhoWhisper-large",
            chunk_length_s=30,
            stride_length_s=(4, 2),
            device=self.device,
        )
    

    def run_strategy(self, audio: str) -> str | None:
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