from inference.transcription.abstract import TranscriptionStrategy

class ThaiStrategy(TranscriptionStrategy):
    def __init__(self, language_code, device = "cpu"):
        super().__init__(language_code, device)

    def load_model(self):
       pass
        
    def transcribe(self, path_to_audio: str) -> str | None:
        pass