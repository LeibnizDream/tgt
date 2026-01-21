import whisper
from inference.transcription.abstract import TranscriptionStrategy

class WhisperStrategy(TranscriptionStrategy):
    def __init__(self, language_code, device = "cpu"):
        super().__init__(language_code, device)

    def load_model(self):
         self.model = whisper.load_model("large-v2", self.device)

    def transcribe(self, path_to_audio):
        res = self.model.transcribe(path_to_audio, language=self.language_code)
        return res["text"]