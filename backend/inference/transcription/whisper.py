import whisper
from inference.transcription.abstract import TranscriptionStrategy

class WhisperStrategy(TranscriptionStrategy):

    def load_model(self):
         self.model = whisper.load_model("large-v3", self.device)

    def transcribe(self, path_to_audio):
        res = self.model.transcribe(path_to_audio, language=self.language_code)
        return res["text"]