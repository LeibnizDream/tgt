import whisper
from inference.strategies.abstract_strategy import AbstractStrategy

class WhisperStrategy(AbstractStrategy):

    def load_model(self):
         self.model = whisper.load_model("large-v3", self.device)
         print(f"Whisper model loaded on device {self.device}")

    def _run_one(self, path_to_audio):
        res = self.model.transcribe(path_to_audio, language=self.language_code)
        text = res["text"]
        print(text)
        return text