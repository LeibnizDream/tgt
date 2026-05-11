from inference.strategies.abstract_strategy import AbstractStrategy
from transformers import pipeline


class ThaiTranscriptionStrategy(AbstractStrategy):

    def load_model(self):
        MODEL_NAME = "biodatlab/whisper-th-large-v3-combined"  # specify the model name

        self.pipe = pipeline(
            task="automatic-speech-recognition",
            model=MODEL_NAME,
            chunk_length_s=30,
            device=self.device,
        )

    
    def run_strategy(self, path_to_audio: str) -> str | None:
        self.pipe.model.config.forced_decoder_ids = self.pipe.tokenizer.get_decoder_prompt_ids(
        language=self.language_code,
        task="transcribe"
        )
        text = self.pipe(path_to_audio)["text"] # give audio mp3 and transcribe text
        return text


