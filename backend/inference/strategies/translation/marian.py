from inference.strategies.abstract_strategy import AbstractStrategy
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class MarianStrategy(AbstractStrategy):
    
    def __init__(language):
        super().__init__(language)

    def load_model(self):
            """
            Attempt to load a MarianMT model for <language_code>→en.
            If it fails, _marian_model and _marian_tokenizer stay as None.
            """
            if self.language_code == "yo":
                model_name = "Helsinki-NLP/opus-mt-mul-en"
            elif self.language_code == "pt":
                model_name = "Helsinki-NLP/opus-mt-tc-big-en-pt"
            else:
                model_name = f"Helsinki-NLP/opus-mt-{self.language_code}-en"
            self._marian_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._marian_model = (
                AutoModelForSeq2SeqLM.from_pretrained(model_name)
                .to(self.device)
            )

    def _run_one(self, text: str) -> str:
            """Run a Marian forward pass. Raises on any failure."""
            if not self._marian_model or not self._marian_tokenizer:
                raise RuntimeError(
                    "Marian model or tokenizer not initialized. "
                    "Call _init_marian_model() before translating."
                )

            inputs = self._marian_tokenizer.encode(text, return_tensors="pt")
            outputs = self._marian_model.generate(inputs)
            return self._marian_tokenizer.decode(outputs[0], skip_special_tokens=True)