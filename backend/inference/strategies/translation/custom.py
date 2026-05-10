from inference.strategies.abstract_strategy import AbstractStrategy

class CustomTranslationStrategy(AbstractStrategy):
    
    def load_model(self):
        #self._init_M2M100_model(model_path=f"models/translation/{model_name}")
        self._init_M2M100_model(model_path=f"models/translation/{self.translationModel}")

    def _run_one(self, text: str) -> str | None:
        #out = self._translate_marian(text)
        out = self._translate_M2M100(text)
        return out