from inference.strategies.translation.M2M100 import M2M100Strategy


class CustomTranslationStrategy(M2M100Strategy):

    def __init__(self, language_code: str, model_name: str):
        self.model_name = model_name  # set before super().__init__ calls load_model()
        super().__init__(language_code)

    def load_model(self) -> None:
        super().load_model(model_path=f"models/translation/{self.model_name}")
