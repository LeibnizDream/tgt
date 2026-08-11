# Copyright (C) 2026 Leibniz-Zentrum Allgemeine Sprachwissenschaft
# Developed as part of the ERC-funded LeibnizDream project.
# Licensed under the GNU General Public License v3.0 or later.
# See LICENSE for details.

from inference.strategies.translation.M2M100 import M2M100Strategy


class CustomTranslationStrategy(M2M100Strategy):

    def __init__(self, language_code: str, model_name: str):
        self.model_name = model_name
        super().__init__(language_code)

    def load_model(self) -> None:
        super().load_model(model_path=f"models/translation/{self.model_name}")
