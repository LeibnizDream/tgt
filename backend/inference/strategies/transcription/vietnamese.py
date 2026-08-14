# Copyright (C) 2026 Leibniz-Zentrum Allgemeine Sprachwissenschaft
# Developed as part of the ERC-funded LeibnizDream project.
# Licensed under the GNU General Public License v3.0 or later.
# See LICENSE for details.

from inference.strategies.abstract_strategy import AbstractStrategy
from transformers import pipeline
from utils.functions import find_ffmpeg

class VietnameseTranscriptionStrategy(AbstractStrategy):
    
    def load_model(self):
        ffmpeg_path = find_ffmpeg()
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