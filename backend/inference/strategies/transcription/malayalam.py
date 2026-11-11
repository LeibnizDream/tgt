# Copyright (C) 2026 Leibniz-Zentrum Allgemeine Sprachwissenschaft
# Developed as part of the ERC-funded LeibnizDream project.
# Licensed under the GNU General Public License v3.0 or later.
# See LICENSE for details.

from faster_whisper import WhisperModel

from inference.strategies.abstract_strategy import AbstractStrategy


class MalayalamTranscriptionStrategy(AbstractStrategy):
    """Transcribe Malayalam using the Vegam CTranslate2 Whisper model."""

    def load_model(self):
        self.model = WhisperModel(
            "smcproject/vegam-whisper-medium-ml",
            device=self.device,
            compute_type="float16" if self.device == "cuda" else "int8",
        )

    def run_strategy(self, path_to_audio: str) -> str:
        segments, _ = self.model.transcribe(
            path_to_audio,
            language="ml",
            task="transcribe",
            beam_size=5,
        )
        return " ".join(text for segment in segments if (text := segment.text.strip()))
