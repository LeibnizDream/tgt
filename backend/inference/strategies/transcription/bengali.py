# Copyright (C) 2026 Leibniz-Zentrum Allgemeine Sprachwissenschaft
# Developed as part of the ERC-funded LeibnizDream project.
# Licensed under the GNU General Public License v3.0 or later.
# See LICENSE for details.

from inference.strategies.abstract_strategy import AbstractStrategy
from transformers import pipeline
from utils.functions import find_ffmpeg

class BengaliTranscriptionStrategy(AbstractStrategy):
        
    def load_model(self):
        ffmpeg_path = find_ffmpeg()
        self.whisper_asr = pipeline(
            "automatic-speech-recognition",
            model="mozilla-ai/whisper-large-v3-bn",
            device=self.device
        )

    def run_strategy(self, path_to_audio: str):
        self.whisper_asr.model.config.forced_decoder_ids = (
            self.whisper_asr.tokenizer.get_decoder_prompt_ids(language=self.language_code, task="transcribe")
        )

        result = self.whisper_asr(path_to_audio, return_timestamps=True)
        result = result.get("text", "").strip()
        if not result:
            print(f"[WARNING] Empty transcription for: {path_to_audio}")
            print(f"[DEBUG] Raw result: {result}")
        else:
            print(f"[SUCCESS] Transcribed {len(result)} chars: {result[:100]}")
            print(result)
        return result