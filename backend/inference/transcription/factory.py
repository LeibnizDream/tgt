from inference.transcription.abstract import TranscriptionStrategy
from inference.transcription.whisperx import WhisperxStrategy
from inference.transcription.whisper import WhisperStrategy
from inference.transcription.bengali import BengaliStrategy
from inference.transcription.vietnamese import VietnameseStrategy
from inference.transcription.thai import ThaiTranscriptionStrategy

class TranscriptionStrategyFactory:
    @staticmethod
    def get_strategy(language_code: str) -> TranscriptionStrategy:
        if language_code in ['en', 'fr', 'de', 'es', 'it', 'ja', 'zh', 'nl',
                             'uk', 'pt', 'ar', 'cs', 'ru', 'pl', 'hu', 'fi',
                             'fa', 'el', 'tr', 'da', 'he', 'ko', 'ur', 
                             'te', 'hi', 'ca', 'ml', 'ka', 'tl', 'ro']:
            return WhisperxStrategy(language_code)
        elif language_code in ['th']:
            return ThaiTranscriptionStrategy(language_code)
        elif language_code in ['vi']:
            return VietnameseStrategy(language_code)
        elif language_code in ['et']:
            return WhisperStrategy(language_code)
        elif language_code in ['bn']:
            return BengaliStrategy(language_code)
        
        else:
            raise ValueError(f"No transcription strategy available for language code: {language_code}")
