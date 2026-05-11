import huggingface_hub
import torch
import whisperx
from huggingface_hub import hf_hub_download as _original_hf_download
from inference.strategies.abstract_strategy import AbstractStrategy
from whisperx.diarize import DiarizationPipeline

# Patch torch.load FIRST
_original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = patched_load

# Patch hf_hub_download BEFORE whisperx imports it
def patched_hf_download(*args, **kwargs):
    if 'use_auth_token' in kwargs:
        kwargs['token'] = kwargs.pop('use_auth_token')
    return _original_hf_download(*args, **kwargs)

huggingface_hub.hf_hub_download = patched_hf_download


class WhisperxStrategy(AbstractStrategy):
    def __init__(self, *args, **kwargs):
        """
        Initialize the Whisperx transcription strategy.	"""
        super().__init__(*args, **kwargs)
        self.batch_size = kwargs.get('batch_size', 8)
        cuda_available = torch.cuda.is_available()
        cudnn_available = torch.backends.cudnn.is_available() if cuda_available else False

        if cuda_available and cudnn_available:
            device = "cuda"
            cudnn_version = torch.backends.cudnn.version()
            print(f"Using CUDA with cuDNN {cudnn_version}")
        else:
            device = "cpu"
            print(f"Using CPU (CUDA available: {cuda_available}, cuDNN available: {cudnn_available})")

        self.device = device
    
    def load_model(self):
        try:
            self.model = whisperx.load_model("large-v2", self.device, compute_type="float16", language=self.language_code)
        except Exception as e:
            print(f"float16 failed: {e}, falling back to int8")
            self.model = whisperx.load_model("large-v2", self.device, compute_type="int8", language=self.language_code)
        print(f"Whisperx model loaded on device {self.device}")

    def _run_one(self, path_to_audio):
        audio = whisperx.load_audio(path_to_audio)
        result = self.model.transcribe(audio, batch_size=self.batch_size, language=self.language_code)

        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"],
            device=self.device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, self.device)

        diarize_model = DiarizationPipeline(
            device=self.device
        )
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)


        full_sentences, buffer_speaker, buffer_text = [], None, ""
        for seg in result["segments"]:
            spk = seg.get("speaker", buffer_speaker)
            if spk is None:
                continue
            txt = seg["text"].strip()

            if buffer_speaker is None:
                buffer_speaker, buffer_text = spk, txt
            elif spk == buffer_speaker:
                buffer_text += " " + txt
            else:
                full_sentences.append(f"{buffer_speaker}: {buffer_text}")
                buffer_speaker, buffer_text = spk, txt

        if buffer_speaker:
            full_sentences.append(f"{buffer_speaker}: {buffer_text}")

        joined_text = "  ".join(full_sentences)

        print('joined_text: ', joined_text)
        
        return joined_text
    
