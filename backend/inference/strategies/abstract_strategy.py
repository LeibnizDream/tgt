"""
Abstract base for all inference strategies.
"""
from abc import ABC, abstractmethod

import torch


class AbstractStrategy(ABC):
    """Common base for all strategies.

    Device selection (CUDA vs CPU) and HuggingFace authentication are handled
    here once.  run_strategy processes the batch item by item by default;
    LLM subclasses override it to send the whole batch in one call.
    """
    batch_mode = False
    def __init__(self, language_code: str):
        self.language_code = language_code.lower()
        cuda_available = torch.cuda.is_available()
        cudnn_available = torch.backends.cudnn.is_available() if cuda_available else False
        self.device = "cuda" if (cuda_available and cudnn_available) else "cpu"

        self.load_model()

    @abstractmethod
    def load_model(self) -> None: ...

    @abstractmethod
    def run_strategy(self, todo) -> dict: ...
