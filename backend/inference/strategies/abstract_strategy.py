"""
Abstract base for all inference strategies.
"""
import os
from abc import ABC, abstractmethod
import torch
from huggingface_hub import login


class AbstractStrategy(ABC):
    """Common base for all strategies.

    Device selection (CUDA vs CPU) and HuggingFace authentication are handled
    here once.  run_strategy processes the batch item by item by default;
    LLM subclasses override it to send the whole batch in one call.
    """

    def __init__(self, language_code: str):
        self.language_code = language_code.lower()
        cuda_available = torch.cuda.is_available()
        cudnn_available = torch.backends.cudnn.is_available() if cuda_available else False
        self.device = "cuda" if (cuda_available and cudnn_available) else "cpu"

        token = os.getenv("HUGGING_KEY", "").strip()
        if token:
            login(token=token)

        self.load_model()

    @abstractmethod
    def load_model(self) -> None: ...

    @abstractmethod
    def _run_one(self, text: str) -> str | None: ...

    def run_strategy(self, todo_items: list, _examples: list = None, progress_cb=None) -> dict:
        """Process a batch item by item. LLM subclasses override for whole-batch calls."""
        result = {}
        total = len(todo_items)
        for done, item in enumerate(todo_items, 1):
            result[item["id"]] = self._run_one(item["text"])
            if progress_cb:
                progress_cb(done, total)
        return result
