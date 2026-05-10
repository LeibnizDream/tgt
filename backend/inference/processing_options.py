
from typing import Optional
from pydantic import BaseModel

class ProcessingOptions(BaseModel):
    language: str
    action: str
    format: Optional[str] = None
    instruction: Optional[str] = None
    model: Optional[str] = None