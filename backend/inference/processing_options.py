

from pydantic import BaseModel


class ProcessingOptions(BaseModel):
    language: str
    action: str
    format: str | None = None
    instruction: str | None = None
    model: str | None = None