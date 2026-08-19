from typing import Literal
from pydantic import BaseModel


class PromptConfig(BaseModel):
    version: str
    provider: str
    model: str
    system_prompt: str


class ClassificationOutput(BaseModel):
    category: Literal[
        "billing",
        "technical",
        "account",
        "general"
    ]
    summary: str