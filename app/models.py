from pydantic import BaseModel


class PromptConfig(BaseModel):
    version: str
    model: str
    system_prompt: str


class ClassificationOutput(BaseModel):
    category: str
    summary: str