from pathlib import Path
import yaml

from models import PromptConfig


def load_prompt(prompt_path: str) -> PromptConfig:
    path = Path(prompt_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {prompt_path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    return PromptConfig(**data)