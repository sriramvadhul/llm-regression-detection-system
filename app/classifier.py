import json

import ollama

from models import PromptConfig, ClassificationOutput


def classify_email(
    email: str,
    prompt_config: PromptConfig
) -> ClassificationOutput:

    prompt = f"""
{prompt_config.system_prompt}

Customer email:

{email}
"""

    response = ollama.chat(
        model=prompt_config.model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        format="json"
    )

    result = response["message"]["content"].strip()

    data = json.loads(result)

    return ClassificationOutput(**data)