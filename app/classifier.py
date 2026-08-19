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

        format=ClassificationOutput.model_json_schema(),

        options={
            "temperature": 0
        }
    )

    result = response["message"]["content"]

    return ClassificationOutput.model_validate_json(
        result
    )