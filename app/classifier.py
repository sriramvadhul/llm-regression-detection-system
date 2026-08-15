import json
import os

from dotenv import load_dotenv
from google import genai

from models import PromptConfig, ClassificationOutput


load_dotenv()


api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found. "
        "Please check your .env file."
    )


client = genai.Client(api_key=api_key)


def classify_email(
    email: str,
    prompt_config: PromptConfig
) -> ClassificationOutput:

    prompt = f"""
{prompt_config.system_prompt}

Customer email:

{email}
"""

    interaction = client.interactions.create(
        model=prompt_config.model,
        input=prompt
    )

    result = interaction.output_text.strip()

    # Remove Markdown code fences if Gemini returns them
    if result.startswith("```"):
        result = result.replace("```json", "")
        result = result.replace("```", "")
        result = result.strip()

    data = json.loads(result)

    return ClassificationOutput(**data)