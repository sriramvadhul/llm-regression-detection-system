import os

from dotenv import load_dotenv
from google import genai


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY was not found.")


client = genai.Client(api_key=api_key)


interaction = client.interactions.create(
    model="gemini-3.5-flash",
    input="Say hello and tell me that you are connected successfully."
)


print(interaction.output_text)