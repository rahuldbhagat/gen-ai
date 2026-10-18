import os

from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY not found. Copy .env.example to .env and paste in your key."
    )

print("Environment looks good. Key found, starts with:", api_key[:7] + "...")