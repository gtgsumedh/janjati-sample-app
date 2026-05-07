import os
import logging
from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger(__name__)

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1"
)


def build_prompt(context: str, query: str):
    return f"""
You are a helpful assistant with knowledge of tribal heroes and history.
Use only the information provided in the context below.
If the answer cannot be found in the context, say "I don't know".

Context:
{context}

Question:
{query}

Answer clearly and concisely.
"""


def get_llm_answer(context: str, query: str):
    logger.info("LLM Service: sending prompt to OpenRouter...")
    prompt = build_prompt(context, query)

    try:
        response = client.chat.completions.create(
            model="openrouter/free",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for tribal history."},
                {"role": "user", "content": prompt}
            ]
        )
        answer = response.choices[0].message.content or ""
        return answer.strip()
    except Exception as e:
        logger.error(f"LLM Error: {e}")
        return "Sorry, I am having trouble connecting to the AI brain right now."
