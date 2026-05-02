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
You are a helpful assistant.

Context:
{context}

Question:
{query}

Answer clearly based only on the context.
"""

def get_llm_answer(context: str, query: str):
    logger.info(f"Team 2 (LLM): Sending prompt to OpenRouter...")
    prompt = build_prompt(context, query)
    try:
        response = client.chat.completions.create(
            model="openrouter/free",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM Error: {e}")
        return "Sorry, I am having trouble connecting to the AI brain right now."
