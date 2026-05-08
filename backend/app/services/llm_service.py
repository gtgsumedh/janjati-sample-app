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
Please answer the following question using ONLY the context provided below.

CONTEXT:
{context}

QUESTION:
{query}

Remember: If the answer is not contained within the CONTEXT, you must refuse to answer. Do not use outside knowledge.
"""


def get_llm_answer(context: str, query: str):
    logger.info("LLM Service: sending prompt to OpenRouter...")
    prompt = build_prompt(context, query)

    try:
        response = client.chat.completions.create(
            model="openrouter/free",
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "You are a strict, factual AI assistant for the Janjati Knowledge System. "
                        "Your primary directive is to answer questions based strictly on the retrieved context. "
                        "You must never hallucinate or invent information. "
                        "If the context does not contain the answer, say exactly: "
                        "'I do not have enough information in my knowledge base to answer this question.'"
                    )
                },
                {"role": "user", "content": prompt}
            ]
        )
        
        # --- Monitoring: Log Token Usage ---
        if response.usage:
            logger.info(
                f"LLM Token Usage - Prompt: {response.usage.prompt_tokens}, "
                f"Completion: {response.usage.completion_tokens}, "
                f"Total: {response.usage.total_tokens}"
            )
            
        answer = response.choices[0].message.content or ""
        return answer.strip()
    except Exception as e:
        logger.error(f"LLM Error: {e}")
        return "Sorry, I am having trouble connecting to the AI brain right now."
