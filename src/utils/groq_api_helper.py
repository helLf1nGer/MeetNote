import logging
import os

import tiktoken
from groq import Groq

from .config_manager import ConfigManager
from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)
config_manager = ConfigManager()

DEFAULT_MODEL = 'llama-3.3-70b-versatile'

# Applied here rather than around a caller's batch loop: this is the function
# that actually reaches the API, so it is the only place a limit has any effect.
_rate_limiter = RateLimiter(max_calls=30, period=60)


def create_groq_client():
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    return Groq(api_key=api_key)


def get_default_model():
    """The chat model configured for LLM-based combiners."""
    return config_manager.get('combiner.model', DEFAULT_MODEL)


def count_tokens(text, model=None):
    """
    Approximate token count.

    Groq's Llama models do not use cl100k_base, so this is an estimate used for
    chunking headroom rather than an exact count.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


@_rate_limiter
def groq_api_call(messages, model=None, max_tokens=None):
    client = create_groq_client()
    model = model or get_default_model()
    try:
        max_tokens = min(max_tokens or 7500, 7500)  # Leave room for the prompt

        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=0.7,
            top_p=1,
            stream=False,
            stop=None,
            max_tokens=max_tokens
        )
        response_content = chat_completion.choices[0].message.content

        logger.debug("Raw response from Groq API: %s", response_content)
        return response_content
    except Exception as e:
        logger.error("Error making API call to Groq (model=%s): %s", model, e)
        raise
