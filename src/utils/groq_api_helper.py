import os
from groq import Groq
import logging
import json
import tiktoken

logger = logging.getLogger(__name__)

def create_groq_client():
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    return Groq(api_key=api_key)

def count_tokens(text, model="llama3-groq-70b-8192-tool-use-preview"):
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def groq_api_call(messages, model="llama3-groq-70b-8192-tool-use-preview", max_tokens=None):
    client = create_groq_client()
    try:
        # Ensure max_tokens doesn't exceed the 8k limit
        max_tokens = min(max_tokens or 7500, 7500)  # Leave some room for the prompt
        
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
        
        logger.info(f"Raw response from Groq API: {response_content}")
        return response_content
    except Exception as e:
        logger.error(f"Error making API call to Groq: {str(e)}")
        logger.error(f"Error details: {e.args}")
        raise