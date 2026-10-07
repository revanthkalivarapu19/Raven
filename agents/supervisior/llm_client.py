"""
LLM Client for Supervisor Agent.
Wraps the call to the local Ollama model.
"""

import logging
import ollama

logger = logging.getLogger(__name__)

def call_supervisor_llm(prompt: str, model: str = "llama3.1:8b") -> str:
    """
    Send the prompt to the local Ollama model using the ollama Python package (ollama.chat),
    with a low temperature (e.g. 0.2) for more consistent structured output.
    Return the raw text response from the model.
    
    Raises:
        RuntimeError: If the call fails (e.g. Ollama not running).
    """
    try:
        logger.info(f"Calling Ollama model {model}...")
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.2}
        )
        logger.info("Ollama call completed.")
        return response['message']['content']
    except Exception as e:
        logger.error(f"Failed to call Ollama model {model}: {e}")
        raise RuntimeError(f"Ollama call failed: {e}") from e
