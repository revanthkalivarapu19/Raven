"""
domain_agent.py

Domain Detection Agent for the RAVEN Framework.
"""

import ollama
from loguru import logger

from .prompt import DOMAIN_PROMPT
from .parser import parse_response
from .validator import validate_response


class DomainDetectionAgent:
    """
    Domain Detection Agent

    Responsibilities:
    - Receive a claim
    - Generate prompt
    - Send prompt to Llama
    - Parse response
    - Validate response
    - Return structured JSON
    """

    def __init__(self, model_name: str = "llama3.1"):
        self.model_name = model_name

    def build_prompt(self, claim: str) -> str:
        """
        Replace the placeholder in the prompt with the actual claim.
        """
        return DOMAIN_PROMPT.format(claim=claim)

    def call_llm(self, prompt: str) -> str:
        """
        Sends the prompt to Ollama and returns the raw response.
        """

        logger.info("Sending prompt to Llama...")

        response = ollama.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        llm_output = response["message"]["content"]

        print("\n========== RAW LLM RESPONSE ==========")
        print(repr(llm_output))
        print("======================================\n")

        return llm_output

    def detect_domain(self, claim: str) -> dict:
        """
        Main function for domain detection.
        """

        try:

            # Step 1
            prompt = self.build_prompt(claim)

            # Step 2
            llm_response = self.call_llm(prompt)

            # Step 3
            parsed_response = parse_response(llm_response)

            # Step 4
            if validate_response(parsed_response):

                logger.success("Domain detected successfully.")

                return parsed_response

            logger.error("Validation failed.")

            return {
                "domain": "Unknown",
                "confidence": 0.0,
                "reason": "Validation failed."
            }

        except Exception as e:

            logger.exception(e)

            return {
                "domain": "Unknown",
                "confidence": 0.0,
                "reason": str(e)
            }