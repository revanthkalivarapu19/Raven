from .services.llm_service import LLMService


class ClaimExtractionAgent:

    def __init__(self):
        self.llm = LLMService()

    def extract_claim(self, text):

        if not text or not text.strip():
            return ""

        text = text.strip()

        prompt = f"""
Extract the main factual claim from the following text.

TEXT:
{text}

Return only the concise main claim.
"""

        claim = self.llm.generate(prompt)

        return self.clean_claim(claim)

    def clean_claim(self, claim):

        if not claim:
            return ""

        claim = claim.strip()

        prefixes = [
            "Claim:",
            "CLAIM:",
            "Main Claim:",
            "Main claim:",
            "Extracted Claim:",
            "Extracted claim:",
            "Output:",
            "OUTPUT:"
        ]

        for prefix in prefixes:
            if claim.startswith(prefix):
                claim = claim[len(prefix):].strip()

        # Remove quotation marks
        claim = claim.strip('"').strip("'")

        # Remove accidental numbering
        if claim.startswith("1."):
            claim = claim[2:].strip()

        # Remove extra spaces/newlines
        claim = " ".join(claim.split())

        return claim

    def extract_claims(self, text):

        if not text or not text.strip():
            return []

        claim = self.extract_claim(text)

        if not claim:
            return []

        return [claim]