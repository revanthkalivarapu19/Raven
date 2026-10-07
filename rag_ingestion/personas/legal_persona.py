# src/personas/legal_persona.py
"""Legal persona LLM implementation.
Analyzes claim from a legal/regulatory perspective focusing on legal assertions and uncertainty.
"""

from __future__ import annotations

from .base_persona import BasePersona


class LegalPersona(BasePersona):
    """LLM-based Legal analysis.
    """

    name = "legal"

    def _get_system_prompt(self) -> str:
        return (
            "You are an expert Legal Analyst evaluating a claim against provided evidence. "
            "Your task is to analyze the evidence and determine if it supports, contradicts, "
            "is insufficient, or is conflicting regarding the claim.\n\n"
            "Focus specifically on:\n"
            "- Legal and regulatory claims.\n"
            "- Whether the evidence supports the legal assertions made.\n"
            "- Qualifications and contextual nuances of the evidence.\n"
            "- Uncertainty where legal conclusions cannot be firmly established.\n\n"
            "Provide a precise legal assessment."
        )
