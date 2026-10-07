# src/personas/scientific_persona.py
"""Scientific persona LLM implementation.
Analyzes claim from a scientific/factual perspective focusing on methodology and evidence strength.
"""

from __future__ import annotations

from .base_persona import BasePersona


class ScientificPersona(BasePersona):
    """LLM-based Scientific analysis.
    """

    name = "scientific"

    def _get_system_prompt(self) -> str:
        return (
            "You are an expert Scientist evaluating a claim against provided evidence. "
            "Your task is to analyze the evidence and determine if it supports, contradicts, "
            "is insufficient, or is conflicting regarding the claim.\n\n"
            "Focus specifically on:\n"
            "- Scientific and factual consistency.\n"
            "- The strength of the evidence presented.\n"
            "- Whether the evidence provides direct support or contradiction.\n"
            "- Methodology and context when available.\n"
            "- Limitations and uncertainty of the evidence.\n\n"
            "Provide a rigorous scientific assessment."
        )
