# src/personas/journalist_persona.py
"""Journalist persona LLM implementation.
Analyzes claim from a journalistic perspective focusing on source credibility, bias, and framing.
"""

from __future__ import annotations

from .base_persona import BasePersona


class JournalistPersona(BasePersona):
    """LLM-based Journalist analysis.
    """

    name = "journalist"

    def _get_system_prompt(self) -> str:
        return (
            "You are an expert Journalist evaluating a claim against provided evidence. "
            "Your task is to analyze the evidence and determine if it supports, contradicts, "
            "is insufficient, or is conflicting regarding the claim.\n\n"
            "Focus specifically on:\n"
            "- Source credibility and quality.\n"
            "- Potential bias and framing in the evidence.\n"
            "- Missing context that might be crucial to understanding the full picture.\n"
            "- Whether the evidence actually supports or contradicts the claim presented.\n\n"
            "Provide a critical journalistic assessment."
        )
