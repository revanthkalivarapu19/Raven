# src/personas/__init__.py
"""Persona package initialization.
Exports the persona classes, schema, and selection utility.
"""

from .base_persona import BasePersona
from .journalist_persona import JournalistPersona
from .scientific_persona import ScientificPersona
from .legal_persona import LegalPersona
from .persona_schema import PersonaInsight

__all__ = [
    "BasePersona",
    "JournalistPersona",
    "ScientificPersona",
    "LegalPersona",
    "PersonaInsight",
    "select_personas",
]

def select_personas(domain: str):
    """Return a list of persona classes appropriate for the given domain.

    Mapping (case‑insensitive):
        medical    → ScientificPersona, JournalistPersona
        science    → ScientificPersona, JournalistPersona
        technology → ScientificPersona, JournalistPersona
        politics   → JournalistPersona, LegalPersona
        finance    → JournalistPersona, LegalPersona
        general    → JournalistPersona
    Unknown domains default to an empty list.
    """
    mapping = {
        "medical": [ScientificPersona, JournalistPersona],
        "science": [ScientificPersona, JournalistPersona],
        "technology": [ScientificPersona, JournalistPersona],
        "politics": [JournalistPersona, LegalPersona],
        "finance": [JournalistPersona, LegalPersona],
        "general": [JournalistPersona],
    }
    key = domain.strip().lower()
    return mapping.get(key, [])
