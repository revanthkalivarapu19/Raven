# src/personas/base_persona.py
"""Abstract base class for all persona agents.
Provides an interface for analysis over a SupervisorContext using local LLMs.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any

import requests
from pydantic import ValidationError

from rag_ingestion.supervisor.supervisor_schema import SupervisorContext
from .persona_schema import PersonaInsight


class BasePersona(ABC):
    """Base class for persona agents.

    Subclasses must set the `name` attribute and implement the `_get_system_prompt`
    method, which returns the role-specific prompt instructions.
    """

    name: str
    _ollama_url: str = "http://localhost:11434/api/generate"
    _model_name: str = "qwen2.5:3b"
    _timeout: int = 300

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Return the role-specific system prompt."""
        raise NotImplementedError

    def analyze(self, context: SupervisorContext) -> PersonaInsight:
        """Analyze the provided context and produce a :class:`PersonaInsight`.

        Calls the local Ollama LLM and structures the response.
        """
        prompt = self._build_prompt(context)

        try:
            response_json = self._call_llm(prompt)
        except ValueError as e:
            raise e
        except Exception as e:
            raise RuntimeError(f"Failed to generate insight from LLM: {str(e)}")

        try:
            # Parse the LLM output into PersonaInsight schema
            # We inject the persona name to ensure it matches the class name
            response_json["persona"] = self.name

            # The schema requires evidence_ids as a list of strings
            evidence_ids = response_json.get("evidence_ids", [])

            # Ensure reasoning is not empty
            if not response_json.get("reasoning") or str(response_json.get("reasoning")).strip() == "":
                raise ValueError("reasoning must not be empty")

            # Ensure evidence IDs only refer to context evidence
            context_eids = {e.evidence_id for e in context.evidence}
            for eid in evidence_ids:
                if eid not in context_eids:
                    raise ValueError(f"evidence_id {eid} not present in SupervisorContext")

            insight = PersonaInsight(**response_json)
            return insight

        except ValidationError as e:
            raise ValueError(f"LLM returned invalid output structure: {e}")
        except Exception as e:
            raise ValueError(f"Failed to process LLM response: {e}")

    def _build_prompt(self, context: SupervisorContext) -> str:
        """Construct the prompt sent to the LLM."""
        system_prompt = self._get_system_prompt()

        evidence_str = ""
        for ev in context.evidence:
            vr = context.mapping.get(ev.evidence_id)
            vr_info = f" (Verification: {vr.relationship}, confidence: {vr.confidence})" if vr else ""
            evidence_str += f"- ID: {ev.evidence_id}\n  Text: {ev.text}{vr_info}\n"

        if not evidence_str:
            evidence_str = "No evidence provided."

        prompt = f"""{system_prompt}

You must reason ONLY from the supplied context and evidence below. Do not invent external sources, facts, citations, or evidence IDs.

Claim: {context.claim}
Domain: {context.domain}

Evidence:
{evidence_str}

Return a valid JSON object matching this structure exactly:
{{
  "assessment": "SUPPORTED" | "CONTRADICTED" | "INSUFFICIENT" | "CONFLICTING",
  "confidence": 0.0 to 1.0 (float),
  "reasoning": "Detailed explanation based strictly on the evidence",
  "evidence_ids": ["id1", "id2"] // Only list IDs of evidence you used
}}
"""
        return prompt

    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """Call the local Ollama API."""
        payload = {
            "model": self._model_name,
            "prompt": prompt,
            "format": "json",
            "stream": False
        }

        try:
            response = requests.post(self._ollama_url, json=payload, timeout=self._timeout)
            response.raise_for_status()
            result = response.json()
            llm_text = result.get("response", "")
            return json.loads(llm_text)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API request failed: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Ollama returned malformed JSON: {e}")
