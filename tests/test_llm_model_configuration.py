from inspect import signature

from agents.claim_extraction.services.llm_service import LLMService
from agents.domain_detection.domain_agent import DomainDetectionAgent
from agents.reflection.reflection_agent import ReflectionAgent
from agents.supervisior.llm_client import call_supervisor_llm
from agents.xai.xai_agent import XAIAgent
from rag_ingestion.personas.base_persona import BasePersona
from rag_ingestion.verification.ollama_verifier import DEFAULT_MODEL
from rag_ingestion.verification.verification_manager import (
    EvidenceVerificationManager,
)


def test_all_active_agent_model_defaults_are_qwen_25_3b():
    expected_model = "qwen2.5:3b"

    assert LLMService().model == expected_model
    assert DomainDetectionAgent().model_name == expected_model
    assert ReflectionAgent().model == expected_model
    assert signature(call_supervisor_llm).parameters["model"].default == expected_model
    assert XAIAgent().model == expected_model
    assert DEFAULT_MODEL == expected_model
    assert EvidenceVerificationManager().model == expected_model
    assert BasePersona._model_name == expected_model
