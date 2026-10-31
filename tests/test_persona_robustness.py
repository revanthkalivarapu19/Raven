from types import SimpleNamespace

from rag_ingestion.personas.base_persona import BasePersona
from rag_ingestion.retrieval.evidence_schema import Evidence


class PersonaProbe(BasePersona):
    name = "probe"

    def _get_system_prompt(self):
        return "Analyze the supplied evidence."


def _context():
    evidence = Evidence(
        evidence_id="evidence-real",
        chunk_id="chunk",
        document_id="document",
        text="Evidence text",
        domain="general",
        source="test",
        language="en",
        metadata={},
    )
    verification = SimpleNamespace(relationship="SUPPORTS", confidence=0.9)
    return SimpleNamespace(
        claim="A claim", domain="general", evidence=[evidence],
        mapping={"evidence-real": verification},
    )


def test_persona_prompt_lists_exact_available_evidence_ids():
    prompt = PersonaProbe()._build_prompt(_context())

    assert "Available evidence IDs: evidence-real" in prompt
    assert "id1" in prompt
    assert "Select one or more evidence_ids ONLY" in prompt


def test_unknown_persona_evidence_id_fails_safely():
    persona = PersonaProbe()
    persona._call_llm = lambda prompt: {
        "assessment": "SUPPORTED",
        "confidence": 0.9,
        "reasoning": "Test reasoning",
        "evidence_ids": ["id1"],
    }

    try:
        persona.analyze(_context())
    except ValueError as error:
        assert "not present" in str(error)
    else:
        raise AssertionError("unknown evidence ID was accepted")


def _response(evidence_ids):
    return {
        "assessment": "SUPPORTED",
        "confidence": 0.9,
        "reasoning": "Test reasoning",
        "evidence_ids": evidence_ids,
    }


def test_valid_response_with_exact_evidence_id():
    persona = PersonaProbe()
    persona._call_llm = lambda prompt: _response(["evidence-real"])

    insight = persona.analyze(_context())

    assert insight.evidence_ids == ["evidence-real"]


def test_missing_ids_first_response_is_repaired_once():
    persona = PersonaProbe()
    responses = iter([_response(None), _response(["evidence-real"])])
    prompts = []
    persona._call_llm = lambda prompt: (prompts.append(prompt) or next(responses))

    insight = persona.analyze(_context())

    assert insight.evidence_ids == ["evidence-real"]
    assert len(prompts) == 2
    assert "evidence-real" in prompts[1]
    assert "mandatory" in prompts[1]


def test_missing_ids_on_both_attempts_remains_controlled_failure():
    persona = PersonaProbe()
    calls = []
    persona._call_llm = lambda prompt: (calls.append(prompt) or _response(None))

    try:
        persona.analyze(_context())
    except ValueError as error:
        assert "after one retry" in str(error)
    else:
        raise AssertionError("malformed persona response was accepted")
    assert len(calls) == 2


def test_unknown_id_is_rejected_after_retry():
    persona = PersonaProbe()
    responses = iter([_response(["unknown"]), _response(["unknown"])])
    persona._call_llm = lambda prompt: next(responses)

    try:
        persona.analyze(_context())
    except ValueError as error:
        assert "after one retry" in str(error)
    else:
        raise AssertionError("unknown persona evidence ID was accepted")


def test_placeholder_id_is_rejected_after_retry():
    persona = PersonaProbe()
    responses = iter([_response(["id1"]), _response(["id1"])])
    persona._call_llm = lambda prompt: next(responses)

    try:
        persona.analyze(_context())
    except ValueError:
        pass
    else:
        raise AssertionError("placeholder persona evidence ID was accepted")


def test_empty_ids_are_rejected_after_retry():
    persona = PersonaProbe()
    responses = iter([_response([]), _response([])])
    persona._call_llm = lambda prompt: next(responses)

    try:
        persona.analyze(_context())
    except ValueError:
        pass
    else:
        raise AssertionError("empty persona evidence IDs were accepted")
