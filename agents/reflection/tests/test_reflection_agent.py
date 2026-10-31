import httpx
import ollama
import pytest
from agents.reflection.reflection_agent import ReflectionAgent
from rag_ingestion.verification.verification_schema import OverallVerification, VerificationResult


pytestmark = pytest.mark.integration
OLLAMA_MODEL = "qwen2.5:3b"


@pytest.fixture(scope="module", autouse=True)
def require_ollama_integration():
    try:
        models = ollama.Client(timeout=5).list().models
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        pytest.skip(f"Ollama integration is unavailable: {exc}")

    available_models = {model.model for model in models}
    if OLLAMA_MODEL not in available_models:
        pytest.skip(f"Ollama integration requires the local {OLLAMA_MODEL} model")


def run_test(test_name, claim, domain, evidence, verification_result=None):
    print("\n" + "=" * 70)
    print(test_name)
    print("=" * 70)

    agent = ReflectionAgent(model=OLLAMA_MODEL)

    input_data = {
        "claim_text": claim,
        "domain": domain,
        "evidence_list": evidence
    }
    if verification_result is not None:
        input_data["verification_result"] = verification_result

    result = agent.reflect(input_data)

    print("\nClaim:", claim)
    print("Domain:", domain)
    print("\nDecision:", result["decision"])
    print("Evidence Sufficient:", result["is_evidence_sufficient"])
    print("Contradictions Found:", result["contradictions_found"])
    print("Claim Supported:", result["claim_supported"])
    print("Claim Contradicted:", result["claim_contradicted"])
    print("Evidence Items Contradict:", result["evidence_items_contradict_each_other"])
    print("Confidence:", result["confidence"])
    print("Source Credibility Score:", result["source_credibility_score"])
    print("Evidence Quality Score:", result["evidence_quality_score"])
    print("Notes:", result["notes"])
    return result


def test_strong_evidence():
    evidence_1 = [
        {
            "chunk_id": "chunk_001",
            "document_id": "doc_001",
            "text": (
                "The Reserve Bank of India announced a 0.50 percent reduction "
                "in the repo rate to support economic growth."
            ),
            "domain": "Finance",
            "source": "Reserve Bank of India",
            "title": "RBI Monetary Policy Announcement",
            "similarity_score": 0.95,
            "authority_score": 0.99,
            "retrieval_method": "faiss_local",
            "url": "https://example.com/rbi",
            "publication_date": "2026-08-10"
        }
    ]
    res = run_test(
        "TEST 1 - Strong Evidence",
        "The Reserve Bank of India has cut interest rates by 0.50%.",
        "Finance",
        evidence_1
    )
    assert res["decision"] == "YES"
    assert res["is_evidence_sufficient"] is True
    assert res["claim_supported"] is True
    assert res["confidence"] >= 0.70


def test_cut_and_reduction_are_semantically_equivalent():
    evidence = [
        {
            "chunk_id": "chunk_rate_reduction",
            "document_id": "doc_rate_reduction",
            "text": (
                "The monetary authority announced a 0.50 percent reduction "
                "in the benchmark repo rate to support economic growth."
            ),
            "domain": "Finance",
            "source": "Central Bank",
            "title": "Monetary Policy Announcement",
            "similarity_score": 0.95,
            "authority_score": 0.99,
            "retrieval_method": "faiss_local",
            "url": "https://example.com/monetary-policy",
            "publication_date": "2026-08-10"
        }
    ]
    res = run_test(
        "TEST - Cut and Reduction Semantic Equivalence",
        "The central bank cut interest rates by 0.50%.",
        "Finance",
        evidence
    )
    assert res["decision"] == "YES"
    assert res["is_evidence_sufficient"] is True
    assert res["claim_supported"] is True
    assert res["claim_contradicted"] is False
    assert res["evidence_items_contradict_each_other"] is False


def test_low_authority_blog():
    evidence_blog = [
        {
            "chunk_id": "chunk_002",
            "document_id": "doc_002",
            "text": (
                "The Reserve Bank of India uses the policy repo rate as an important "
                "instrument of monetary policy."
            ),
            "domain": "Finance",
            "source": "Unknown Finance Blog",
            "title": "Finance Blog",
            "similarity_score": 0.95,
            "authority_score": 0.10,
            "retrieval_method": "faiss_local",
            "url": "https://random-finance-news-123.example/rbi-policy",
            "publication_date": "2026-09-01"
        }
    ]
    res = run_test(
        "TEST 2 - High Similarity But Very Low Authority",
        "The Reserve Bank of India uses the policy repo rate as an important instrument of monetary policy.",
        "Finance",
        evidence_blog
    )
    assert res["decision"] == "NO"
    assert res["is_evidence_sufficient"] is False
    assert res["source_credibility_score"] <= 0.35
    assert res["confidence"] <= 0.50


def test_no_evidence():
    res = run_test(
        "TEST 3 - No Evidence",
        "NASA launched a new spacecraft to study Mars.",
        "Science",
        []
    )
    assert res["decision"] == "NO"
    assert res["is_evidence_sufficient"] is False
    assert res["confidence"] == 0.0


def test_contradictory_evidence_items():
    evidence_contradicting = [
        {
            "chunk_id": "chunk_006",
            "document_id": "doc_006",
            "text": "The Reserve Bank of India announced a 0.50 percent reduction in the repo rate.",
            "domain": "Finance",
            "source": "Reserve Bank of India",
            "title": "RBI Rate Decision",
            "similarity_score": 0.94,
            "authority_score": 0.99,
            "retrieval_method": "faiss_local",
            "url": "https://example.com/rbi",
            "publication_date": "2026-08-10"
        },
        {
            "chunk_id": "chunk_007",
            "document_id": "doc_007",
            "text": "The Reserve Bank of India did not reduce the repo rate and instead kept rates unchanged.",
            "domain": "Finance",
            "source": "Financial Times",
            "title": "Rate Speculation",
            "similarity_score": 0.88,
            "authority_score": 0.85,
            "retrieval_method": "faiss_local",
            "url": "https://example.com/ft",
            "publication_date": "2026-08-11"
        }
    ]
    claim = "The Reserve Bank of India has cut interest rates by 0.50%."
    verification = OverallVerification(
        claim=claim,
        results=[
            VerificationResult(
                evidence_id=chunk_id, claim=claim, relationship=relationship,
                confidence=0.9, reasoning="Fixture relationship", verifier="test",
                same_event=True, claim_aligned=True,
            )
            for chunk_id, relationship in (("chunk_006", "SUPPORTS"), ("chunk_007", "CONTRADICTS"))
        ],
        overall_assessment="CONFLICTING_EVIDENCE",
        overall_confidence=0.9,
    )
    res = run_test(
        "TEST 4 - Contradictory Evidence Items",
        claim,
        "Finance",
        evidence_contradicting,
        verification,
    )
    assert res["evidence_items_contradict_each_other"] is True
    assert res["contradictions_found"] is True


def test_claim_contradicted_single_evidence():
    evidence_single_contra = [
        {
            "chunk_id": "chunk_contra",
            "document_id": "doc_contra",
            "text": "The Reserve Bank of India does not use the policy repo rate.",
            "domain": "Finance",
            "source": "Reserve Bank of India",
            "title": "RBI Monetary Policy Clarification",
            "similarity_score": 0.93,
            "authority_score": 0.98,
            "retrieval_method": "faiss_local",
            "url": "https://example.com/rbi",
            "publication_date": "2026-08-10"
        }
    ]
    res = run_test(
        "TEST 5 - Claim vs Evidence Contradiction (Single Evidence)",
        "The Reserve Bank of India uses the policy repo rate.",
        "Finance",
        evidence_single_contra
    )
    assert res["claim_supported"] is False
    assert res["claim_contradicted"] is True
    assert res["evidence_items_contradict_each_other"] is False


if __name__ == "__main__":
    test_strong_evidence()
    test_low_authority_blog()
    test_no_evidence()
    test_contradictory_evidence_items()
    test_claim_contradicted_single_evidence()
