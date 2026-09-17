"""
Mock data for the Supervisor Agent.
Provides realistic sample supervisor_input dicts for testing.
"""

from datetime import datetime
from .evidence_schema import Evidence

def get_mock_input(scenario: str = "supporting") -> dict:
    """
    Return a mock supervisor_input dict for one of three scenarios:
        "supporting"    - evidence mostly supports the claim
        "contradicting" - evidence mostly contradicts the claim
        "no_reflection" - same as "supporting" but reflection_notes is None,
                           to test the pipeline works without it
    
    Raises:
        ValueError: For any other scenario string.
    """
    if scenario == "supporting":
        return {
            "claim_text": "The Reserve Bank of India has cut interest rates by 0.50%.",
            "domain": "Finance",
            "evidence_list": [
                Evidence(
                    chunk_id="chunk_001",
                    document_id="doc_001",
                    text="RBI announced a 0.50% repo rate cut on...",
                    domain="Finance",
                    source="Reuters",
                    url="https://reuters.com/some-article",
                    title="RBI Cuts Repo Rate",
                    language="en",
                    similarity_score=0.91,
                    authority_score=0.9,
                    retrieval_method="faiss_local",
                    metadata={},
                    publication_date=datetime(2026, 8, 10)
                ),
                Evidence(
                    chunk_id="chunk_002",
                    document_id="doc_002",
                    text="RBI is expected to raise rates, not cut them...",
                    domain="Finance",
                    source="Random blog",
                    url="https://randomblog.com/rbi-news",
                    title="RBI Rate Speculation",
                    language="en",
                    similarity_score=0.62,
                    authority_score=0.2,
                    retrieval_method="faiss_local",
                    metadata={},
                    publication_date=datetime(2026, 8, 9)
                ),
            ],
            "reflection_notes": {
                "is_evidence_sufficient": True,
                "contradictions_found": True,
                "confidence_adjustment": -0.1,
                "notes": "One low-credibility source contradicts the claim; "
                         "high-credibility sources support it. Recommend proceeding."
            },
            "original_language": "en",
            "was_translated": False
        }
    elif scenario == "contradicting":
        return {
            "claim_text": "The Moon is made entirely of green cheese.",
            "domain": "Science",
            "evidence_list": [
                Evidence(
                    chunk_id="chunk_003",
                    document_id="doc_003",
                    text="Geological analysis of lunar samples shows the Moon is composed of rock and dust, not dairy products.",
                    domain="Science",
                    source="NASA Science",
                    url="https://nasa.gov/moon-rocks",
                    title="Lunar Composition",
                    language="en",
                    similarity_score=0.85,
                    authority_score=0.95,
                    retrieval_method="faiss_local",
                    metadata={},
                    publication_date=datetime(2024, 1, 15)
                ),
                Evidence(
                    chunk_id="chunk_004",
                    document_id="doc_004",
                    text="I saw the moon and it definitely looks like green cheese. Trust me.",
                    domain="Science",
                    source="Conspiracy Forum",
                    url="https://forum.conspiracy.fake/moon-cheese",
                    title="The Truth About The Moon",
                    language="en",
                    similarity_score=0.92,
                    authority_score=0.1,
                    retrieval_method="faiss_local",
                    metadata={},
                    publication_date=datetime(2025, 5, 20)
                ),
            ],
            "reflection_notes": {
                "is_evidence_sufficient": True,
                "contradictions_found": True,
                "confidence_adjustment": 0.0,
                "notes": "High-credibility sources contradict the claim; low-credibility sources support it."
            },
            "original_language": "en",
            "was_translated": False
        }
    elif scenario == "no_reflection":
        # Reuse supporting but with None reflection_notes
        data = get_mock_input("supporting")
        data["reflection_notes"] = None
        return data
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
