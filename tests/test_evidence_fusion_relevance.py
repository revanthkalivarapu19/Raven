from datetime import datetime

from rag_ingestion.retrieval.evidence_fusion import EvidenceFusionManager
from rag_ingestion.retrieval.evidence_schema import Evidence
from rag_ingestion.verification.verification_manager import EvidenceVerificationManager


def _ev(eid, text, year=None, similarity=0.9, authority=0.8):
    return Evidence(
        evidence_id=eid, chunk_id=eid, document_id=eid, text=text,
        title=text, domain="general", source="test", language="en",
        metadata={}, similarity_score=similarity, authority_score=authority,
        publication_date=datetime(year, 1, 1) if year else None,
    )


def test_exact_historical_event_is_retained_over_other_year_evidence():
    claim = "Entity changed instrument by 0.50% in 2020."
    historical = _ev("historical", "Entity changed instrument by 0.50% in 2020.", 2020)
    current = _ev("current", "Entity changed instrument in 2026.", 2026)
    result = EvidenceFusionManager().fuse([historical, current], [], 2, claim=claim)
    assert [item.evidence_id for item in result] == ["historical"]


def test_matching_numeric_value_improves_relevance():
    claim = "The rate changed by 0.50%."
    matching = _ev("matching", "The rate changed by 0.50%.", 2020, similarity=0.7)
    different = _ev("different", "The rate changed by 0.25%.", 2020, similarity=0.8)
    result = EvidenceFusionManager().fuse([matching, different], [], 2, claim=claim)
    assert result[0].evidence_id == "matching"


def test_authority_alone_does_not_beat_direct_relevance():
    claim = "Entity changed instrument in 2020."
    relevant = _ev("relevant", "Entity changed instrument in 2020.", 2020, authority=0.4)
    unrelated = _ev("unrelated", "General unrelated discussion.", 2020, authority=1.0)
    result = EvidenceFusionManager().fuse([relevant, unrelated], [], 2, claim=claim)
    assert result[0].evidence_id == "relevant"


def test_ids_are_preserved_and_unrelated_evidence_is_filtered():
    claim = "A specific event occurred."
    first = _ev("first", "A specific event occurred.", similarity=0.9)
    second = _ev("second", "Unrelated text.", similarity=0.1)
    result = EvidenceFusionManager().fuse([first, second], [], 2, claim=claim)
    assert [item.evidence_id for item in result] == ["first"]
    assert result[0].evidence_id == "first"


def test_relevant_political_news_is_retained_over_academic_evidence():
    claim = "The minister defended the election commission against the opposition party."
    local = _ev(
        "local", "Minister defended election commission against opposition party.",
        similarity=0.99,
    )
    academic = _ev(
        "academic", "Research on the minister, election commission and opposition party.",
        similarity=0.95,
    ).model_copy(update={
        "retrieval_method": "external",
        "metadata": {"external_source_type": "crossref"},
        "url": "https://doi.example.test/research",
    })
    news_one = _ev(
        "news-one", "Minister defended the election commission.",
        similarity=None,
    ).model_copy(update={
        "retrieval_method": "external",
        "metadata": {
            "external_source_type": "news_api",
            "publisher": "Times Example",
        },
        "url": "https://news.example.test/article-one",
    })
    news_two = _ev(
        "news-two", "Opposition party questioned the minister's remarks.",
        similarity=None,
    ).model_copy(update={
        "retrieval_method": "external",
        "metadata": {
            "external_source_type": "gdelt",
            "publisher": "Business Example",
        },
        "url": "https://news.example.test/article-two",
    })

    result = EvidenceFusionManager().fuse(
        [local], [academic, news_one, news_two], 3,
        claim=claim, domain="Politics",
    )

    selected_ids = {item.evidence_id for item in result}
    assert {"news-one", "news-two"} <= selected_ids
    assert len(result) == 3
    assert next(item for item in result if item.evidence_id == "news-one").url == news_one.url
    assert next(item for item in result if item.evidence_id == "news-one").metadata["publisher"] == "Times Example"


def test_smoking_claim_filters_unrelated_medical_evidence_and_keeps_relevant_evidence():
    claim = "Smoking increases the risk of lung cancer."
    relevant = _ev(
        "smoking-lung-cancer",
        "Smoking is associated with increased lung cancer risk.",
        similarity=0.72,
    )
    covid = _ev(
        "covid-guidance",
        "COVID-19 guidance discusses health risk and lung cancer screening.",
        similarity=0.53,
    )
    antibiotic = _ev(
        "antibiotic-resistance",
        "Antibiotic resistance guidance reduces infection risk.",
        similarity=0.51,
    )

    result = EvidenceFusionManager().fuse(
        [relevant, covid, antibiotic], [], 5, claim=claim
    )

    assert [item.evidence_id for item in result] == ["smoking-lung-cancer"]


def test_relevant_evidence_without_similarity_metadata_is_kept():
    claim = "Smoking increases the risk of lung cancer."
    relevant = _ev(
        "relevant-missing-similarity",
        "Smoking is associated with increased lung cancer risk.",
        similarity=None,
    ).model_copy(update={"metadata": {}})

    result = EvidenceFusionManager().fuse([], [relevant], 5, claim=claim)

    assert result == [relevant]


def test_low_relevance_evidence_becomes_insufficient_not_supported():
    claim = "Smoking increases the risk of lung cancer."
    unrelated = _ev(
        "one-overlap",
        "General health risk guidance for COVID-19 vaccination.",
        similarity=0.99,
    )
    fused = EvidenceFusionManager().fuse([unrelated], [], 5, claim=claim)
    verification = EvidenceVerificationManager().verify(claim, fused)

    assert fused == []
    assert verification.overall_assessment == "INSUFFICIENT_EVIDENCE"
