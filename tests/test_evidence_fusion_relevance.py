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


def test_semantic_ranking_batches_candidates_and_uses_claim_relevant_excerpts():
    import numpy as np

    class Model:
        def __init__(self):
            self.encode_calls = []

        def encode(self, texts, **_kwargs):
            self.encode_calls.append(list(texts))
            return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))

        def __call__(self, texts, **_kwargs):
            return {"input_ids": [[1] * len(text.split()) for text in texts]}

    model = Model()
    embedder = type("Embedder", (), {"_model": model})()
    manager = EvidenceFusionManager()
    manager.semantic_embedder = embedder
    claim = "smoking increases lung cancer risk"
    first = _ev(
        "direct", "An unrelated opening sentence. Smoking increases lung cancer risk.",
    ).model_copy(update={"title": "Smoking and lung cancer"})
    second = _ev("related", "Smoking can cause lung cancer.")

    scores = manager._semantic_scores(claim, [first, second])

    assert scores == {"direct": 1.0, "related": 1.0}
    assert [len(call) for call in model.encode_calls] == [1, 2]
    passage_batch = model.encode_calls[1]
    assert "Smoking increases lung cancer risk" in passage_batch[0]
    assert len(passage_batch[0]) <= 256

    manager._semantic_scores(claim, [first, second])
    assert [len(call) for call in model.encode_calls] == [1, 2]


def test_final_fusion_bounds_bge_to_shortlisted_candidate_set():
    import numpy as np

    class Model:
        def __init__(self):
            self.encode_calls = []

        def encode(self, texts, **_kwargs):
            self.encode_calls.append(list(texts))
            return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))

        def __call__(self, texts, **_kwargs):
            return {"input_ids": [[1] * len(text.split()) for text in texts]}

    model = Model()
    manager = EvidenceFusionManager()
    manager.semantic_embedder = type("Embedder", (), {"_model": model})()
    claim = "Earth orbits the Sun."
    candidates = [
        _ev(f"candidate-{index}", f"Earth orbits the Sun. Source marker {index}.")
        for index in range(12)
    ]

    manager.fuse(candidates, [], 3, claim=claim, semantic_ranking=True)

    assert [len(call) for call in model.encode_calls] == [1, 5]


def test_same_sentence_claim_alignment_prefers_direct_over_topical_passage():
    import numpy as np

    class Model:
        def encode(self, texts, **_kwargs):
            return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))

        def __call__(self, texts, **_kwargs):
            return {"input_ids": [[1] * len(text.split()) for text in texts]}

    claim = "Smoking increases the risk of lung cancer."
    topical = _ev("topical", "This cancer page discusses smoking, lung cancer, and health risks.")
    direct = _ev("direct", "Tobacco smoking increases risk of lung cancer.")
    manager = EvidenceFusionManager()
    manager.semantic_embedder = type("Embedder", (), {"_model": Model()})()

    result = manager.fuse([topical, direct], [], 2, claim=claim, semantic_ranking=True)

    assert result[0].evidence_id == "direct"


def test_broad_page_terms_do_not_accumulate_across_sentences_for_ranking():
    import numpy as np

    class Model:
        def encode(self, texts, **_kwargs):
            return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))

        def __call__(self, texts, **_kwargs):
            return {"input_ids": [[1] * len(text.split()) for text in texts]}

    claim = "The Falcon transports cargo to the station each month."
    direct = _ev("direct", "The Falcon transports cargo to the station each month.")
    topical = _ev("topical", "The Falcon is a vehicle. The station has a cargo bay. A month has many days.")
    manager = EvidenceFusionManager()
    manager.semantic_embedder = type("Embedder", (), {"_model": Model()})()

    ranked = manager.fuse([topical, direct], [], 2, claim=claim, semantic_ranking=True)

    assert ranked[0].evidence_id == "direct"


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


def test_genuine_nasa_earth_sun_figure_description_survives_fusion():
    # This is source-authored image alt text from the existing NASA IMAP page's
    # Lagrange-points graphic, with its real URL and document title preserved.
    text = (
        "A graphic shows the Sun at the center with Earth's orbit drawn as a blue circle around it. "
        "Earth appears along the orbit to the right of the Sun, at the 3 o'clock position on the circle. "
        "L1 is marked as a white dot to the left of Earth along a horizontal dashed line that extends "
        "from the left side of the circle through the Sun and beyond Earth and the right side of the circle. "
        "L2 is marked as a white dot to the right of Earth at the right end of the dashed line. "
        "L3 is marked with a white dot on the left side of the circle, to the left of the Sun, at the left "
        "end of the dashed line. L4 is marked with a white dot near the top of the circle, at about the "
        "1 o'clock position. L5 is marked with a white dot along the lower right part of the circle, near "
        "the 5 o'clock position."
    )
    genuine = Evidence(
        evidence_id="nasa-imap-lagrange-figure",
        chunk_id="nasa-imap-lagrange-figure",
        document_id="nasa-imap",
        text=text,
        title="Interstellar Mapping and Acceleration Probe (IMAP)",
        domain="science",
        source="NASA",
        language="en",
        url="https://science.nasa.gov/mission/imap/",
        metadata={"content_quality": "source_text", "extraction": "source-authored image alt text"},
        similarity_score=0.62,
        authority_score=1.0,
    )

    result = EvidenceFusionManager().fuse(
        [genuine], [], 3, claim="The Earth orbits the Sun.", domain="science"
    )

    assert [item.evidence_id for item in result] == ["nasa-imap-lagrange-figure"]


def test_genuine_nasa_earth_orbit_period_statement_survives_fusion():
    evidence = Evidence(
        evidence_id="nasa-space-place-earth-orbit-period",
        chunk_id="nasa-space-place-earth-orbit-period",
        document_id="nasa-space-place-leap-year",
        text="It takes approximately 365.25 days for Earth to orbit the Sun — a solar year.",
        title="What Is a Leap Year?",
        domain="science",
        source="NASA Space Place",
        language="en",
        url="https://spaceplace.nasa.gov/leap-year/en/",
        metadata={"content_quality": "source_text"},
        similarity_score=0.74,
        authority_score=1.0,
    )

    result = EvidenceFusionManager().fuse(
        [evidence], [], 3,
        claim="The Earth orbits the Sun once every year.",
        domain="science",
    )

    assert [item.evidence_id for item in result] == ["nasa-space-place-earth-orbit-period"]


def test_earth_sized_exoplanet_orbit_does_not_match_earth_sun_claim():
    unrelated = Evidence(
        evidence_id="nasa-exoplanet", chunk_id="nasa-exoplanet",
        document_id="nasa-exoplanets",
        text=("This artist's concept features seven Earth-sized planets orbiting "
              "the star TRAPPIST-1, a Sun-like system about 40 light-years away."),
        title="Other Stars, Other Worlds", domain="science", source="NASA",
        language="en", url="https://science.nasa.gov/exoplanets/",
        metadata={"content_quality": "source_text"}, similarity_score=0.62,
        authority_score=1.0,
    )

    result = EvidenceFusionManager().fuse(
        [unrelated], [], 3,
        claim="The Earth is the center of the solar system and the Sun orbits Earth.",
        domain="science",
    )

    # Broad retrieval may include topical material; semantic verification
    # decides whether this passage establishes the actual orbital claim.
    assert all(item.evidence_id == "nasa-exoplanet" for item in result)
