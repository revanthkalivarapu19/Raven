import logging
import urllib.parse
from datetime import datetime
from typing import List, Set, Dict, Any

from rag_ingestion.retrieval.evidence_schema import Evidence

logger = logging.getLogger(__name__)


def _canonicalize_url(url: str) -> str:
    """Return a deterministic, canonical version of a URL.

    - Normalises scheme and netloc to lower case.
    - Strips trailing slashes.
    - Removes fragment identifiers.
    - Removes common UTM tracking query parameters.
    - Preserves path and other query parameters.
    """
    parsed = urllib.parse.urlparse(url)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip('/')
    # Preserve query params except known tracking ones (utm_*)
    query_params = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    filtered = [(k, v) for k, v in query_params if not k.lower().startswith('utm_')]
    query = urllib.parse.urlencode(filtered, doseq=True)
    # Rebuild without fragment
    return urllib.parse.urlunparse((scheme, netloc, path, '', query, ''))


class EvidenceFusionManager:
    """Combine local and external :class:`~rag_ingestion.retrieval.evidence_schema.Evidence` objects.

    The manager performs three deterministic steps:

    1. **Merge** local (FAISS) and external evidence preserving the original order
       (local first).
    2. **Deduplicate** based on ``evidence_id``, canonical ``url`` and exact ``text``.
       When a duplicate URL is found we keep the first occurrence and enrich its
       ``metadata`` with a ``duplicate_sources`` entry describing the removed
       evidence (source, retrieval_method and evidence_id).  No input objects are
       mutated – a shallow copy is created when metadata needs to be merged.
    3. **Rank** the remaining items using a deterministic *retrieval‑quality* score.
       The score is a weighted sum of similarity, authority and recency (see the
       docstring of :meth:`_ranking_score`).  After scoring, items are sorted with
       explicit tie‑breakers to guarantee repeatable ordering.
    """

    def __init__(self) -> None:
        # No external configuration – all constants are deterministic.
        self._utm_params = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content"}

    # ---------------------------------------------------------------------
    # Deduplication helpers
    # ---------------------------------------------------------------------
    def _deduplicate(self, evidences: List[Evidence]) -> List[Evidence]:
        """Return a list with duplicates removed.

        Duplicate detection order:
        * ``evidence_id`` – first occurrence wins.
        * canonical ``url`` – first occurrence wins; provenance of later
          duplicates is merged into the retained item under ``duplicate_sources``.
        * exact ``text`` (whitespace‑normalised) – first occurrence wins.
        """
        seen_ids: Set[str] = set()
        seen_urls: Set[str] = set()
        seen_texts: Set[str] = set()
        unique: List[Evidence] = []

        for ev in evidences:
            # ----- evidence_id -----
            if ev.evidence_id in seen_ids:
                logger.debug("Duplicate evidence_id %s", ev.evidence_id)
                continue
            # ----- url -----
            canon = None
            if ev.url:
                try:
                    canon = _canonicalize_url(ev.url)
                except Exception as exc:  # pragma: no cover – defensive
                    logger.warning("Failed to canonicalise URL %s: %s", ev.url, exc)
                    canon = ev.url
                if canon in seen_urls:
                    logger.debug("Duplicate URL %s (canonical %s)", ev.url, canon)
                    original = next(item for item in unique if _canonicalize_url(item.url or "") == canon)
                    merged_meta = dict(original.metadata)
                    dup_entry = {
                        "evidence_id": ev.evidence_id,
                        "source": ev.source,
                        "retrieval_method": ev.retrieval_method,
                    }
                    dup_list = merged_meta.get("duplicate_sources", [])
                    dup_list.append(dup_entry)
                    merged_meta["duplicate_sources"] = dup_list
                    updated = original.copy(update={"metadata": merged_meta})
                    idx = unique.index(original)
                    unique[idx] = updated
                    continue
            # ----- text -----
            text_norm = ev.text.strip() if ev.text else ""
            # Duplicate text detection applies only when evidence has no URL.
            if ev.url is None:
                # No special-casing; apply duplicate text detection for all evidences without a URL.
                if text_norm in seen_texts:
                    logger.debug("Duplicate text for evidence_id %s", ev.evidence_id)
                    continue
                # Record this text as seen among URL‑less evidences.
                seen_texts.add(text_norm)

            # ----- keep -----
            seen_ids.add(ev.evidence_id)
            if canon:
                seen_urls.add(canon)
            # For evidences with URLs we still track their text to avoid exact duplicates later.
            if ev.url is not None:
                seen_texts.add(text_norm)
            unique.append(ev)
        return unique

    # ---------------------------------------------------------------------
    # Ranking helpers
    # ---------------------------------------------------------------------
    def _normalize_similarity(self, scores: List[float | None]) -> List[float | None]:
        """Min‑max normalise similarity scores to the interval [0, 1].

        If all scores are identical (or the list is empty) the function returns
        ``1.0`` for every element so that similarity does not affect ranking.
        """
        if not scores:
            return []
        known = [score for score in scores if score is not None]
        if not known:
            return [None for _ in scores]
        min_score = min(known)
        max_score = max(known)
        if max_score == min_score:
            return [1.0 if score is not None else None for score in scores]
        return [
            None if score is None else (score - min_score) / (max_score - min_score)
            for score in scores
        ]

    def _ranking_score(self, ev: Evidence, norm_sim: float | None) -> float:
        """Deterministic *retrieval‑quality* score.

        Weighted components (weights sum to 1.0):
        * **Similarity** – ``norm_sim`` (0‑1) weight **0.5**.
        * **Authority** – raw ``authority_score`` (or ``0.0`` if missing) weight **0.3**.
        * **Recency** – linear decay based on ``publication_date`` (newer = higher),
          weight **0.2**.  The decay is ``max(0, 1 - days/365)`` where *days* is the
          difference between ``datetime.utcnow()`` (naïve UTC) and the publication
          date.

        Missing values are treated as ``0`` (except similarity which is always
        provided after normalisation).  The score is *not* a truth or credibility
        metric – it purely reflects how well the evidence matches the query and
        its freshness.
        """
        # Normalized similarity (0‑1) is already provided as ``norm_sim``.
        # Missing similarity is explicitly neutral/absent, never maximum relevance.
        sim = norm_sim if norm_sim is not None else 0.0
        # Normalise authority to a 0‑1 scale. The project treats ``authority_score`` as 0‑10.
        if ev.authority_score is not None:
            auth_norm = max(0.0, min(ev.authority_score / 10.0, 1.0))
        else:
            auth_norm = 0.0
        # Recency factor: newer publications get higher scores, bounded to [0,1].
        recency = 0.0
        if ev.publication_date:
            try:
                days = (datetime.utcnow() - ev.publication_date).days
                recency = max(0.0, 1.0 - days / 365.0)
            except Exception:
                recency = 0.0
        # Final deterministic retrieval‑quality score.
        return 0.5 * sim + 0.3 * auth_norm + 0.2 * recency

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def fuse(self, local_evidence: List[Evidence], external_evidence: List[Evidence], top_k: int) -> List[Evidence]:
        """Merge, deduplicate and rank evidence.

        Parameters
        ----------
        local_evidence: List[Evidence]
            Evidence produced by the FAISS local retriever (``retrieval_method`` = ``"faiss_local"``).
        external_evidence: List[Evidence]
            Evidence from external providers (``retrieval_method`` = ``"external"``).
        top_k: int
            Number of items to return; must be positive.
        """
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        combined = list(local_evidence) + list(external_evidence)
        unique = self._deduplicate(combined)
        raw_sims = [ev.similarity_score for ev in unique]
        norm_sims = self._normalize_similarity(raw_sims)
        scored = [(self._ranking_score(ev, ns), ev) for ev, ns in zip(unique, norm_sims)]
        def sort_key(item):
            score, ev = item
            auth = ev.authority_score if ev.authority_score is not None else -1.0
            sim = ev.similarity_score if ev.similarity_score is not None else -1.0
            pub_ts = ev.publication_date.timestamp() if ev.publication_date else -1.0
            return (-score, -auth, -sim, -pub_ts, ev.evidence_id)
        scored.sort(key=sort_key)
        return [ev for _, ev in scored[:top_k]]

# -------------------------------------------------------------------------
# Source‑diversity note
# -------------------------------------------------------------------------
# At this stage we deliberately do **not** enforce a hard source‑diversity rule.
# The ranking formula already favours higher‑quality evidence (similarity,
# authority, recency).  Forcing diversity could demote a clearly superior item,
# which would violate the deterministic‑quality‑first philosophy of Phase 5F.
# Future phases may introduce explicit diversity constraints if required.
