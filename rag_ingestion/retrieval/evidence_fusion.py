import logging
import urllib.parse
import re
import time
from datetime import datetime
from typing import List, Set, Dict, Any

from rag_ingestion.retrieval.evidence_schema import Evidence

logger = logging.getLogger(__name__)

MINIMUM_CLAIM_RELEVANCE = 0.10
MINIMUM_SEMANTIC_SIMILARITY = 0.55
MAX_SEMANTIC_PASSAGE_CHARS = 256
MAX_SEMANTIC_CANDIDATES = 5


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
        self.semantic_embedder = None
        self._semantic_query_cache: Dict[str, Any] = {}
        self._semantic_passage_cache: Dict[tuple[str, str, str], float] = {}
        self.last_stage_durations: Dict[str, float] = {}

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
                logger.info(
                    "evidence_diag drop stage=fusion_dedup evidence_id=%s reason=duplicate",
                    ev.evidence_id,
                )
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
                    logger.info(
                        "evidence_diag drop stage=fusion_dedup evidence_id=%s reason=duplicate",
                        ev.evidence_id,
                    )
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
                    logger.info(
                        "evidence_diag drop stage=fusion_dedup evidence_id=%s reason=duplicate",
                        ev.evidence_id,
                    )
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

    @staticmethod
    def _claim_features(claim: str) -> set[str]:
        stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "been", "by",
            "did", "do", "does", "for", "from", "has", "have", "in",
            "into", "is", "it", "of", "on", "or", "that", "the", "this",
            "to", "was", "were", "with",
        }
        return {
            token.casefold()
            for token in re.findall(r"[^\W_]+(?:[.%][^\W_]+)*", claim or "")
            if len(token) > 1 and token.casefold() not in stopwords
        }

    def _claim_relevance(
        self, ev: Evidence, similarity: float | None, claim: str
    ) -> tuple[float, int, int, bool, bool]:
        """Return claim relevance and lexical overlap counts, excluding quality signals."""
        claim_tokens = self._claim_features(claim)
        if not claim_tokens:
            return 1.0, 0, 0, False, False
        evidence_text = f"{ev.title or ''} {ev.text or ''}"
        evidence_tokens = self._claim_features(evidence_text)
        overlap = len(claim_tokens & evidence_tokens)
        lexical = overlap / len(claim_tokens)
        claim_values = set(re.findall(r"\d+(?:\.\d+)?%?", claim or ""))
        evidence_values = set(re.findall(r"\d+(?:\.\d+)?%?", evidence_text))
        numeric = 1.0 if claim_values and claim_values & evidence_values else 0.0
        claim_years = {t for t in claim_tokens if re.fullmatch(r"(?:19|20)\d{2}", t)}
        evidence_years = {t for t in evidence_tokens if re.fullmatch(r"(?:19|20)\d{2}", t)}
        if claim_years and not evidence_years and ev.publication_date:
            evidence_years = {str(ev.publication_date.year)}
        temporal_mismatch = bool(claim_years and evidence_years and not claim_years & evidence_years)
        temporal = (
            1.0 if claim_years and claim_years & evidence_years
            else -0.35 if temporal_mismatch
            else 0.0
        )
        semantic = max(0.0, min(float(similarity), 1.0)) if similarity is not None else 0.0
        score = 0.25 * semantic + 0.25 * lexical + 0.35 * numeric + 0.10 * temporal
        return score, overlap, len(claim_tokens), bool(numeric or temporal > 0), temporal_mismatch

    @staticmethod
    def _political_news_provider(ev: Evidence) -> str | None:
        if ev.retrieval_method != "external":
            return None
        metadata = ev.metadata or {}
        provider = str(
            metadata.get("external_source_type") or metadata.get("provider") or ""
        ).casefold()
        if provider in {"news_api", "newsapi", "gdelt"}:
            return provider
        return None

    def _political_news_is_relevant(self, ev: Evidence, claim: str) -> bool:
        claim_terms = {
            token for token in self._claim_features(claim)
            if len(token) >= 4
            and token not in {
                "about", "after", "against", "from", "have", "into", "over",
                "that", "their", "there", "these", "they", "this", "those",
                "were", "with",
            }
        }
        evidence_terms = self._claim_features(f"{ev.title or ''} {ev.text or ''}")
        return bool(claim_terms & evidence_terms)

    def _ranking_score(self, ev: Evidence, norm_sim: float | None, claim: str | None = None) -> float:
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
            auth_norm = max(0.0, min(ev.authority_score if ev.authority_score <= 1.0 else ev.authority_score / 10.0, 1.0))
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
        if not claim:
            return 0.5 * sim + 0.3 * auth_norm + 0.2 * recency

        claim_tokens = self._claim_features(claim)
        evidence_text = f"{ev.title or ''} {ev.text or ''}"
        # Use co-occurrence in one passage for ranking. Whole-page overlap lets
        # broad reference pages accumulate every claim term across unrelated
        # sections and crowd out a concise statement of the relationship.
        lexical = self._passage_alignment_score(claim, ev)
        evidence_tokens = self._claim_features(evidence_text)
        title_tokens = self._claim_features(ev.title or "")
        title_match = 0.15 if title_tokens and title_tokens <= claim_tokens else 0.0
        claim_values = set(re.findall(r"\d+(?:\.\d+)?%?", claim or ""))
        evidence_values = set(re.findall(r"\d+(?:\.\d+)?%?", evidence_text))
        numeric = 1.0 if claim_values and claim_values & evidence_values else 0.0
        claim_years = {t for t in claim_tokens if re.fullmatch(r"(?:19|20)\d{2}", t)}
        evidence_years = {t for t in evidence_tokens if re.fullmatch(r"(?:19|20)\d{2}", t)}
        temporal = 0.0
        if claim_years:
            temporal = 1.0 if claim_years & evidence_years else (-0.35 if evidence_years else 0.0)
        content_factor = 0.5 if (ev.metadata or {}).get("content_quality") == "metadata_or_snippet" else 1.0
        return content_factor * (
            0.25 * sim + 0.25 * lexical + 0.35 * numeric + 0.10 * temporal + 0.05 * auth_norm + title_match
        )

    @classmethod
    def _passage_alignment_score(cls, claim: str, evidence: Evidence) -> float:
        """Measure ordered claim-term alignment within one evidence sentence."""
        stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "been", "by",
            "did", "do", "does", "for", "from", "has", "have", "in",
            "into", "is", "it", "of", "on", "or", "that", "the", "this",
            "to", "was", "were", "with",
        }

        def sequence(value: str) -> list[str]:
            result = []
            for token in re.findall(r"[^\W_]+(?:[.%][^\W_]+)*", value.casefold()):
                if len(token) <= 1 or token in stopwords:
                    continue
                for suffix in ("ingly", "edly", "ing", "ed", "es", "s"):
                    if len(token) > len(suffix) + 3 and token.endswith(suffix):
                        token = token[:-len(suffix)]
                        break
                result.append(token)
            return result

        terms = sequence(claim)
        if not terms:
            return 0.0
        sentences = [
            sentence for sentence in re.split(r"(?<=[.!?])\s+|\r?\n+", evidence.text or "")
            if sentence.strip()
        ]
        if not sentences:
            sentences = [evidence.text or ""]

        def ordered_coverage(sentence: str) -> float:
            passage_terms = sequence(sentence)
            # Longest common subsequence rewards preserving subject/action/
            # object order while allowing ordinary extra words in evidence.
            row = [0] * (len(passage_terms) + 1)
            for claim_token in terms:
                next_row = [0]
                for index, passage_token in enumerate(passage_terms, start=1):
                    next_row.append(
                        row[index - 1] + 1 if claim_token == passage_token
                        else max(row[index], next_row[-1])
                    )
                row = next_row
            return row[-1] / len(terms)

        return max((ordered_coverage(sentence) for sentence in sentences), default=0.0)

    @staticmethod
    def _entity_overlap_score(claim: str, evidence: Evidence) -> float:
        """Measure overlap of generically detected capitalized entity tokens."""
        ignored = {"the", "a", "an", "this", "that"}
        claim_entities = {
            token.casefold()
            for token in re.findall(r"\b[A-Z][a-zA-Z0-9]+\b", claim or "")
            if token.casefold() not in ignored
        }
        if not claim_entities:
            return 0.0
        evidence_entities = {
            token.casefold()
            for token in re.findall(
                r"\b[A-Z][a-zA-Z0-9]+\b", f"{evidence.title or ''} {evidence.text or ''}"
            )
            if token.casefold() not in ignored
        }
        return len(claim_entities & evidence_entities) / len(claim_entities)

    @classmethod
    def _entity_relation_alignment_score(cls, claim: str, evidence: Evidence) -> float:
        """Find a shared claim predicate between the claim's entities in one sentence."""
        ignored = {"the", "a", "an", "this", "that"}
        claim_entities = [
            token.casefold()
            for token in re.findall(r"\b[A-Z][a-zA-Z0-9]+\b", claim or "")
            if token.casefold() not in ignored
        ]
        if len(claim_entities) < 2:
            return 0.0
        entities = {claim_entities[0], claim_entities[-1]}
        claim_predicates = cls._claim_features(claim) - entities
        if not claim_predicates:
            return 0.0
        sentences = [
            sentence for sentence in re.split(r"(?<=[.!?])\s+|\r?\n+", evidence.text or "")
            if sentence.strip()
        ]
        best = 0.0
        for sentence in sentences:
            words = [token.casefold() for token in re.findall(r"[^\W_]+", sentence)]
            positions = {
                entity: [index for index, word in enumerate(words) if word == entity]
                for entity in entities
            }
            if not all(positions.values()):
                continue
            left_entity, right_entity = claim_entities[0].casefold(), claim_entities[-1].casefold()
            for left in positions[left_entity]:
                for right in positions[right_entity]:
                    between = words[min(left, right) + 1:max(left, right)]
                    if claim_predicates & set(between):
                        best = 1.0
        return best

    def _semantic_scores(self, claim: str | None, evidences: List[Evidence]) -> Dict[str, float]:
        """Score claim/passage meaning with the already configured local embedder.

        This is a ranking signal only. It does not accept or reject evidence and
        deliberately fails open to the established deterministic ranker when
        the embedding model is unavailable.
        """
        embedder = self.semantic_embedder
        if not claim or not evidences or embedder is None:
            return {}
        model = getattr(embedder, "_model", None)
        if model is None or not callable(getattr(model, "encode", None)):
            return {}
        try:
            import numpy as np
            total_started = time.perf_counter()
            query_encode_seconds = 0.0
            query_encode_count = 0
            query_vector = self._semantic_query_cache.get(claim)
            if query_vector is None:
                query_started = time.perf_counter()
                query_vector = np.asarray(model.encode(
                    [claim], normalize_embeddings=True, show_progress_bar=False,
                )[0], dtype=float)
                query_encode_seconds = time.perf_counter() - query_started
                query_encode_count = 1
                self._semantic_query_cache[claim] = query_vector
            missing = []
            keys = {}
            output = {}
            for item in evidences:
                passage = self._semantic_passage_text(claim, item)
                key = (claim, item.evidence_id, passage)
                keys[item.evidence_id] = key
                if key in self._semantic_passage_cache:
                    output[item.evidence_id] = self._semantic_passage_cache[key]
                else:
                    missing.append((item.evidence_id, key, passage))
            if missing:
                passage_started = time.perf_counter()
                vectors = model.encode(
                    [row[2] for row in missing],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                passage_encode_seconds = time.perf_counter() - passage_started
                for (evidence_id, key, _), vector in zip(missing, vectors):
                    score = max(0.0, min(1.0, float(np.dot(query_vector, np.asarray(vector, dtype=float)))))
                    self._semantic_passage_cache[key] = score
                    output[evidence_id] = score
                if len(self._semantic_passage_cache) > 512:
                    self._semantic_passage_cache.clear()
            else:
                passage_encode_seconds = 0.0
            passages = [row[2] for row in missing]
            token_counts = self._semantic_token_counts(model, passages)
            claim_token_count = self._semantic_token_counts(model, [claim])
            logger.info(
                "semantic_embedding_batch model_loaded=true model=%s candidate_count=%d "
                "query_encode_count=%d query_encode_seconds=%.4f "
                "passage_encode_calls=%d passage_encode_count=%d cached_passages=%d "
                "passage_encode_seconds=%.4f total_embedding_seconds=%.4f "
                "claim_tokens=%s passage_chars_total=%d passage_chars_max=%d passage_tokens=%s",
                type(model).__name__, len(evidences), query_encode_count,
                query_encode_seconds, int(bool(missing)), len(missing),
                len(evidences) - len(missing), passage_encode_seconds,
                time.perf_counter() - total_started,
                claim_token_count,
                sum(map(len, passages)), max(map(len, passages), default=0),
                token_counts,
            )
            return output
        except Exception as exc:
            logger.info("evidence_diag semantic_rank=unavailable reason=%s", type(exc).__name__)
            return {}

    @classmethod
    def _semantic_passage_text(cls, claim: str, evidence: Evidence) -> str:
        """Build a bounded title-plus-passage excerpt using generic claim overlap."""
        title = (evidence.title or "").strip()
        body = (evidence.text or "").strip()
        budget = max(0, MAX_SEMANTIC_PASSAGE_CHARS - len(title) - 2)
        if not body or budget == 0:
            return title[:MAX_SEMANTIC_PASSAGE_CHARS]

        claim_terms = cls._claim_features(claim)
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+|\r?\n+", body)
            if sentence.strip()
        ]
        ranked = []
        for index, sentence in enumerate(sentences):
            overlap = len(claim_terms & cls._claim_features(sentence))
            if overlap:
                alignment = cls._passage_alignment_score(
                    claim, evidence.model_copy(update={"text": sentence}),
                )
                ranked.append((alignment, overlap, -index, sentence))
        if ranked:
            selected = sorted(ranked, key=lambda item: (-item[0], -item[1], -item[2]))[:3]
            selected.sort(key=lambda item: -item[2])
            body_excerpt = " ".join(sentence for _, _, _, sentence in selected)
        else:
            body_excerpt = body
        body_excerpt = body_excerpt[:budget].strip()
        return f"{title}. {body_excerpt}".strip(" .") if title else body_excerpt

    @staticmethod
    def _semantic_token_counts(model, passages: List[str]) -> List[int] | None:
        """Return the bounded input token lengths used by the loaded BGE tokenizer."""
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None or not passages:
            return []
        try:
            encoded = tokenizer(
                passages,
                add_special_tokens=True,
                truncation=True,
                max_length=getattr(model, "max_seq_length", 512),
                padding=False,
            )
            return [len(tokens) for tokens in encoded.get("input_ids", [])]
        except Exception:
            return None

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def fuse(
        self,
        local_evidence: List[Evidence],
        external_evidence: List[Evidence],
        top_k: int,
        claim: str | None = None,
        domain: str | None = None,
        apply_minimum_relevance: bool = True,
        semantic_ranking: bool = False,
    ) -> List[Evidence]:
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
        stage_started = time.perf_counter()
        combined = list(local_evidence) + list(external_evidence)
        unique = self._deduplicate(combined)
        dedup_seconds = time.perf_counter() - stage_started
        logger.info(
            "evidence_diag fusion_input local=%d external=%d combined=%d deduplicated=%d",
            len(local_evidence),
            len(external_evidence),
            len(combined),
            len(unique),
        )
        # Filter on claim relevance before selecting top-k. Similarity is taken
        # from the existing FAISS signal (when present); lexical, numeric and
        # temporal signals come from the same claim-aware fusion implementation.
        quality_started = time.perf_counter()
        if claim and apply_minimum_relevance:
            retained = []
            for ev in unique:
                (
                    relevance,
                    overlap,
                    claim_term_count,
                    has_exact_context,
                    temporal_mismatch,
                ) = self._claim_relevance(ev, ev.similarity_score, claim)
                minimum_overlap = min(2, claim_term_count)
                semantic_below_floor = (
                    ev.similarity_score is not None
                    and ev.similarity_score < MINIMUM_SEMANTIC_SIMILARITY
                )
                claim_years = set(re.findall(r"(?:19|20)\d{2}", claim))
                evidence_years = set(re.findall(
                    r"(?:19|20)\d{2}", f"{ev.title or ''} {ev.text or ''}"
                ))
                if not evidence_years and ev.publication_date:
                    evidence_years = {str(ev.publication_date.year)}
                year_match = (
                    None if not claim_years else bool(claim_years & evidence_years)
                )
                rejection_reasons = []
                if temporal_mismatch:
                    rejection_reasons.append("year_mismatch")
                if relevance < MINIMUM_CLAIM_RELEVANCE:
                    rejection_reasons.append("below_minimum_relevance")
                if semantic_below_floor:
                    rejection_reasons.append("semantic_below_floor")
                if overlap < minimum_overlap and not has_exact_context:
                    rejection_reasons.append("insufficient_lexical_overlap")
                logger.info(
                    "evidence_diag candidate stage=fusion_relevance evidence_id=%s "
                    "source=%r semantic_similarity=%s lexical_overlap=%d/%d "
                    "claim_years=%s evidence_years=%s year_match=%s "
                    "composite_relevance=%.4f accepted=%s rejection_reason=%s",
                    ev.evidence_id,
                    ev.source,
                    f"{ev.similarity_score:.4f}" if ev.similarity_score is not None else "missing",
                    overlap,
                    claim_term_count,
                    sorted(claim_years),
                    sorted(evidence_years),
                    "not_applicable" if year_match is None else str(year_match).lower(),
                    relevance,
                    not rejection_reasons,
                    ",".join(rejection_reasons) if rejection_reasons else "none",
                )
                if rejection_reasons:
                    logger.info(
                        "evidence_diag drop stage=fusion_relevance evidence_id=%s "
                        "reason=%s relevance=%.4f similarity=%s "
                        "overlap=%d/%d threshold=%.2f semantic_floor=%.2f",
                        ev.evidence_id,
                        ",".join(rejection_reasons),
                        relevance,
                        f"{ev.similarity_score:.4f}" if ev.similarity_score is not None else "missing",
                        overlap,
                        claim_term_count,
                        MINIMUM_CLAIM_RELEVANCE,
                        MINIMUM_SEMANTIC_SIMILARITY,
                    )
                    continue
                retained.append(ev)
            unique = retained
        quality_seconds = time.perf_counter() - quality_started

        raw_sims = [ev.similarity_score for ev in unique]
        norm_sims = self._normalize_similarity(raw_sims)
        semantic_started = time.perf_counter()
        semantic_candidates = unique
        if semantic_ranking and claim and len(unique) > MAX_SEMANTIC_CANDIDATES:
            semantic_candidates = sorted(
                unique,
                key=lambda ev: self._ranking_score(ev, None, claim),
                reverse=True,
            )[:MAX_SEMANTIC_CANDIDATES]
        semantic_scores = self._semantic_scores(claim, semantic_candidates) if semantic_ranking else {}
        semantic_seconds = time.perf_counter() - semantic_started
        rank_started = time.perf_counter()
        scored = []
        for ev, ns in zip(unique, norm_sims):
            base_score = self._ranking_score(ev, ns, claim)
            semantic_score = semantic_scores.get(ev.evidence_id)
            # Preserve all prior ranking/quality signals, while giving a direct
            # claim-to-passage semantic match enough influence to order external
            # passages that have no FAISS similarity value.
            if semantic_score is not None and claim:
                alignment = self._passage_alignment_score(claim, ev)
                score = 0.30 * base_score + 0.30 * semantic_score + 0.40 * alignment
            else:
                score = base_score
            scored.append((score, ev))
        # Syndicated copies remain traceable, but are not treated as independent
        # confirmations. The first item from a source family keeps its score;
        # later items receive a generic diminishing factor.
        family_counts = {}
        adjusted = []
        for score, ev in sorted(scored, key=lambda item: -item[0]):
            family = (ev.metadata or {}).get("source_family")
            if family and ev.title:
                # Distinct articles from one publisher are separate candidates;
                # the diminishing factor is for syndicated copies of one story.
                family = (family, ev.title.strip().casefold())
            if family:
                count = family_counts.get(family, 0)
                family_counts[family] = count + 1
                score *= 1.0 if count == 0 else 0.75 ** count
            adjusted.append((score, ev))
        scored = adjusted
        def sort_key(item):
            score, ev = item
            auth = ev.authority_score if ev.authority_score is not None else -1.0
            sim = ev.similarity_score if ev.similarity_score is not None else -1.0
            pub_ts = ev.publication_date.timestamp() if ev.publication_date else -1.0
            return (-score, -auth, -sim, -pub_ts, ev.evidence_id)
        scored.sort(key=sort_key)
        selected = scored[:top_k]
        if domain and domain.casefold() in {"politics", "political"} and claim:
            relevant_news = [
                item for item in scored
                if self._political_news_provider(item[1])
                and self._political_news_is_relevant(item[1], claim)
            ]
            if relevant_news:
                # Reserve a small part of the fused set for independent current
                # news reporting, then fill remaining slots by normal relevance.
                reserved_count = min(2, top_k, len(relevant_news))
                reserved: list[tuple[float, Evidence]] = []
                seen_providers: set[str] = set()
                for item in relevant_news:
                    provider = self._political_news_provider(item[1])
                    if provider not in seen_providers:
                        reserved.append(item)
                        seen_providers.add(provider)
                    if len(reserved) == reserved_count:
                        break
                if len(reserved) < reserved_count:
                    reserved_ids = {item[1].evidence_id for item in reserved}
                    reserved.extend(
                        item for item in relevant_news
                        if item[1].evidence_id not in reserved_ids
                    )
                    reserved = reserved[:reserved_count]
                reserved_ids = {item[1].evidence_id for item in reserved}
                selected = reserved + [
                    item for item in scored
                    if item[1].evidence_id not in reserved_ids
                ][: top_k - len(reserved)]
        selected_ids = {ev.evidence_id for _, ev in selected}
        for _, ev in scored:
            if ev.evidence_id not in selected_ids:
                logger.info(
                    "evidence_diag drop stage=fusion_rank evidence_id=%s reason=top_k_limit",
                    ev.evidence_id,
                )
        logger.info(
            "evidence_diag fusion_output ranked=%d final=%d top_k=%d",
            len(scored),
            len(selected),
            top_k,
        )
        if claim:
            for final_score, ev in selected:
                tokens = self._claim_features(claim)
                evidence_tokens = self._claim_features(f"{ev.title or ''} {ev.text or ''}")
                overlap = len(tokens & evidence_tokens)
                claim_years = set(re.findall(r"(?:19|20)\d{2}", claim))
                evidence_years = set(re.findall(r"(?:19|20)\d{2}", f"{ev.title or ''} {ev.text or ''}"))
                if claim_years:
                    temporal_score = 1.0 if claim_years & evidence_years else (-0.35 if evidence_years else 0.0)
                else:
                    temporal_score = 0.0
                entity_overlap = self._entity_overlap_score(claim, ev)
                authority = ev.authority_score
                authority_score = max(0.0, min(authority if authority <= 1 else authority / 10, 1.0)) if authority is not None else 0.0
                logger.info(
                    "evidence_selection claim=%r evidence_id=%s source=%r title=%r provider=%r "
                    "excerpt=%r lexical_score=%.4f semantic_score=%s entity_overlap=%.4f "
                    "temporal_score=%.4f credibility_score=%.4f final_ranking_score=%.4f",
                    claim, ev.evidence_id, ev.source, ev.title,
                    (ev.metadata or {}).get("provider") or (ev.metadata or {}).get("external_source_type") or ev.retrieval_method,
                    (ev.text or "")[:900], overlap / max(1, len(tokens)),
                    f"{semantic_scores[ev.evidence_id]:.4f}" if ev.evidence_id in semantic_scores else "unavailable",
                    entity_overlap, temporal_score,
                    authority_score, final_score,
                )
        self.last_stage_durations = {
            "deduplication": dedup_seconds,
            "quality_filter": quality_seconds,
            "semantic_scoring": semantic_seconds,
            "ranking_and_selection": time.perf_counter() - rank_started,
        }
        logger.info(
            "evidence_fusion_stage_durations deduplication=%.4fs quality_filter=%.4fs "
            "semantic_scoring=%.4fs ranking_and_selection=%.4fs candidate_count=%d "
            "filtered_count=%d ranked_count=%d selected_count=%d",
            dedup_seconds, quality_seconds, semantic_seconds,
            self.last_stage_durations["ranking_and_selection"], len(combined),
            len(unique), len(scored), len(selected),
        )
        return [ev for _, ev in selected]

# -------------------------------------------------------------------------
# Source‑diversity note
# -------------------------------------------------------------------------
# At this stage we deliberately do **not** enforce a hard source‑diversity rule.
# The ranking formula already favours higher‑quality evidence (similarity,
# authority, recency).  Forcing diversity could demote a clearly superior item,
# which would violate the deterministic‑quality‑first philosophy of Phase 5F.
# Future phases may introduce explicit diversity constraints if required.
