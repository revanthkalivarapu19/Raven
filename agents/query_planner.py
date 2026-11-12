"""Small bounded query-planning entry point for claim components."""

import re
from datetime import datetime
from typing import Callable, Optional


_STOP_WORDS = {
    "a", "an", "and", "as", "at", "be", "been", "being", "by", "for",
    "from", "had", "has", "have", "in", "into", "is", "it", "its", "of",
    "on", "or", "that", "the", "their", "this", "to", "was", "were",
    "will", "with", "while", "would",
}


def _opposing_relation_query(claim: str) -> str | None:
    """Produce a polarity-reversed query for simple declarative relations."""
    text = " ".join((claim or "").split()).strip().rstrip(".!?;:")
    # Modal predicates take negation directly after the modal. This also
    # handles passive clauses ("can be inserted") without mistaking the noun
    # immediately before the modal for the predicate.
    modal = re.match(
        r"^(?P<subject>.+?)\s+(?P<modal>can|could|may|might|must|should|will|would)\s+(?P<neg>not\s+)?(?P<predicate>.+)$",
        text, re.IGNORECASE,
    )
    if modal:
        subject, word, predicate = modal.group("subject", "modal", "predicate")
        negative = bool(modal.group("neg"))
        if negative:
            if word.casefold() == "cannot":
                return f"{subject} can {predicate}"
            return f"{subject} {word} {predicate}"
        if word.casefold() == "can":
            return f"{subject} cannot {predicate}"
        return f"{subject} {word} not {predicate}"
    # Auxiliary predicates have an unambiguous generic polarity transform.
    aux = re.match(
        r"^(?P<subject>.+?)\s+(?P<aux>does|do|did)\s+(?P<neg>not\s+)?(?P<verb>[\w'-]+)(?P<tail>\s+.+)?$",
        text, re.IGNORECASE,
    )
    if aux:
        subject, verb = aux.group("subject", "verb")
        tail = aux.group("tail") or ""
        negative = bool(aux.group("neg"))
        if negative:
            return f"{subject} {aux.group('aux')} {verb}{tail}"
        return f"{subject} {aux.group('aux')} not {verb}{tail}"

    copula = re.match(
        r"^(?P<subject>.+?)\s+(?P<aux>is|are|was|were|has|have)\s+(?P<neg>not\s+)?(?P<tail>.+)$",
        text, re.IGNORECASE,
    )
    if copula:
        subject, aux_word, tail = copula.group("subject", "aux", "tail")
        if copula.group("neg"):
            return f"{subject} {aux_word} {tail}"
        if aux_word.casefold() in {"has", "have"}:
            # Perfect aspect (has/have + participle) negates after the
            # auxiliary; possessive have uses do-support instead.
            first_tail_word = tail.split(maxsplit=1)[0].casefold()
            if first_tail_word.endswith(("ed", "en")):
                return f"{subject} {aux_word} not {tail}"
            # Retain a natural agreement for common plural subjects.
            do_aux = "do" if subject.casefold().endswith("s") or subject.casefold() in {"they", "we", "you"} else "does"
            return f"{subject} {do_aux} not have {tail}"
        return f"{subject} {aux_word} not {tail}"

    # Simple present and past transitive predicates are common in atomic
    # factual claims. Reverse their polarity without changing arguments.
    simple = re.match(
        r"^(?P<subject>.+?)\s+(?P<verb>[A-Za-z]+(?:ies|ied|es|s|ed))\s+(?P<tail>.+)$",
        text,
    )
    if not simple:
        return None
    subject, verb, tail = simple.group("subject", "verb", "tail")
    lower = verb.casefold()
    if lower.endswith("ies") and len(lower) > 4:
        base = verb[:-3] + "y"
        aux_word = "does"
    elif lower.endswith("zes") and len(lower) > 5 and lower[-4].casefold() in "aeiou":
        # Preserve the final e in verbs such as "freezes" -> "freeze".
        base = verb[:-1]
        aux_word = "does"
    elif lower.endswith("es") and len(lower) > 4 and lower[:-2].endswith(("s", "x", "z", "ch", "sh")):
        base = verb[:-2]
        aux_word = "does"
    elif lower.endswith("s") and len(lower) > 3:
        base = verb[:-1]
        aux_word = "does"
    elif lower.endswith("ied") and len(lower) > 4:
        base = verb[:-3] + "y"
        aux_word = "did"
    elif lower.endswith("ed") and len(lower) > 4:
        base = verb[:-2]
        if len(base) > 2 and base[-1] == base[-2] and base[-1].casefold() in {"b", "d", "g", "m", "n", "p", "r", "t"}:
            base = base[:-1]
        elif base.endswith(("c", "v", "g")):
            base += "e"
        aux_word = "did"
    else:
        return None
    return f"{subject} {aux_word} not {base} {tail}"


def _with_opposing_query(queries: list[str], claim: str, max_queries: int) -> list[str]:
    opposing = _opposing_relation_query(claim)
    if not opposing:
        return queries[:max_queries]
    direct = " ".join((claim or "").split()).rstrip(".!?;:")
    # The first two slots retain the direct claim and its compact relationship
    # query; the final slot searches for the generic polarity-opposed relation.
    if len(queries) >= 2:
        # Numeric multi-fact expansions are already separate focused queries;
        # retain those, while ordinary claims get direct + compact relation.
        numbers = re.findall(r"(?<!\w)\d[\d,.]*", claim)
        if len(set(numbers)) > 1:
            combined = [*queries[:max_queries - 1], opposing]
        else:
            combined = [direct, queries[0], opposing]
    else:
        combined = [direct, *queries, opposing]
    return list(dict.fromkeys(combined))[:max_queries]


def plan_atomic_queries(
    claim: str, *, context: str | None = None, max_queries: int = 3,
    include_current_date: bool = False,
) -> list[str]:
    """Build compact, deterministic, source-grounded queries for one atomic claim.

    This deliberately avoids an LLM planner for concise propositions. It keeps
    names, acronyms, numbers and dates intact, while dropping grammatical
    filler and sentence punctuation.
    """
    text = " ".join((claim or "").split()).strip()
    if not text:
        return []

    # Capture multiword proper names and acronyms before token filtering.
    entities = []
    entity_pattern = r"\b[A-Z][\w&.'’-]*(?:\s+[A-Z][\w&.'’-]*)*\b"
    for match in re.finditer(entity_pattern, text):
        value = re.sub(r"^(?:The|A|An)\s+", "", match.group(0).strip(" .,;:()"))
        if value and value.casefold() not in _STOP_WORDS and value not in entities:
            entities.append(value)
    component_entities = list(entities)

    # Recover grounded names, places and dates omitted from the atomic wording
    # by selecting the source clause with the strongest lexical connection.
    context_text = " ".join((context or "").split()).strip()
    if context_text:
        claim_terms = {
            token.casefold() for token in re.findall(r"[\w]+", text)
            if token.casefold() not in _STOP_WORDS
        }
        clauses = re.split(r"[,;]|\bwhile\b|\bwhereas\b", context_text, flags=re.IGNORECASE)
        ranked_clauses = sorted(
            clauses,
            key=lambda clause: len(claim_terms & {
                token.casefold() for token in re.findall(r"[\w]+", clause)
                if token.casefold() not in _STOP_WORDS
            }),
            reverse=True,
        )
        if ranked_clauses:
            best_clause = ranked_clauses[0]
            for match in re.finditer(entity_pattern, best_clause):
                value = re.sub(r"^(?:The|A|An)\s+", "", match.group(0).strip(" .,;:()"))
                if value and value.casefold() not in _STOP_WORDS and value.casefold() not in {
                    entity.casefold() for entity in entities
                }:
                    entities.append(value)
            for value in re.findall(
                r"(?:19|20)\d{2}|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b",
                best_clause, flags=re.IGNORECASE,
            ):
                if value.casefold() not in {entity.casefold() for entity in entities}:
                    entities.append(value)

    tokens = re.findall(r"(?:₹|\$|€|£)?\d[\d,.]*(?:%| percent)?|[\w&.'’-]+", text)
    salient = []
    for token in tokens:
        clean = token.strip(" .,;:()")
        if not clean or clean.casefold() in _STOP_WORDS:
            continue
        if clean not in salient:
            salient.append(clean)

    # Favor named entities and numeric/date anchors, then retain content words.
    anchors = []
    for value in entities + [token for token in salient if re.search(r"\d", token)]:
        if value.casefold() not in {item.casefold() for item in anchors}:
            anchors.append(value)
    anchor_tokens = {
        token.casefold()
        for value in anchors
        for token in re.findall(r"[\w&.'’-]+", value)
    }
    content = [token for token in salient if token.casefold() not in anchor_tokens]
    primary = (anchors + content)[:12]
    queries = []
    if primary:
        queries.append(" ".join(primary))

    # A second phrase query emphasizes the event/action terms while keeping
    # every term grounded in this component.
    secondary_content = content[-4:] if len(content) > 4 else content
    secondary = " ".join(
        [f'"{value}"' for value in anchors[:5]] + secondary_content
    ).strip()
    if secondary and secondary.casefold() not in {query.casefold() for query in queries}:
        queries.append(secondary)
    if entities and re.search(r"\b(?:is|are|was|were)\b", text, re.I):
        subject_query = f'"{entities[0]}"'
        if subject_query.casefold() not in {query.casefold() for query in queries}:
            queries.insert(1, subject_query)

    # For current-event phrasing without an explicit date, add today's month
    # and year as search context, never as a claim assertion.
    wants_current_context = include_current_date or bool(re.search(
        r"\b(?:today|yesterday|currently|recently|following|amid|breaking)\b",
        text, re.IGNORECASE,
    ))
    explicit_year = bool(re.search(r"\b(?:19|20)\d{2}\b", text))
    current = datetime.now().strftime("%B %Y") if wants_current_context and not explicit_year else ""

    number_pattern = r"(?<!\w)\d[\d,.]*(?:\s*(?:%|percent|million|billion|thousand))?"
    claim_numbers = list(dict.fromkeys(
        re.sub(r"\D", "", match.group(0))
        for match in re.finditer(number_pattern, text, flags=re.IGNORECASE)
        if re.sub(r"\D", "", match.group(0))
    ))
    if claim_numbers and context_text:
        context_numbers = list(re.finditer(number_pattern, context_text or text, flags=re.IGNORECASE))
        target_queries = []
        for target in claim_numbers[:3]:
            match_index = next((index for index, match in enumerate(context_numbers)
                                if re.sub(r"\D", "", match.group(0)) == target), None)
            source = context_text or text
            words = list(re.finditer(r"[\w&.'’₹$€£,-]+", source))
            center = None
            if match_index is not None:
                number_match = context_numbers[match_index]
                center = next((index for index, word in enumerate(words)
                               if word.start() <= number_match.start() < word.end()), None)
            if center is None:
                source_numbers = list(re.finditer(number_pattern, text, flags=re.IGNORECASE))
                source_match = next((item for item in source_numbers
                                     if re.sub(r"\D", "", item.group(0)) == target), None)
                if source_match:
                    center = next((index for index, word in enumerate(words)
                                   if word.start() <= source_match.start() < word.end()), None)
            if center is None:
                continue
            window = words[max(0, center - 4):center + 5]
            terms = [word.group(0).strip(" .,;:()") for word in window]
            terms = [term for term in terms if term and term.casefold() not in _STOP_WORDS]
            terms = [term for term in terms if not re.search(r"\d", term) or re.sub(r"\D", "", term) == target]
            terms = [term for term in terms if re.sub(r"\D", "", term) == target
                     or term.isupper() or len(term) >= 4]
            contextual_entities = component_entities + [
                entity for entity in entities
                if entity.casefold() not in {item.casefold() for item in component_entities}
            ]
            term_set = {term.casefold() for term in terms}
            contextual_entities = [entity for entity in contextual_entities
                                   if entity.casefold() not in term_set
                                   and (not entity.isupper() or entity.casefold() in term_set)]
            query_terms = []
            # Keep at most two distinctive nonnumeric terms for context. A
            # longer tail reintroduces noisy words from the full sentence.
            remainder = [token for token in salient
                         if not re.search(r"\d", token)
                         and token.casefold() not in _STOP_WORDS
                         and token.casefold() not in {term.casefold() for term in terms}]
            for term in contextual_entities + terms + remainder[:2]:
                if term.casefold() not in {item.casefold() for item in query_terms}:
                    query_terms.append(term)
            if current:
                query_terms = query_terms[:10] + current.split()
            query = " ".join(query_terms[:12])
            if query and query.casefold() not in {item.casefold() for item in target_queries}:
                target_queries.append(query)
        if target_queries:
            if len(claim_numbers) == 1 and len(target_queries) == 1:
                focused_terms = target_queries[0].split()
                quoted = " ".join(f'"{term}"' for term in focused_terms[:2]) + " " + " ".join(focused_terms[2:])
                target_queries.append(quoted)
            return _with_opposing_query(
                target_queries, text, max(1, min(max_queries, 3)),
            )

    if current:
        dated = f"{queries[0]} {current}" if queries else current
        if dated.casefold() not in {query.casefold() for query in queries}:
            queries.append(dated)
    return _with_opposing_query(queries, text, max(1, min(max_queries, 3)))


def plan_queries(
    claim: str,
    *,
    context: Optional[str] = None,
    domain: Optional[str] = None,
    expanded_planner: Optional[Callable[..., list[str]]] = None,
) -> list[str]:
    """Use the original concise claim directly; delegate only long claims.

    The existing grounded expansion logic remains the source of any expanded
    queries. This wrapper keeps planning separate from retrieval orchestration.
    """
    text = " ".join((claim or "").split()).strip()
    if not text:
        return []
    word_count = len(re.findall(r"[^\W_]+(?:[.%][^\W_]+)*", text))
    concise = 1 <= word_count <= 32 and len(text) <= 280 and "\n" not in claim
    if concise or expanded_planner is None:
        return [text]
    try:
        planned = expanded_planner(
            claim=text, retrieval_context=context, domain=domain,
        )
        if isinstance(planned, list) and planned and all(isinstance(q, str) and q.strip() for q in planned):
            return list(dict.fromkeys(q.strip() for q in planned))[:3]
    except Exception:
        pass
    return [text]
