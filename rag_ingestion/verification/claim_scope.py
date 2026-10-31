"""Claim scope and the shared eligibility rule for verification evidence."""

import re
from typing import Literal


ClaimScope = Literal["GENERAL", "EVENT_SPECIFIC"]

_PAST_ACTION = re.compile(
    r"\b(?:[A-Za-z]+ed|cut|fell|rose|won|lost|held|made|took|gave|brought)\b",
    re.IGNORECASE,
)
_TIME_ANCHOR = re.compile(
    r"\b(?:today|yesterday|tomorrow)\b|"
    r"\b(?:on|in|at|during|last|next|this)\s+"
    r"(?:\d{4}|\d{1,2}|Monday|Tuesday|Wednesday|Thursday|Friday|"
    r"Saturday|Sunday|January|February|March|April|May|June|July|"
    r"August|September|October|November|December|week|month|year)\b",
    re.IGNORECASE,
)
_SUBJECT_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
_NAME_CONNECTORS = {"the", "of", "and", "for", "a", "an"}


def classify_claim_scope(claim: str) -> ClaimScope:
    """Identify a completed occurrence; leave ambiguous propositions general.

    A dated completed action or a completed action by a named actor identifies
    an event even when the extracted claim omits its date. Callers can provide
    an explicit scope when the wording alone does not settle that distinction.
    """
    action = _PAST_ACTION.search(claim)
    if action is None:
        return "GENERAL"
    subject = _SUBJECT_WORD.findall(claim[:action.start()])
    named_actor = any(
        (word.isupper() and len(word) > 1)
        or (index > 0 and word[0].isupper() and word.lower() not in _NAME_CONNECTORS)
        for index, word in enumerate(subject)
    )
    if named_actor or _TIME_ANCHOR.search(claim):
        return "EVENT_SPECIFIC"
    return "GENERAL"


def verification_is_eligible(result: object, claim_scope: ClaimScope) -> bool:
    """Whether an aligned result can establish support or contradiction."""
    if isinstance(result, dict):
        aligned = result.get("claim_aligned", False)
        same_event = result.get("same_event", False)
    else:
        aligned = getattr(result, "claim_aligned", False)
        same_event = getattr(result, "same_event", False)
    return aligned is True and (claim_scope == "GENERAL" or same_event is True)


def verification_can_establish_claim(result: object, claim_scope: ClaimScope) -> bool:
    """Whether a result is scope-eligible and directly establishes its relation."""
    if isinstance(result, dict):
        direct = result.get("directly_establishes_proposition", False)
    else:
        direct = getattr(result, "directly_establishes_proposition", False)
    return direct is True and verification_is_eligible(result, claim_scope)
