"""Prompt Builder for Supervisor Agent.
Converts a supervisor_input dict into a single prompt string for the LLM.
"""

from typing import Dict, Any, List, Union
from .evidence_schema import Evidence


def _extract_evidence_dict(evidence: Any) -> Dict[str, Any]:
    """Extract dictionary from Evidence object or dict."""
    if hasattr(evidence, "model_dump"):
        return evidence.model_dump()
    elif hasattr(evidence, "dict"):
        return evidence.dict()
    elif isinstance(evidence, dict):
        return dict(evidence)
    return {"text": str(evidence)}


def _format_evidence_item(evidence: Any, index: int) -> str:
    """Format a single Evidence object or dict into readable prompt text."""
    ev_data = _extract_evidence_dict(evidence)
    ev_id = ev_data.get("evidence_id") or ev_data.get("chunk_id", f"evidence_{index}")
    source = ev_data.get("source", "Unknown Source")
    url = ev_data.get("url") if ev_data.get("url") else "not available"
    authority = ev_data.get("authority_score") if ev_data.get("authority_score") is not None else "unknown"
    similarity = ev_data.get("similarity_score", "unknown")
    text = ev_data.get("text", "")

    return (
        f"[Evidence ID: {ev_id} | Source: {source}]\n"
        f"  - Evidence ID: {ev_id}\n"
        f"  - Source: {source}\n"
        f"  - Authority Score: {authority}\n"
        f"  - Similarity Score: {similarity}\n"
        f"  - URL: {url}\n"
        f"  - Text: \"{text}\"\n"
    )


def build_supervisor_prompt(supervisor_input: Dict[str, Any]) -> str:
    """
    Build a comprehensive prompt for the Supervisor Agent:
        1. States the claim and its domain.
        2. Lists each evidence item with actual ID, source, url, authority, similarity, text.
        3. Includes reflection_notes (quality, credibility, contradictions, sufficiency).
        4. Informs the agent of attempt count and whether maximum attempts were reached.
        5. Gives clear rules on final verdict (Real, Fake, Unverified) and anti-hallucination.
        6. Strictly enforces referencing only real evidence IDs in key_evidence_used.
    """
    claim = supervisor_input.get("claim_text") or supervisor_input.get("claim", "")
    domain = supervisor_input.get("domain", "")
    evidence_list: List[Any] = supervisor_input.get("evidence_list", [])
    reflection_notes = supervisor_input.get("reflection_notes")
    persona_insights = supervisor_input.get("persona_insights", [])
    verification_result = supervisor_input.get("verification_result")
    attempt_count = supervisor_input.get("attempt_count", 1)
    maximum_attempts_reached = supervisor_input.get("maximum_attempts_reached", False)

    prompt_parts = []
    
    prompt_parts.append("==================================================")
    prompt_parts.append("SUPERVISOR FACT-CHECKING ASSESSMENT")
    prompt_parts.append("==================================================")
    prompt_parts.append(f"Claim: \"{claim}\"")
    prompt_parts.append(f"Domain: {domain}")
    prompt_parts.append(f"Retrieval Attempt Count: {attempt_count}")
    prompt_parts.append(f"Maximum Attempts Reached: {maximum_attempts_reached}\n")

    evidence_count = len(evidence_list)
    prompt_parts.append(f"Total Evidence Items Provided: {evidence_count}")
    if not evidence_list:
        prompt_parts.append("No evidence provided.\n")
    else:
        for idx, ev in enumerate(evidence_list, 1):
            prompt_parts.append(f"Evidence Item {idx}:")
            prompt_parts.append(_format_evidence_item(ev, idx))

    prompt_parts.append("Reflection Analysis:")
    if reflection_notes:
        if isinstance(reflection_notes, dict):
            decision = reflection_notes.get("decision", "N/A")
            is_suff = reflection_notes.get("is_evidence_sufficient", "N/A")
            conf = reflection_notes.get("confidence", "N/A")
            source_cred = reflection_notes.get("source_credibility_score", "N/A")
            ev_quality = reflection_notes.get("evidence_quality_score", "N/A")
            claim_supp = reflection_notes.get("claim_supported", "N/A")
            claim_contra = reflection_notes.get("claim_contradicted", "N/A")
            ev_contra = reflection_notes.get("evidence_items_contradict_each_other", "N/A")
            notes = reflection_notes.get("notes", "")

            refl_text = (
                f"- Decision: {decision}\n"
                f"- Is Evidence Sufficient: {is_suff}\n"
                f"- Reflection Confidence: {conf}\n"
                f"- Source Credibility Score: {source_cred}\n"
                f"- Evidence Quality Score: {ev_quality}\n"
                f"- Claim Supported: {claim_supp}\n"
                f"- Claim Contradicted: {claim_contra}\n"
                f"- Evidence Items Contradict Each Other: {ev_contra}\n"
                f"- Analysis Notes: {notes}"
            )
            prompt_parts.append(refl_text + "\n")
        else:
            prompt_parts.append(f"{reflection_notes}\n")
    else:
        prompt_parts.append("No reflection analysis is available.\n")

    prompt_parts.append("Overall Verification Analysis:")
    if not verification_result:
        prompt_parts.append("No overall verification result is available.\n")
    else:
        if hasattr(verification_result, "model_dump"):
            verification_data = verification_result.model_dump()
        elif hasattr(verification_result, "dict"):
            verification_data = verification_result.dict()
        elif isinstance(verification_result, dict):
            verification_data = verification_result
        else:
            verification_data = {}

        overall_assessment = verification_data.get(
            "overall_assessment",
            "N/A",
        )
        overall_confidence = verification_data.get(
            "overall_confidence",
            "N/A",
        )

        prompt_parts.append(
            f"- Overall Assessment: {overall_assessment}\n"
            f"- Overall Verification Confidence: {overall_confidence}\n"
        )

        verification_items = verification_data.get("results", [])

        if verification_items:
            prompt_parts.append("Per-Evidence Verification Results:")

            for idx, item in enumerate(verification_items, 1):
                if hasattr(item, "model_dump"):
                    item_data = item.model_dump()
                elif hasattr(item, "dict"):
                    item_data = item.dict()
                elif isinstance(item, dict):
                    item_data = item
                else:
                    item_data = {}

                prompt_parts.append(
                    f"  {idx}. Evidence ID: "
                    f"{item_data.get('evidence_id', 'N/A')} | "
                    f"Relationship: "
                    f"{item_data.get('relationship', 'N/A')} | "
                    f"Confidence: "
                    f"{item_data.get('confidence', 'N/A')} | "
                    f"Reasoning: "
                    f"{item_data.get('reasoning', '')}\n"
                )

        prompt_parts.append(
            "Treat the Overall Verification Analysis as a direct signal "
            "from the evidence verification layer. If the assessment is "
            "CONFLICTING_EVIDENCE, do not ignore the conflict; only select "
            "Real or Fake when the supplied evidence and downstream "
            "analysis clearly resolve it. Otherwise prefer Unverified.\n"
        )

    prompt_parts.append("Persona Analysis:")
    if not persona_insights:
        prompt_parts.append("No persona analysis is available.\n")
    else:
        for idx, persona in enumerate(persona_insights, 1):
            if hasattr(persona, "model_dump"):
                persona_data = persona.model_dump()
            elif hasattr(persona, "dict"):
                persona_data = persona.dict()
            elif isinstance(persona, dict):
                persona_data = persona
            else:
                persona_data = {"reasoning": str(persona)}

            prompt_parts.append(
                f"Persona {idx}: {persona_data.get('persona', 'Unknown')}\n"
                f"- Assessment: {persona_data.get('assessment', 'N/A')}\n"
                f"- Confidence: {persona_data.get('confidence', 'N/A')}\n"
                f"- Evidence IDs: {persona_data.get('evidence_ids', [])}\n"
                f"- Reasoning: {persona_data.get('reasoning', '')}\n"
            )

    prompt_parts.append("Use persona analysis as supporting reasoning only. "
                        "Do not treat persona opinions as independent evidence.\n")

    prompt_parts.append(f"""
CRITICAL VERDICT GUIDELINES:
1. DECISION RULES:
   - "Real": The claim is verified as true by reliable, authoritative evidence.
   - "Fake": The claim is proven false or contradicted by authoritative evidence, or fabricated.
   - "Unverified": Evidence is insufficient, low-authority (e.g. unknown blogs with low authority score), unconfirmed, or contradictory.
   - If maximum attempts were reached ({maximum_attempts_reached}) and evidence remained low quality or insufficient, decide "Unverified" or "Fake" based on the evidence; do NOT label it "Real" without high-authority corroboration.
   - NEVER call a claim "Real" merely because evidence text has similar keywords or high semantic similarity if source authority is low.

2. STRICT ANTI-HALLUCINATION RULES:
   - Use ONLY actual evidence provided in the Evidence list above.
   - Do NOT invent sources, URLs, or evidence.
   - If Total Evidence Items Provided is {evidence_count}, do NOT say "multiple sources" or "various sources" if {evidence_count} <= 1!
   - For "key_evidence_used", include ONLY the actual Evidence ID and Source name from the list above, formatted as 'SourceName (evidence_id)'. Example: 'Reserve Bank of India (evidence_1)'.
   - NEVER invent evidence IDs like 'doc_999' or use generic placeholders like 'Item 1' or 'Source 1'.

Return ONLY valid JSON matching this exact structure:
{{
  "final_verdict": "Real" | "Fake" | "Unverified",
  "confidence_score": <integer 0-100>,
  "trust_score": <integer 0-100>,
  "reasoning_summary": "<2-4 sentences explaining the decision based strictly on provided facts>",
  "key_evidence_used": ["<actual SourceName (actual evidence_id)>"]
}}
""")

    return "\n".join(prompt_parts)
