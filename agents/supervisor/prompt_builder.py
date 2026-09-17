"""
Prompt Builder for Supervisor Agent.
Converts a supervisor_input dict into a single prompt string for the LLM.
"""

from typing import Dict, Any, List
from .evidence_schema import Evidence

def _format_evidence_item(evidence: Evidence) -> str:
    """Format a single Evidence object into readable prompt text."""
    source = evidence.source
    url = evidence.url if evidence.url else "not available"
    authority = evidence.authority_score if evidence.authority_score is not None else "unknown"
    similarity = evidence.similarity_score
    text = evidence.text
    
    return (
        f"[Evidence ID: {evidence.evidence_id} | Source: {source}]\n"
        f"- Source: {source}\n"
        f"  URL: {url}\n"
        f"  Authority Score: {authority}\n"
        f"  Similarity Score: {similarity}\n"
        f"  Text: {text}\n"
    )

def build_supervisor_prompt(supervisor_input: Dict[str, Any]) -> str:
    """
    Build a prompt that:
        1. States the claim and its domain.
        2. Lists each evidence item with source, url, authority, similarity, text.
        3. Includes reflection_notes if present, otherwise explicitly tells the LLM.
        4. Gives clear instructions for decision making.
        5. Requires JSON format matching a specific schema.
        6. Explains output fields.
    """
    claim = supervisor_input.get("claim_text", "")
    domain = supervisor_input.get("domain", "")
    evidence_list: List[Evidence] = supervisor_input.get("evidence_list", [])
    reflection_notes = supervisor_input.get("reflection_notes")

    prompt_parts = []
    
    prompt_parts.append(f"Claim: {claim}")
    prompt_parts.append(f"Domain: {domain}\n")
    
    prompt_parts.append("Evidence:")
    if not evidence_list:
        prompt_parts.append("No evidence provided.\n")
    else:
        for idx, ev in enumerate(evidence_list, 1):
            prompt_parts.append(f"Evidence Item {idx}:")
            prompt_parts.append(_format_evidence_item(ev))
    
    prompt_parts.append("Reflection Analysis:")
    if reflection_notes:
        prompt_parts.append(f"{reflection_notes}\n")
    else:
        prompt_parts.append("No reflection analysis is available.\n")
        
    prompt_parts.append("""
Instructions:
You must decide whether each piece of evidence supports or contradicts the claim itself. This has not been precomputed.
Based on the evidence, reflection analysis (if any), and your judgment, evaluate the claim.

You MUST respond with ONLY valid JSON, matching the exact schema below, and nothing else - no preamble, no markdown code fences:

{
  "final_verdict": "Real" | "Fake" | "Unverified",
  "confidence_score": <integer 0-100>,
  "trust_score": <integer 0-100>,
  "reasoning_summary": "<string>",
  "key_evidence_used": ["<evidence_id or source>", ...]
}

Field Descriptions:
- final_verdict: overall judgment on the claim
- confidence_score: how confident the model is in this verdict, based on evidence quality/quantity
- trust_score: how trustworthy the overall evidence base is (source authority, agreement across sources)
- reasoning_summary: 2-4 sentences explaining the decision
- key_evidence_used: which evidence_id(s) most influenced the decision. For key_evidence_used, you MUST reference each item using its actual source name and evidence_id from the evidence list provided above, formatted as 'SourceName (evidence_id)' - for example 'Reuters (chunk_001)'. NEVER use generic labels like 'Evidence Item 1', 'Item 1', 'Source 1', or similar placeholders. If an evidence item's source is unknown, use its evidence_id alone.
""")
    
    return "\n".join(prompt_parts)
