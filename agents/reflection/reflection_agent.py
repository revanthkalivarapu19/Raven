import json
import re
from typing import Any, Dict, List, Optional, Union
import ollama

# Centralized Quality & Credibility Thresholds
MIN_AUTHORITY_THRESHOLD = 0.60
LOW_AUTHORITY_THRESHOLD = 0.35
MIN_SIMILARITY_THRESHOLD = 0.65
MIN_EVIDENCE_QUALITY_THRESHOLD = 0.60
HIGH_CONFIDENCE_THRESHOLD = 0.70


class ReflectionAgent:
    """Reflection Agent for multi-agent fake news verification.
    
    Evaluates retrieved evidence sufficiency, source credibility, semantic similarity,
    and distinguishes between:
      1. Contradiction between Claim and Evidence (claim_supported vs claim_contradicted)
      2. Contradiction between different Evidence items (evidence_items_contradict_each_other)
    
    Outputs YES/NO for evidence sufficiency/reliability (NOT Real/Fake/Unverified).
    """

    def __init__(self, model: str = "llama3.1"):
        self.model = model

    def _convert_evidence(self, evidence: Any) -> Dict[str, Any]:
        """Normalize evidence into a dictionary representation."""
        if hasattr(evidence, "model_dump"):
            return evidence.model_dump()
        elif hasattr(evidence, "dict"):
            return evidence.dict()
        elif isinstance(evidence, dict):
            return dict(evidence)
        else:
            return {"text": str(evidence)}

    def _prepare_evidence_list(self, evidence_list: List[Any]) -> List[Dict[str, Any]]:
        return [self._convert_evidence(ev) for ev in evidence_list]

    def _deterministic_metrics(self, evidence_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute objective deterministic metrics from evidence metadata."""
        if not evidence_list:
            return {
                "count": 0,
                "avg_similarity": 0.0,
                "avg_authority": 0.0,
                "max_authority": 0.0,
                "min_authority": 0.0,
                "has_low_authority_only": True,
                "has_authoritative_source": False
            }

        sims = []
        auths = []
        for ev in evidence_list:
            sim = ev.get("similarity_score")
            try:
                sims.append(float(sim) if sim is not None else 0.5)
            except (ValueError, TypeError):
                sims.append(0.5)

            auth = ev.get("authority_score")
            try:
                auths.append(float(auth) if auth is not None else 0.3)
            except (ValueError, TypeError):
                auths.append(0.3)

        avg_sim = sum(sims) / len(sims)
        avg_auth = sum(auths) / len(auths)
        max_auth = max(auths)
        min_auth = min(auths)

        return {
            "count": len(evidence_list),
            "avg_similarity": round(avg_sim, 4),
            "avg_authority": round(avg_auth, 4),
            "max_authority": round(max_auth, 4),
            "min_authority": round(min_auth, 4),
            "has_low_authority_only": max_auth <= LOW_AUTHORITY_THRESHOLD,
            "has_authoritative_source": max_auth >= MIN_AUTHORITY_THRESHOLD
        }

    def _clean_hallucinations(self, text: str, evidence_count: int) -> str:
        """Deterministic guard to avoid hallucinating multiple sources when count is 1."""
        if evidence_count <= 1:
            # Replace hallucinated plural references
            text = re.sub(r"\bmultiple sources\b", "the single retrieved source", text, flags=re.IGNORECASE)
            text = re.sub(r"\bvarious sources\b", "the retrieved source", text, flags=re.IGNORECASE)
            text = re.sub(r"\bsources confirm\b", "the source confirms", text, flags=re.IGNORECASE)
            text = re.sub(r"\bsources agree\b", "the source states", text, flags=re.IGNORECASE)
            text = re.sub(r"\bsources contradict each other\b", "no conflicting sources exist", text, flags=re.IGNORECASE)
        return text

    def reflect(
        self,
        claim_text: Union[str, Dict[str, Any]] = None,
        domain: Optional[str] = None,
        evidence_list: Optional[List[Any]] = None,
        input_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Evaluate retrieved evidence for a claim."""
        # Support input_data dictionary format
        if isinstance(claim_text, dict) and domain is None and evidence_list is None:
            input_data = claim_text
            claim_text = input_data.get("claim_text", "")
            domain = input_data.get("domain", "")
            evidence_list = input_data.get("evidence_list", [])
        elif input_data is not None:
            claim_text = input_data.get("claim_text", claim_text or "")
            domain = input_data.get("domain", domain or "")
            evidence_list = input_data.get("evidence_list", evidence_list or [])

        # Validate inputs
        if not claim_text or not str(claim_text).strip():
            return {
                "decision": "NO",
                "is_evidence_sufficient": False,
                "contradictions_found": False,
                "confidence": 0.0,
                "notes": "No claim was provided.",
                "support_score": 0.0,
                "source_credibility_score": 0.0,
                "evidence_quality_score": 0.0,
                "claim_supported": False,
                "claim_contradicted": False,
                "evidence_items_contradict_each_other": False
            }

        domain = str(domain).strip() if domain else "General"

        if not evidence_list:
            return {
                "decision": "NO",
                "is_evidence_sufficient": False,
                "contradictions_found": False,
                "confidence": 0.0,
                "notes": "No evidence was provided. Evidence retrieval is required.",
                "support_score": 0.0,
                "source_credibility_score": 0.0,
                "evidence_quality_score": 0.0,
                "claim_supported": False,
                "claim_contradicted": False,
                "evidence_items_contradict_each_other": False
            }

        # Normalize evidence items
        normalized_evidence = self._prepare_evidence_list(evidence_list)
        metrics = self._deterministic_metrics(normalized_evidence)
        evidence_count = metrics["count"]

        # Format evidence for prompt
        evidence_blocks = []
        for idx, ev in enumerate(normalized_evidence, start=1):
            source = ev.get("source", "Unknown Source")
            sim = ev.get("similarity_score", "N/A")
            auth = ev.get("authority_score", "N/A")
            url = ev.get("url", "N/A")
            ev_id = ev.get("evidence_id") or ev.get("chunk_id", f"evidence_{idx}")
            title = ev.get("title", "N/A")
            text = ev.get("text", "")
            retrieval_method = ev.get("retrieval_method", "N/A")

            evidence_blocks.append(
                f"EVIDENCE ITEM {idx}:\n"
                f"  - Evidence ID: {ev_id}\n"
                f"  - Source: {source}\n"
                f"  - Authority Score: {auth}\n"
                f"  - Similarity Score: {sim}\n"
                f"  - Title: {title}\n"
                f"  - URL: {url}\n"
                f"  - Retrieval Method: {retrieval_method}\n"
                f"  - Text: \"{text}\"\n"
            )

        evidence_str = "\n".join(evidence_blocks)

        prompt = f"""You are the Reflection Agent in a multi-agent fact-checking system.
Your job is to evaluate whether the retrieved evidence is sufficient, reliable, and relevant for the Supervisor Agent.
You DO NOT issue a final truth verdict (Real/Fake/Unverified). That is solely the Supervisor's job.

CLAIM:
"{claim_text}"

DOMAIN:
{domain}

TOTAL EVIDENCE ITEMS RETRIEVED: {evidence_count}

RETRIEVED EVIDENCE:
{evidence_str}

CRITICAL ARCHITECTURAL RULES:
1. EVALUATE SOURCE CREDIBILITY vs SIMILARITY:
   - Similarity measures semantic relevance. Authority measures source reliability.
   - A high similarity score with low authority (e.g., Unknown Blog, similarity=0.95, authority=0.10) is NOT strong evidence.
   - If sources are unknown, blog-based, or have low authority (authority < 0.40), source_credibility_score MUST be low and decision must be NO.

2. CRITICAL DISTINCTION ABOUT CONTRADICTIONS:
   - Distinguish contradiction between CLAIM and EVIDENCE:
     If evidence states the opposite of the claim: claim_supported=false, claim_contradicted=true.
     This does NOT mean evidence items contradict each other if only one evidence exists or all evidence agree with each other!
   - Evidence vs Evidence contradiction:
     evidence_items_contradict_each_other=true ONLY if two or more evidence items in the retrieved list contradict each other.
     If there is only 1 evidence item, evidence_items_contradict_each_other MUST be false.

3. STRICT ANTI-HALLUCINATION:
   - Do NOT say "multiple sources confirm" if only 1 evidence item is present! (Total evidence items retrieved: {evidence_count}).
   - Do NOT invent sources, URLs, or evidence.
   - Do NOT assume RBI or official sources exist unless explicitly listed in the RETRIEVED EVIDENCE above.

4. DECISION LOGIC:
   - Return "YES" only if evidence has reasonable authority, relevance, supports or clearly addresses the claim, and is sufficient.
   - Return "NO" if evidence is weak, unauthoritative, from an unknown source, low similarity, missing essential facts, or contains unresolved conflicts.

Return ONLY a valid JSON object with EXACTLY this structure:
{{
    "decision": "YES" or "NO",
    "is_evidence_sufficient": true or false,
    "contradictions_found": true or false,
    "confidence": 0.0 to 1.0,
    "notes": "Concise factual analysis without hallucinating sources",
    "support_score": 0.0 to 1.0,
    "source_credibility_score": 0.0 to 1.0,
    "evidence_quality_score": 0.0 to 1.0,
    "claim_supported": true or false,
    "claim_contradicted": true or false,
    "evidence_items_contradict_each_other": true or false
}}
"""

        raw_result = None
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"temperature": 0.1}
            )
            content = response.get("message", {}).get("content", "").strip()

            # Clean code fences if present
            cleaned = re.sub(r"^```json\s*", "", content, flags=re.IGNORECASE)
            cleaned = re.sub(r"```\s*$", "", cleaned)
            cleaned = cleaned.strip()

            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1:
                raw_result = json.loads(cleaned[start:end + 1])
        except Exception as e:
            raw_result = {
                "decision": "NO",
                "is_evidence_sufficient": False,
                "contradictions_found": False,
                "confidence": 0.0,
                "notes": f"LLM execution error: {str(e)}",
                "support_score": 0.0,
                "source_credibility_score": 0.0,
                "evidence_quality_score": 0.0,
                "claim_supported": False,
                "claim_contradicted": False,
                "evidence_items_contradict_each_other": False
            }

        if not raw_result or not isinstance(raw_result, dict):
            raw_result = {
                "decision": "NO",
                "is_evidence_sufficient": False,
                "contradictions_found": False,
                "confidence": 0.0,
                "notes": "Failed to parse valid JSON from LLM.",
                "support_score": 0.0,
                "source_credibility_score": 0.0,
                "evidence_quality_score": 0.0,
                "claim_supported": False,
                "claim_contradicted": False,
                "evidence_items_contradict_each_other": False
            }

        # =========================================================
        # DETERMINISTIC SAFEGUARDS & VALIDATION AROUND LLM OUTPUT
        # =========================================================

        # Parse & Validate Booleans
        claim_supported = bool(raw_result.get("claim_supported", False))
        claim_contradicted = bool(raw_result.get("claim_contradicted", False))
        ev_items_contradict = bool(raw_result.get("evidence_items_contradict_each_other", False))

        # Enforce: 1 evidence item CANNOT contradict each other
        if evidence_count <= 1:
            ev_items_contradict = False

        # contradictions_found: true if evidence items contradict each other
        contradictions_found = ev_items_contradict or bool(raw_result.get("contradictions_found", False))
        # If only 1 evidence item and claim is contradicted, contradictions_found between items is False
        if evidence_count <= 1:
            contradictions_found = False

        # Parse Scores with Clamping
        def safe_float(val: Any, default: float = 0.0) -> float:
            try:
                f = float(val)
                return max(0.0, min(1.0, f))
            except (ValueError, TypeError):
                return default

        support_score = safe_float(raw_result.get("support_score"), 0.0)
        source_credibility_score = safe_float(raw_result.get("source_credibility_score"), metrics["avg_authority"])
        evidence_quality_score = safe_float(raw_result.get("evidence_quality_score"), (metrics["avg_similarity"] * 0.4 + metrics["avg_authority"] * 0.6))
        confidence = safe_float(raw_result.get("confidence"), 0.0)

        raw_decision = str(raw_result.get("decision", "NO")).upper().strip()
        decision = "YES" if raw_decision == "YES" else "NO"

        raw_notes = str(raw_result.get("notes", "Evidence evaluation completed."))
        notes = self._clean_hallucinations(raw_notes, evidence_count)

        # ---------------------------------------------------------
        # SAFEGUARD: LOW SOURCE AUTHORITY MUST NOT PASS AS HIGH-CONFIDENCE YES
        # ---------------------------------------------------------
        if metrics["has_low_authority_only"]:
            # If all sources have low authority (e.g., Unknown Blog authority <= 0.35)
            source_credibility_score = min(source_credibility_score, metrics["max_authority"])
            evidence_quality_score = min(evidence_quality_score, metrics["max_authority"])

            if decision == "YES" or confidence > 0.50:
                decision = "NO"
                confidence = min(confidence, 0.40)
                notes = (
                    f"Low source credibility detected (max authority: {metrics['max_authority']:.2f}). "
                    f"Even if semantic similarity is high, unverified/blog sources are insufficient."
                )

        # ---------------------------------------------------------
        # SAFEGUARD: UNRESOLVED CONTRADICTIONS AMONG EVIDENCE ITEMS
        # ---------------------------------------------------------
        if ev_items_contradict:
            decision = "NO"
            confidence = min(confidence, 0.50)
            notes += " Unresolved contradictions found between retrieved evidence items."

        # ---------------------------------------------------------
        # SAFEGUARD: EVIDENCE QUALITY THRESHOLD FOR "YES"
        # ---------------------------------------------------------
        if decision == "YES":
            if evidence_quality_score < MIN_EVIDENCE_QUALITY_THRESHOLD or source_credibility_score < LOW_AUTHORITY_THRESHOLD:
                decision = "NO"
                confidence = min(confidence, 0.45)
                notes += f" Evidence quality score ({evidence_quality_score:.2f}) or source credibility ({source_credibility_score:.2f}) falls below threshold."

        is_evidence_sufficient = (decision == "YES")

        return {
            "decision": decision,
            "is_evidence_sufficient": is_evidence_sufficient,
            "contradictions_found": contradictions_found,
            "confidence": round(confidence, 2),
            "notes": notes,
            "support_score": round(support_score, 2),
            "source_credibility_score": round(source_credibility_score, 2),
            "evidence_quality_score": round(evidence_quality_score, 2),
            "claim_supported": claim_supported,
            "claim_contradicted": claim_contradicted,
            "evidence_items_contradict_each_other": ev_items_contradict
        }