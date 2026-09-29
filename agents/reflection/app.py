import sys
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional

from agents.reflection.reflection_agent import ReflectionAgent
from agents.supervisior.supervisor_agent import run_supervisor
from agents.supervisior.evidence_schema import Evidence
from agents.xai.xai_agent import XAIAgent

MAX_ATTEMPTS = 3
HIGH_CONFIDENCE_THRESHOLD = 0.70


def create_test_evidence(
    evidence_text: str,
    domain: str,
    count: int,
    source: str,
    authority_score: float,
    similarity_score: float,
    url: Optional[str] = None,
    title: Optional[str] = None
) -> Evidence:
    return Evidence(
        evidence_id=f"evidence_{count}",
        chunk_id=f"chunk_{count}",
        document_id=f"document_{count}",
        text=evidence_text,
        domain=domain,
        source=source,
        language="en",
        similarity_score=similarity_score,
        retrieval_method="faiss_local",
        metadata={},
        claim_id=None,
        url=url,
        title=title or f"Test Evidence {count}",
        publication_date=datetime.now(),
        authority_score=authority_score
    )


def run_reflection(claim: str, domain: str, evidence_list: List[Evidence]) -> Dict[str, Any]:
    reflection_agent = ReflectionAgent()
    return reflection_agent.reflect(
        claim_text=claim,
        domain=domain,
        evidence_list=evidence_list
    )


def execute_pipeline(
    claim: str,
    domain: str,
    attempts_data: Optional[List[List[Evidence]]] = None,
    interactive: bool = True
) -> Dict[str, Any]:
    """Execute the full Claim -> Retrieval -> Reflection -> Supervisor -> XAI pipeline.
    
    Adheres strictly to the 3-attempt maximum retry rule.
    """
    print("\n" + "=" * 70)
    print("       MULTI-AGENT VERIFICATION PIPELINE (Ollama / llama3.1)")
    print("=" * 70)
    print(f"Claim: \"{claim}\"")
    print(f"Domain: {domain}")

    reflection_agent = ReflectionAgent()
    xai_agent = XAIAgent()

    final_evidence_list: List[Evidence] = []
    last_reflection_result: Dict[str, Any] = {}
    maximum_attempts_reached = False
    completed_attempt = 1

    for attempt in range(1, MAX_ATTEMPTS + 1):
        completed_attempt = attempt
        print("\n" + "-" * 70)
        print(f"ATTEMPT {attempt} of {MAX_ATTEMPTS}: Evidence Retrieval & Reflection")
        print("-" * 70)

        # 1. Retrieve / Collect Evidence
        current_attempt_evidence: List[Evidence] = []

        if attempts_data and attempt <= len(attempts_data):
            current_attempt_evidence = attempts_data[attempt - 1]
            print(f"[Attempt {attempt}] Loaded {len(current_attempt_evidence)} evidence item(s).")
            for idx, ev in enumerate(current_attempt_evidence, 1):
                print(f"  ({idx}) Source: {ev.source} | Authority: {ev.authority_score} | Sim: {ev.similarity_score}")
                print(f"      Text: {ev.text}")
        elif interactive:
            print(f"\nEnter evidence for Attempt {attempt} (press Enter on empty text when done):")
            count = 1
            while True:
                ev_text = input(f"Evidence {count} text: ").strip()
                if not ev_text:
                    break
                ev_source = input(f"Source {count}: ").strip() or "Unknown Source"
                
                try:
                    sim_input = input(f"Similarity score {count} (0-1, default 0.85): ").strip()
                    ev_sim = float(sim_input) if sim_input else 0.85
                except ValueError:
                    ev_sim = 0.85

                try:
                    auth_input = input(f"Authority score {count} (0-1, default 0.50): ").strip()
                    ev_auth = float(auth_input) if auth_input else 0.50
                except ValueError:
                    ev_auth = 0.50

                ev_url = input(f"URL {count} (optional): ").strip() or None

                item = create_test_evidence(
                    evidence_text=ev_text,
                    domain=domain,
                    count=len(final_evidence_list) + count,
                    source=ev_source,
                    authority_score=ev_auth,
                    similarity_score=ev_sim,
                    url=ev_url
                )
                current_attempt_evidence.append(item)
                count += 1
        else:
            print("[Warning] No evidence provided for this attempt.")

        if not current_attempt_evidence:
            print(f"No evidence retrieved in Attempt {attempt}.")
            if not final_evidence_list:
                print("Aborting: Pipeline requires at least some evidence to proceed.")
                return {}
            # Keep previous attempt's evidence if this attempt was empty
            current_attempt_evidence = final_evidence_list

        final_evidence_list = current_attempt_evidence

        # 2. Run Reflection
        print(f"\n[Attempt {attempt}] Running Reflection Agent...")
        last_reflection_result = reflection_agent.reflect(
            claim_text=claim,
            domain=domain,
            evidence_list=final_evidence_list
        )

        print("\nReflection Evaluation:")
        print(f"  - Decision: {last_reflection_result.get('decision')}")
        print(f"  - Is Evidence Sufficient: {last_reflection_result.get('is_evidence_sufficient')}")
        print(f"  - Confidence: {last_reflection_result.get('confidence')}")
        print(f"  - Source Credibility Score: {last_reflection_result.get('source_credibility_score')}")
        print(f"  - Evidence Quality Score: {last_reflection_result.get('evidence_quality_score')}")
        print(f"  - Claim Supported: {last_reflection_result.get('claim_supported')}")
        print(f"  - Claim Contradicted: {last_reflection_result.get('claim_contradicted')}")
        print(f"  - Evidence Items Contradict Each Other: {last_reflection_result.get('evidence_items_contradict_each_other')}")
        print(f"  - Contradictions Found: {last_reflection_result.get('contradictions_found')}")
        print(f"  - Notes: {last_reflection_result.get('notes')}")

        conf = last_reflection_result.get("confidence", 0.0)
        decision = last_reflection_result.get("decision", "NO")

        # 3. Decision Logic for Retry vs Supervisor
        if decision == "YES" and conf >= HIGH_CONFIDENCE_THRESHOLD:
            print(f"\n>>> [Attempt {attempt}] Reflection returned YES with high confidence ({conf:.2f}).")
            print(">>> Evidence is sufficient and reliable. Proceeding directly to Supervisor Agent.")
            break
        else:
            if attempt < MAX_ATTEMPTS:
                print(f"\n>>> [Attempt {attempt}] Reflection confidence is LOW / insufficient ({conf:.2f}, decision={decision}).")
                print(f">>> Triggering Evidence Retrieval again for Attempt {attempt + 1}...")
            else:
                print(f"\n>>> [Attempt 3] Maximum attempts reached (3/3). STOP.")
                print(">>> There will NEVER be an Attempt 4.")
                print(">>> Forwarding final available evidence and Reflection result to Supervisor Agent.")
                maximum_attempts_reached = True

    # 4. Supervisor Agent
    print("\n" + "=" * 70)
    print("                 RUNNING SUPERVISOR AGENT")
    print("=" * 70)

    supervisor_input = {
        "claim_text": claim,
        "domain": domain,
        "evidence_list": final_evidence_list,
        "reflection_notes": last_reflection_result,
        "attempt_count": completed_attempt,
        "maximum_attempts_reached": maximum_attempts_reached,
        "original_language": "en",
        "was_translated": False
    }

    try:
        supervisor_result = run_supervisor(supervisor_input)
    except Exception as e:
        print(f"\nSupervisor Agent failed: {e}")
        return {}

    print("\nSupervisor Result:")
    print(f"  - Final Verdict: {supervisor_result.get('final_verdict')}")
    print(f"  - Confidence Score: {supervisor_result.get('confidence_score')}")
    print(f"  - Trust Score: {supervisor_result.get('trust_score')}")
    print(f"  - Reasoning Summary: {supervisor_result.get('reasoning_summary')}")
    print(f"  - Key Evidence Used: {supervisor_result.get('key_evidence_used')}")

    # 5. XAI Agent
    print("\n" + "=" * 70)
    print("                    RUNNING XAI AGENT")
    print("=" * 70)

    try:
        xai_result = xai_agent.explain(
            claim_text=claim,
            domain=domain,
            evidence_list=final_evidence_list,
            reflection_result=last_reflection_result,
            supervisor_result=supervisor_result,
            attempt_count=completed_attempt,
            maximum_attempts_reached=maximum_attempts_reached
        )
    except Exception as e:
        print(f"\nXAI Agent failed: {e}")
        return {}

    print("\nXAI Explanation:")
    print(f"  - Final Verdict: {xai_result.get('final_verdict')}")
    print(f"  - Explanation: {xai_result.get('explanation')}")
    print("  - Evidence Summary:")
    for point in xai_result.get("evidence_summary", []):
        print(f"      * {point}")
    print(f"  - Source Analysis: {xai_result.get('source_analysis')}")
    print(f"  - Reasoning: {xai_result.get('reasoning')}")
    print(f"  - XAI Grounded Confidence: {xai_result.get('confidence')}")

    # 6. Milestone Pipeline Report
    print("\n" + "=" * 70)
    print("FULL PIPELINE TEST COMPLETED")
    print("=" * 70)
    print(f"Claim: {claim}")
    print(f"Attempt count: {completed_attempt}")
    print(f"Maximum attempts reached: {maximum_attempts_reached}")
    print(f"Reflection result: {last_reflection_result.get('decision')} (Confidence: {last_reflection_result.get('confidence')})")
    print(f"Supervisor verdict: {supervisor_result.get('final_verdict')}")
    print(f"Supervisor confidence: {supervisor_result.get('confidence_score')}")
    print(f"Supervisor trust score: {supervisor_result.get('trust_score')}")
    print(f"Supervisor key evidence: {supervisor_result.get('key_evidence_used')}")
    print(f"XAI verdict: {xai_result.get('final_verdict')}")
    print(f"XAI confidence: {xai_result.get('confidence')}")
    print("=" * 70)

    return {
        "claim": claim,
        "attempt_count": completed_attempt,
        "maximum_attempts_reached": maximum_attempts_reached,
        "reflection_result": last_reflection_result,
        "supervisor_result": supervisor_result,
        "xai_result": xai_result
    }


# ======================================================================
# PRE-BUILT VERIFICATION TEST CASES
# ======================================================================

def run_test_case_1():
    """TEST CASE 1: STRONG AUTHORITATIVE EVIDENCE"""
    print("\n==================== TEST CASE 1: STRONG AUTHORITATIVE EVIDENCE ====================")
    claim = "The Reserve Bank of India uses the policy repo rate as an important instrument of monetary policy."
    domain = "finance"
    evidence_1 = [
        create_test_evidence(
            evidence_text="The Reserve Bank of India uses the policy repo rate as an important instrument of monetary policy.",
            domain=domain,
            count=1,
            source="Reserve Bank of India",
            authority_score=0.98,
            similarity_score=0.94,
            url="https://www.rbi.org.in/",
            title="RBI Monetary Policy"
        )
    ]
    return execute_pipeline(claim, domain, attempts_data=[evidence_1], interactive=False)


def run_test_case_2():
    """TEST CASE 2: HIGH SIMILARITY BUT VERY LOW SOURCE AUTHORITY"""
    print("\n==================== TEST CASE 2: HIGH SIMILARITY BUT LOW SOURCE AUTHORITY ====================")
    claim = "The Reserve Bank of India uses the policy repo rate as an important instrument of monetary policy."
    domain = "finance"
    evidence_blog = [
        create_test_evidence(
            evidence_text="The Reserve Bank of India uses the policy repo rate as an important instrument of monetary policy.",
            domain=domain,
            count=1,
            source="Unknown Finance Blog",
            authority_score=0.10,
            similarity_score=0.95,
            url="https://random-finance-news-123.example/rbi-policy",
            title="Blog Speculation"
        )
    ]
    # In test case 2 with low authority, Reflection recognizes weak source; if it was Attempt 1, retry can occur or Supervisor assesses low authority
    return execute_pipeline(claim, domain, attempts_data=[evidence_blog], interactive=False)


def run_test_case_3():
    """TEST CASE 3: LOW CONFIDENCE FOR THREE ATTEMPTS"""
    print("\n==================== TEST CASE 3: 3-ATTEMPT RETRY LIMIT TEST ====================")
    claim = "The Reserve Bank of India uses the policy repo rate as an important instrument of monetary policy."
    domain = "finance"

    # Attempt 1: Weak blog stating opposite
    attempt_1 = [
        create_test_evidence(
            evidence_text="A financial blog claims that the Reserve Bank of India does not use the policy repo rate for monetary policy.",
            domain=domain,
            count=1,
            source="Unknown Finance Blog",
            authority_score=0.10,
            similarity_score=0.80,
            url="https://random-finance-news-123.example/rbi-blog1"
        )
    ]

    # Attempt 2: Another weak blog claiming unrelated
    attempt_2 = [
        create_test_evidence(
            evidence_text="The policy repo rate is unrelated to the monetary policy framework of the Reserve Bank of India.",
            domain=domain,
            count=2,
            source="Unknown Finance Blog",
            authority_score=0.15,
            similarity_score=0.85,
            url="https://random-finance-news-123.example/rbi-blog2"
        )
    ]

    # Attempt 3: 3 authoritative sources
    attempt_3 = [
        create_test_evidence(
            evidence_text="The Reserve Bank of India conducts monetary policy to maintain price stability while supporting economic growth.",
            domain=domain,
            count=3,
            source="Reserve Bank of India",
            authority_score=0.95,
            similarity_score=0.91,
            url="https://www.rbi.org.in/"
        ),
        create_test_evidence(
            evidence_text="The Reserve Bank of India uses various monetary-policy instruments to influence financial conditions.",
            domain=domain,
            count=4,
            source="RBI Official Publication",
            authority_score=0.95,
            similarity_score=0.90,
            url="https://www.rbi.org.in/publications"
        ),
        create_test_evidence(
            evidence_text="The policy repo rate is an important instrument in the monetary-policy framework of the Reserve Bank of India.",
            domain=domain,
            count=5,
            source="Government Publication",
            authority_score=0.90,
            similarity_score=0.89,
            url="https://gov.in/finance"
        )
    ]

    return execute_pipeline(claim, domain, attempts_data=[attempt_1, attempt_2, attempt_3], interactive=False)


def run_contradiction_test():
    """ADDITIONAL CONTRADICTION TEST: Claim vs Evidence vs Evidence Contradiction"""
    print("\n==================== ADDITIONAL CONTRADICTION TEST ====================")
    claim = "The Reserve Bank of India uses the policy repo rate as an important instrument of monetary policy."
    domain = "finance"
    evidence_contradiction = [
        create_test_evidence(
            evidence_text="The Reserve Bank of India uses the policy repo rate as an important instrument of monetary policy.",
            domain=domain,
            count=1,
            source="Reserve Bank of India",
            authority_score=0.98,
            similarity_score=0.95,
            url="https://www.rbi.org.in/"
        ),
        create_test_evidence(
            evidence_text="The policy repo rate is unrelated to the monetary policy framework of the Reserve Bank of India.",
            domain=domain,
            count=2,
            source="Unknown Finance Blog",
            authority_score=0.10,
            similarity_score=0.82,
            url="https://random-finance-news-123.example/contra"
        )
    ]
    return execute_pipeline(claim, domain, attempts_data=[evidence_contradiction], interactive=False)


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Fake News Verification Pipeline")
    parser.add_argument("--test-case", type=str, choices=["1", "2", "3", "contradiction"], help="Run a specific test case")
    parser.add_argument("--run-all-tests", action="store_true", help="Run all 4 verification test cases sequentially")
    args = parser.parse_args()

    if args.run_all_tests:
        print("\nRunning all verification test cases...")
        run_test_case_1()
        run_test_case_2()
        run_test_case_3()
        run_contradiction_test()
        return

    if args.test_case == "1":
        run_test_case_1()
        return
    elif args.test_case == "2":
        run_test_case_2()
        return
    elif args.test_case == "3":
        run_test_case_3()
        return
    elif args.test_case == "contradiction":
        run_contradiction_test()
        return

    # Interactive manual entry if no arguments
    print("=" * 70)
    print("       REFLECTION → SUPERVISOR → XAI INTEGRATION")
    print("=" * 70)
    claim = input("\nEnter claim: ").strip()
    if not claim:
        print("No claim entered. Exiting.")
        return
    domain = input("Enter domain: ").strip() or "general"

    execute_pipeline(claim, domain, interactive=True)


if __name__ == "__main__":
    main()