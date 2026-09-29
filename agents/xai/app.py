from agents.xai.xai_agent import XAIAgent


def main():

    print("=" * 70)
    print("                    XAI AGENT TEST")
    print("=" * 70)

    claim = (
        "The Reserve Bank of India uses the policy repo rate "
        "as an important instrument of monetary policy."
    )

    domain = "finance"

    evidence_list = [

        {
            "evidence_id": "evidence_1",
            "chunk_id": "chunk_1",
            "document_id": "document_1",
            "text": (
                "The Reserve Bank of India uses the policy repo "
                "rate as an important instrument of monetary policy."
            ),
            "domain": "finance",
            "source": "Reserve Bank of India",
            "language": "en",
            "similarity_score": 0.94,
            "authority_score": 0.98,
            "retrieval_method": "faiss_local",
            "title": "RBI Monetary Policy",
            "url": "https://www.rbi.org.in/"
        },

        {
            "evidence_id": "evidence_2",
            "chunk_id": "chunk_2",
            "document_id": "document_2",
            "text": (
                "Changes in the policy repo rate influence "
                "borrowing costs and financial conditions."
            ),
            "domain": "finance",
            "source": "Reserve Bank of India",
            "language": "en",
            "similarity_score": 0.89,
            "authority_score": 0.98,
            "retrieval_method": "faiss_local",
            "title": "RBI Policy Framework",
            "url": "https://www.rbi.org.in/"
        }
    ]

    reflection_result = {
        "decision": "YES",
        "is_evidence_sufficient": True,
        "contradictions_found": False,
        "confidence": 0.95,
        "notes": (
            "The evidence directly supports the claim and "
            "comes from a highly authoritative source."
        )
    }

    supervisor_result = {
        "final_verdict": "Real",
        "confidence_score": 95,
        "trust_score": 90,
        "reasoning_summary": (
            "The claim is supported by multiple authoritative "
            "RBI sources."
        ),
        "key_evidence_used": [
            "Reserve Bank of India (evidence_1)",
            "Reserve Bank of India (evidence_2)"
        ]
    }

    print("\nRunning XAI Agent...")

    xai_agent = XAIAgent()

    result = xai_agent.explain(
        claim_text=claim,
        domain=domain,
        evidence_list=evidence_list,
        reflection_result=reflection_result,
        supervisor_result=supervisor_result
    )

    print("\n" + "=" * 70)
    print("                    XAI RESULT")
    print("=" * 70)

    print("\nFinal Verdict:", result["final_verdict"])

    print("\nExplanation:")
    print(result["explanation"])

    print("\nEvidence Summary:")

    for item in result["evidence_summary"]:
        print("-", item)

    print("\nSource Analysis:")
    print(result["source_analysis"])

    print("\nReasoning:")
    print(result["reasoning"])

    print("\nXAI Confidence:")
    print(result["confidence"])


if __name__ == "__main__":
    main()