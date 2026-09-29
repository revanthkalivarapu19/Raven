from agents.nlp_pipeline.nlp_pipeline import run_nlp_pipeline
from agents.claim_extraction.claim_extraction_agent import ClaimExtractionAgent
from agents.domain_detection.domain_agent import DomainDetectionAgent


def main():

    claim_agent = ClaimExtractionAgent()
    domain_agent = DomainDetectionAgent()

    # -----------------------------
    # 1. USER INPUT
    # -----------------------------

    user_input = input("\nEnter news/text: ")

    # -----------------------------
    # 2. NLP PIPELINE
    # -----------------------------

    nlp_result = run_nlp_pipeline(user_input)

    print("\n========== NLP PIPELINE RESULT ==========")
    print("Text:", nlp_result["text"])
    print("Tokens:", nlp_result["tokens"])
    print("Sentences:", nlp_result["sentences"])
    print("POS Tags:", nlp_result["pos_tags"])
    print("Entities:", nlp_result["entities"])
    print("Dependencies:", nlp_result["dependencies"])
    print("Keywords:", nlp_result["keywords"])

    # -----------------------------
    # 3. CLAIM EXTRACTION
    # -----------------------------

    extracted_claim = claim_agent.extract_claim(
        nlp_result["text"]
    )

    print("\n========== CLAIM EXTRACTION ==========")
    print("Extracted Claim:", extracted_claim)

    if not extracted_claim:
        print("No claim could be extracted.")
        return

    # -----------------------------
    # 4. DOMAIN DETECTION
    # -----------------------------

    domain_result = domain_agent.detect_domain(
        extracted_claim
    )

    print("\n========== DOMAIN DETECTION ==========")
    print("Domain     :", domain_result["domain"])
    print("Confidence :", domain_result["confidence"])
    print("Reason     :", domain_result["reason"])


if __name__ == "__main__":
    main()