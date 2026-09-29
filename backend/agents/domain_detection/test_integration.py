from claim_extraction.claim_extraction_agent import ClaimExtractionAgent
from domain_agent import DomainDetectionAgent


def main():

    # Create both agents
    claim_agent = ClaimExtractionAgent()
    domain_agent = DomainDetectionAgent()

    # User input
    text = input("\nEnter news/text: ")

    # Step 1: Extract main claim
    extracted_claim = claim_agent.extract_claim(text)

    print("\nExtracted Claim:")
    print(extracted_claim)

    if not extracted_claim:
        print("\nNo claim could be extracted.")
        return

    # Step 2: Detect domain
    domain_result = domain_agent.detect_domain(extracted_claim)

    print("\nDomain Detection Result:")
    print("Domain     :", domain_result["domain"])
    print("Confidence :", domain_result["confidence"])
    print("Reason     :", domain_result["reason"])


if __name__ == "__main__":
    main()