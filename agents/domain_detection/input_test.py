from agents.domain_detection.domain_agent import DomainDetectionAgent


if __name__ == "__main__":
    agent = DomainDetectionAgent()
    claim = input("\nEnter your claim: ")
    result = agent.detect_domain(claim)
    print("\nResult:")
    print("Domain     :", result["domain"])
    print("Confidence :", result["confidence"])
    print("Reason     :", result["reason"])
