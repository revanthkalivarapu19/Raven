from domain_agent import DomainDetectionAgent


agent = DomainDetectionAgent()

claim = input("\nEnter your claim: ")

result = agent.detect_domain(claim)

print("\nResult:")
print("Domain     :", result["domain"])
print("Confidence :", result["confidence"])
print("Reason     :", result["reason"])