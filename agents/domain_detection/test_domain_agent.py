"""
test_domain_agent.py

Tests the Domain Detection Agent using sample claims.
"""

from domain_agent import DomainDetectionAgent
from examples import TEST_CLAIMS


def run_tests():
    agent = DomainDetectionAgent()

    print("=" * 70)
    print("RAVEN - Domain Detection Agent Test")
    print("=" * 70)

    for index, test in enumerate(TEST_CLAIMS, start=1):

        print(f"\nTest Case {index}")

        print(f"Claim            : {test['claim']}")
        print(f"Expected Domain  : {test['expected_domain']}")

        result = agent.detect_domain(test["claim"])

        print(f"Predicted Domain : {result['domain']}")
        print(f"Confidence       : {result['confidence']}")
        print(f"Reason           : {result['reason']}")

        if result["domain"] == test["expected_domain"]:
            print("Status           : PASS ✅")
        else:
            print("Status           : FAIL ❌")


if __name__ == "__main__":
    run_tests()