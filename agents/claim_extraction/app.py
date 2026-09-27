from agents.claim_extraction_agent import ClaimExtractionAgent


def main():

    print("=" * 50)
    print("       CLAIM EXTRACTION AGENT")
    print("=" * 50)

    agent = ClaimExtractionAgent()

    while True:

        print("\nEnter your text:")
        text = input("> ")

        if text.lower() in ["exit", "quit"]:
            print("\nExiting...")
            break

        if not text.strip():
            print("\nPlease enter some text.")
            continue

        print("\nExtracting claim...\n")

        claim = agent.extract_claim(text)

        print("Claim:")
        print(claim)


if __name__ == "__main__":
    main()