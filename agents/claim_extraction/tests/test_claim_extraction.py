from agents.claim_extraction_agent import ClaimExtractionAgent


def test_rbi_repeated_claim():

    agent = ClaimExtractionAgent()

    text = """
    The RBI raised the repo rate to 6.5 percent. The central bank
    increased interest rates to 6.5 percent. India's repo rate now
    stands at 6.5 percent. The decision was taken as part of the
    central bank's monetary policy measures to control inflation
    and maintain economic stability.
    """

    claim = agent.extract_claim(text)

    print("\nInput:")
    print(text)

    print("\nExtracted Claim:")
    print(claim)

    assert claim != ""
    assert len(claim) < len(text) / 2


def test_long_paragraph():

    agent = ClaimExtractionAgent()

    text = """
    The Reserve Bank of India announced its latest monetary policy
    decision after reviewing several economic indicators. The
    central bank decided to maintain the repo rate at 6.5 percent.
    Officials considered inflation trends, economic growth,
    liquidity conditions, global economic developments and other
    financial indicators before making the decision. The RBI also
    stated that it would continue monitoring economic conditions
    before making future changes to interest rates.
    """

    claim = agent.extract_claim(text)

    print("\nExtracted Claim:")
    print(claim)

    assert claim != ""
    assert len(claim) < len(text) / 2


def test_nasa():

    agent = ClaimExtractionAgent()

    text = """
    NASA successfully launched a new spacecraft from Florida on
    Tuesday. The spacecraft is part of a scientific mission that
    will study a distant planet. Scientists expect the mission to
    collect important scientific data over several years. NASA
    engineers monitored the launch and confirmed that the spacecraft
    entered its planned trajectory.
    """

    claim = agent.extract_claim(text)

    print("\nExtracted Claim:")
    print(claim)

    assert claim != ""
    assert "NASA" in claim


def test_government():

    agent = ClaimExtractionAgent()

    text = """
    The government announced a new renewable energy policy.
    The policy is intended to increase renewable energy production.
    Officials said the policy will support clean energy development
    and reduce dependence on conventional energy sources. The
    government expects the policy to encourage additional investment
    in renewable energy projects.
    """

    claim = agent.extract_claim(text)

    print("\nExtracted Claim:")
    print(claim)

    assert claim != ""
    assert len(claim) < len(text) / 2


def test_empty_input():

    agent = ClaimExtractionAgent()

    claim = agent.extract_claim("")

    assert claim == ""