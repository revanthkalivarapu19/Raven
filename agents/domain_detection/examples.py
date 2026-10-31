"""
examples.py

Sample claims for testing the Domain Detection Agent.
"""

TEST_CLAIMS = [

    # Medical
    {
        "claim": "WHO approves a new malaria vaccine.",
        "expected_domain": "Medical"
    },
    {
        "claim": "A new COVID-19 variant has been detected in India.",
        "expected_domain": "Medical"
    },
    {
        "claim": "Doctors recommend annual flu vaccination.",
        "expected_domain": "Medical"
    },

    # Politics
    {
        "claim": "The Election Commission announced polling dates.",
        "expected_domain": "Politics"
    },
    {
        "claim": "Parliament passed the Digital India Bill.",
        "expected_domain": "Politics"
    },
    {
        "claim": "The Prime Minister addressed the nation on economic reforms.",
        "expected_domain": "Politics"
    },

    # Finance
    {
        "claim": "RBI increased the repo rate by 25 basis points.",
        "expected_domain": "Finance"
    },
    {
        "claim": "SEBI introduced new stock market regulations.",
        "expected_domain": "Finance"
    },
    {
        "claim": "Bitcoin crossed ₹80 lakh in trading today.",
        "expected_domain": "Finance"
    },

    # Science
    {
        "claim": "Researchers observed a new exoplanet using a space telescope.",
        "expected_domain": "Science"
    },
    {
        "claim": "Scientists reproduced the experiment in an independent laboratory.",
        "expected_domain": "Science"
    },
    {
        "claim": "A new study describes a previously unknown species of bacteria.",
        "expected_domain": "Science"
    }

]