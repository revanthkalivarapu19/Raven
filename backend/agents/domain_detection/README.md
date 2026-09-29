# Domain Detection Agent

## Overview

The Domain Detection Agent is a specialized AI agent in the RAVEN (Real-Time Evidence Analysis Engine) framework. Its primary responsibility is to classify an extracted news claim into one of the predefined domains.

This agent enables the system to route the claim to the appropriate expert and evidence retrieval process, improving both efficiency and accuracy.

---

## Supported Domains

- Medical
- Politics
- Finance
- Technology

---

## Input

The agent receives a single extracted claim from the Claim Extraction Agent.

Example:

"The RBI has increased the repo rate by 50 basis points."

---

## Output

The agent returns a structured JSON response.

Example:

```json
{
    "domain": "Finance",
    "confidence": 0.97,
    "reason": "The claim discusses RBI and monetary policy."
}
```

---

## Workflow

1. Receive the extracted claim.
2. Load the domain classification prompt.
3. Send the prompt to the LLM.
4. Receive the LLM response.
5. Parse the JSON response.
6. Validate the output.
7. Return the detected domain.

---

## Technologies Used

- Python
- Ollama
- LangGraph
- Pydantic
- Loguru
- Python-dotenv

---

## Project Structure

```
domain_detection/
│
├── __init__.py
├── domain_agent.py
├── prompt.py
├── parser.py
├── validator.py
├── examples.py
└── README.md
```

---

## Future Improvements

- Multi-domain classification
- Confidence threshold tuning
- Domain explanation enhancement
- Support for additional domains

---

## Author

RAVEN Project Team