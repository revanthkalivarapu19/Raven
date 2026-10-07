import ollama


class LLMService:

    def __init__(self):
        self.model = "llama3.1:8b"

    def generate(self, prompt):

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": """
You are a Claim Extraction Agent.

Your job is to read a paragraph and identify its MAIN factual claim.

Rules:
- Return only ONE main claim.
- The claim must be concise.
- The claim must normally be ONE sentence.
- Keep the important person, organization, event, number, date or percentage.
- Remove background information.
- Remove explanations.
- Remove supporting details.
- Remove repeated information.
- Never copy the entire paragraph.
- Never list multiple claims.
- Never explain your reasoning.
- Never say whether the claim is true or false.
- Do not add information that is not present.
- Keep the result to approximately 10-30 words.
- The result should normally fit within one or two lines.

Return ONLY the final claim.
"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0
            }
        )

        return response["message"]["content"].strip()