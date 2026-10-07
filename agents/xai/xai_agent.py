import ollama

from agents.xai.prompt_builder import build_xai_prompt
from agents.xai.output_parser import parse_xai_output


class XAIAgent:

    def __init__(self, model="llama3.1:8b"):
        self.model = model

    def explain(
        self,
        claim_text,
        domain,
        evidence_list,
        reflection_result,
        supervisor_result,
        attempt_count=1,
        maximum_attempts_reached=False
    ):
        prompt = build_xai_prompt(
            claim_text=claim_text,
            domain=domain,
            evidence_list=evidence_list,
            reflection_result=reflection_result,
            supervisor_result=supervisor_result,
            attempt_count=attempt_count,
            maximum_attempts_reached=maximum_attempts_reached
        )

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an Explainable AI agent for a "
                        "fact-checking system. Explain the final "
                        "Supervisor result clearly and factually. "
                        "Do not change the Supervisor verdict."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0.1
            }
        )

        raw_output = response["message"]["content"]

        return parse_xai_output(
            raw_output,
            supervisor_result=supervisor_result,
            evidence_list=evidence_list
        )