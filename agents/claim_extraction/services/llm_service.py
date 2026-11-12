import logging
import time

import httpx
import ollama
from rag_ingestion.llm_resilience import LLMInvocationError, remaining_request_budget


logger = logging.getLogger(__name__)
CLAIM_MODEL_REQUEST_TIMEOUT_SECONDS = 20.0


class LLMService:

    def __init__(self):
        self.model = "qwen2.5:3b"

    def generate(self, prompt):
        remaining = remaining_request_budget()
        timeout_seconds = CLAIM_MODEL_REQUEST_TIMEOUT_SECONDS
        if remaining is not None:
            if remaining <= 0:
                raise LLMInvocationError("Claim extraction deadline exceeded", "timeout")
            timeout_seconds = min(timeout_seconds, remaining)

        started = time.monotonic()
        logger.info(
            "claim_extraction model_request_start monotonic=%.6f timeout_seconds=%.2f",
            started, timeout_seconds,
        )
        client = ollama.Client(timeout=timeout_seconds)
        try:
            # Set the timeout on HTTPX itself. Unlike a daemon-thread wait,
            # this bounds the socket operation and does not leave a worker
            # running after the caller returns.
            response = client.chat(
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
- Keep every materially relevant person, organization, event, number, date,
  percentage, unit, and relative time phrase from the factual claim.
- Preserve every explicitly named entity in the claim, including every name in
  a multi-entity list; never replace those names with a generic group label
  such as "major companies".
- When a group is followed by its explicitly named members (including after
  wording such as "including" or punctuation such as dashes), include every
  listed member in the extracted claim. This is mandatory even if the result
  exceeds the usual word target.
- Preserve every explicit temporal qualifier that materially affects the
  claim, including relative expressions (for example, "recently", "yesterday",
  "last week", or "this year") and specific time references (for example,
  "in 2025" or "on Friday"). Do not omit, weaken, or generalize these
  expressions to make the claim shorter.
- Preserve negation (for example, "did not" or "never") exactly when it
  changes the factual proposition.
- Never change a number, percentage, date, year, quantity, named entity, or
  explicit temporal qualifier, and never introduce factual information absent
  from the input.
- Preserve factual polarity and directional/comparative meaning, including
  increase/decrease, rise/fall, before/after, more/less, and whether a quantity
  changed "by" a value or changed "to" a value.
- A concise paraphrase is allowed only when all underlying factual values and
  relations remain unchanged.
- Preserve a causal or contextual qualifier when it changes the event's
  meaning, timing, scope, or verifiability. A qualifier such as "to boost
  economic growth" may be retained as context in the same concise claim.
- Remove background information.
- Remove explanations.
- Remove background and commentary, but do not remove factual qualifiers,
  causal context, or the time period needed to verify the event.
- Remove repeated information.
- Never copy the entire paragraph.
- Never list multiple claims.
- Never explain your reasoning.
- Never say whether the claim is true or false.
- Do not add information that is not present.
- Keep the result concise, normally 10-30 words; use more words when necessary
  to preserve material facts, explicitly named entities, and temporal
  qualifiers.
- The result should normally fit within one or two lines.

Return ONLY the final claim.
"""
                    },
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0},
            )
            content = response["message"]["content"].strip()
            if not content:
                raise LLMInvocationError("Claim extraction returned empty output", "invalid_response")
            return content
        except LLMInvocationError:
            raise
        except (httpx.TimeoutException, TimeoutError) as exc:
            raise LLMInvocationError(
                f"Claim extraction model request timed out ({type(exc).__name__})",
                "timeout",
                retryable=False,
            ) from exc
        except Exception as exc:
            text = str(exc).casefold()
            kind = "timeout" if "timeout" in text or "timed out" in text else "transport"
            raise LLMInvocationError(
                f"Claim extraction model request failed ({type(exc).__name__})",
                kind,
                retryable=False,
            ) from exc
        finally:
            client.close()
            logger.info(
                "claim_extraction model_request_end monotonic=%.6f duration_ms=%.1f "
                "timeout_seconds=%.2f",
                time.monotonic(),
                (time.monotonic() - started) * 1000,
                timeout_seconds,
            )
