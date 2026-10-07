import hashlib
import logging
import os
from typing import List, Optional, Dict, Any
from datetime import datetime
import requests

from .base_source import BaseExternalSource
from .source_schema import ExternalEvidence

logger = logging.getLogger(__name__)

class FactCheckSource(BaseExternalSource):
    """Concrete external source for the Google Fact Check Tools API."""

    def __init__(self, timeout: int = 15) -> None:
        self.name = "fact_check"
        self.timeout = timeout
        self.base_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

    def search(
        self,
        query: str,
        domain: Optional[str] = None,
        top_k: int = 5,
        language: Optional[str] = None,
        country: Optional[str] = None,
    ) -> List[ExternalEvidence]:
        if not query or not query.strip():
            raise ValueError("Query must be non-empty")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        api_key = os.environ.get("GOOGLE_FACTCHECK_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_FACTCHECK_API_KEY is not configured.")

        params = {
            "query": query,
            "key": api_key,
            "pageSize": top_k
        }
        if language and language.strip():
            params["languageCode"] = language.strip()

        try:
            response = requests.get(self.base_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.Timeout as e:
            # Do not include the requests exception text: it may contain the
            # fully rendered URL, including the API key query parameter.
            raise RuntimeError("Fact Check API request timed out") from e
        except requests.exceptions.HTTPError as e:
            if response.status_code in (401, 403):
                raise RuntimeError(f"Fact Check API authentication error: {response.status_code}") from e
            elif response.status_code == 429:
                raise RuntimeError("Fact Check API rate limit exceeded") from e
            raise RuntimeError(f"Fact Check API request failed with HTTP {response.status_code}") from e
        except requests.exceptions.RequestException as e:
            # RequestException strings can also include the authenticated URL.
            raise RuntimeError("Fact Check API request failed") from e
        except ValueError as e:
            raise RuntimeError(f"Invalid JSON response from Fact Check API: {e}") from e

        if not isinstance(data, dict):
            raise RuntimeError("Unexpected response structure: JSON is not a dictionary")

        claims = data.get("claims", [])
        if not isinstance(claims, list):
            raise RuntimeError("Unexpected response structure: 'claims' is not a list")

        results = []
        for idx, claim in enumerate(claims[:top_k]):
            claim_text = claim.get("text", "")
            claimant = claim.get("claimant", "")
            claim_date = claim.get("claimDate")

            reviews = claim.get("claimReview", [])
            if not reviews:
                continue

            review = reviews[0]
            publisher_dict = review.get("publisher", {})
            publisher = publisher_dict.get("name", "")
            publisher_site = publisher_dict.get("site", "")

            review_url = review.get("url", "")
            review_title = review.get("title", "")
            review_date = review.get("reviewDate")
            rating_text = review.get("textualRating", "")

            if review_url:
                evidence_id = f"factcheck_{hashlib.sha256(review_url.encode('utf-8')).hexdigest()}"
            else:
                fallback = f"{claim_text}_{publisher}_{idx}"
                evidence_id = f"factcheck_{hashlib.sha256(fallback.encode('utf-8')).hexdigest()}"

            text_parts = []
            if claim_text:
                text_parts.append(f"Claim: {claim_text}")
            if review_title:
                text_parts.append(f"Review: {review_title}")

            text = " | ".join(text_parts) if text_parts else "No textual summary available."

            pub_date = None
            if review_date:
                try:
                    # Example format: "2023-10-24T12:00:00Z"
                    pub_date = datetime.fromisoformat(review_date.replace("Z", "+00:00")).replace(tzinfo=None)
                except ValueError:
                    pass

            metadata: Dict[str, Any] = {
                "fact_check_rating": rating_text,
                "text_limitation": "Extracted from structured fact-check metadata."
            }
            if claimant:
                metadata["claimant"] = claimant
            if claim_date:
                metadata["claim_date"] = claim_date
            if publisher_site:
                metadata["publisher_site"] = publisher_site

            language = review.get("languageCode")
            if language:
                metadata["language"] = language

            ev = ExternalEvidence(
                evidence_id=evidence_id,
                text=text,
                source=publisher if publisher else "Google Fact Check Tools",
                title=review_title if review_title else claim_text,
                url=review_url,
                domain=domain if domain else "general",
                retrieved_date=datetime.utcnow(),
                metadata=metadata,
                publication_date=pub_date,
                language=language,
                source_type="fact_check_api",
                authority_score=None
            )
            results.append(ev)

        return results
    supported_domains = frozenset({"medical", "political", "science", "finance", "general"})
