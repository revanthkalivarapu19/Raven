# src/external_retrieval/world_bank_source.py
"""External evidence provider for the World Bank Indicators API.

The World Bank API is a structured indicator source, not a general
natural-language claim-verification service.  This provider therefore resolves
only explicitly supported indicator concepts and requires a resolvable country.
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Tuple
from urllib.parse import quote

import requests

from .base_source import BaseExternalSource
from .source_schema import ExternalEvidence

logger = logging.getLogger(__name__)

# Strong authority for an official statistical data source, on the project's
# existing 0-10 scale. This reflects source quality, not claim truth.
WORLD_BANK_AUTHORITY_SCORE = 8.5

DEFAULT_INDICATOR_MAPPING: Dict[str, Tuple[str, str]] = {
    "gdp": ("NY.GDP.MKTP.CD", "GDP (current US$)"),
    "gross domestic product": ("NY.GDP.MKTP.CD", "GDP (current US$)"),
    "gdp growth": ("NY.GDP.MKTP.KD.ZG", "GDP growth (annual %)"),
    "economic growth": ("NY.GDP.MKTP.KD.ZG", "GDP growth (annual %)"),
    "gdp per capita": ("NY.GDP.PCAP.CD", "GDP per capita (current US$)"),
    "inflation": ("FP.CPI.TOTL.ZG", "Inflation, consumer prices (annual %)"),
    "unemployment": ("SL.UEM.TOTL.ZS", "Unemployment, total (% of total labor force)"),
}

COUNTRY_CODES: Dict[str, str] = {
    "united states": "USA",
    "united states of america": "USA",
    "usa": "USA",
    "us": "USA",
    "india": "IND",
    "united kingdom": "GBR",
    "uk": "GBR",
    "china": "CHN",
    "japan": "JPN",
    "germany": "DEU",
    "france": "FRA",
    "canada": "CAN",
    "australia": "AUS",
    "brazil": "BRA",
    "mexico": "MEX",
    "south africa": "ZAF",
    "nigeria": "NGA",
    "kenya": "KEN",
}


class WorldBankSource(BaseExternalSource):
    """Concrete source for bounded World Bank indicator observations."""

    def __init__(
        self,
        timeout: int = 30,
        indicator_mapping: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.name = "world_bank"
        self.timeout = timeout
        self.base_url = "https://api.worldbank.org/v2"
        self.indicator_mapping: Dict[str, Tuple[str, str]] = dict(DEFAULT_INDICATOR_MAPPING)
        if indicator_mapping:
            for concept, value in indicator_mapping.items():
                if isinstance(value, str):
                    self.indicator_mapping[concept.lower()] = (value, concept)
                elif isinstance(value, (tuple, list)) and value and isinstance(value[0], str):
                    label = value[1] if len(value) > 1 and isinstance(value[1], str) else concept
                    self.indicator_mapping[concept.lower()] = (value[0], label)

    @staticmethod
    def _clean(value: Any) -> str:
        return value.strip() if isinstance(value, str) else ""

    def _resolve_indicator(self, claim: str) -> Optional[Tuple[str, str]]:
        normalized = re.sub(r"[^a-z0-9]+", " ", claim.lower()).strip()
        matches = [
            (concept, mapping)
            for concept, mapping in self.indicator_mapping.items()
            if re.search(rf"\b{re.escape(concept)}\b", normalized)
        ]
        if not matches:
            return None
        # Prefer the most specific concept (e.g. GDP per capita over GDP).
        return max(matches, key=lambda item: len(item[0]))[1]

    @staticmethod
    def _resolve_country(claim: str, country: Optional[str]) -> Optional[Tuple[str, str]]:
        if country:
            candidate = country.strip()
            if re.fullmatch(r"[A-Za-z]{2,3}", candidate):
                return candidate.upper(), candidate.upper()
            mapped = COUNTRY_CODES.get(candidate.lower())
            if mapped:
                return mapped, candidate
            return None

        normalized = re.sub(r"[^a-z0-9]+", " ", claim.lower()).strip()
        matches = [
            (name, code)
            for name, code in COUNTRY_CODES.items()
            if re.search(rf"\b{re.escape(name)}\b", normalized)
        ]
        if not matches:
            return None
        name, code = max(matches, key=lambda item: len(item[0]))
        return code, name

    @staticmethod
    def _safe_component(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "unknown"

    @staticmethod
    def _observation_date(value: Any) -> Optional[datetime]:
        text = str(value).strip() if value is not None else ""
        match = re.fullmatch(r"(\d{4})(?:-(\d{1,2})(?:-(\d{1,2}))?)?", text)
        if not match:
            return None
        try:
            return datetime(int(match.group(1)), int(match.group(2) or 1), int(match.group(3) or 1))
        except ValueError:
            return None

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

        indicator = self._resolve_indicator(query)
        resolved_country = self._resolve_country(query, country)
        # No reliable indicator or country means no API request and no guess.
        if indicator is None or resolved_country is None:
            return []

        indicator_code, fallback_name = indicator
        country_code, country_name_from_query = resolved_country
        endpoint = (
            f"{self.base_url}/country/{quote(country_code, safe='')}/indicator/"
            f"{quote(indicator_code, safe='.') }"
        )
        params = {"format": "json", "per_page": top_k}
        try:
            response = requests.get(
                endpoint,
                params=params,
                headers={"User-Agent": "MERIDIAN/1.0 (evidence retrieval)"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(f"World Bank API request timed out: {exc}") from exc
        except requests.exceptions.HTTPError as exc:
            status_code = getattr(response, "status_code", None)
            if status_code == 429:
                raise RuntimeError("World Bank API rate limit exceeded") from exc
            raise RuntimeError(f"World Bank API request failed: {exc}") from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"World Bank API request failed: {exc}") from exc
        except ValueError as exc:
            raise RuntimeError(f"Invalid JSON response from World Bank: {exc}") from exc

        if not isinstance(data, list) or len(data) < 2:
            return []
        metadata = data[0] if isinstance(data[0], dict) else {}
        observations = data[1]
        if not isinstance(observations, list):
            raise RuntimeError("Unexpected response structure: World Bank observations are not a list")

        results: List[ExternalEvidence] = []
        retrieved_date = datetime.utcnow()
        for index, observation in enumerate(observations[:top_k]):
            if not isinstance(observation, dict) or observation.get("value") is None:
                continue
            date_text = self._clean(observation.get("date"))
            observation_date = self._observation_date(date_text)
            if not date_text:
                continue
            country_info = observation.get("country")
            country_name = (
                self._clean(country_info.get("value"))
                if isinstance(country_info, dict)
                else country_name_from_query
            ) or country_code
            indicator_info = observation.get("indicator")
            indicator_name = (
                self._clean(indicator_info.get("value"))
                if isinstance(indicator_info, dict)
                else fallback_name
            ) or fallback_name
            value = observation.get("value")
            unit = self._clean(observation.get("unit"))
            text = (
                f"World Bank indicator: {indicator_name}. Country: {country_name}. "
                f"Year: {date_text}. Value: {value}"
                f"{f' {unit}' if unit else ''}."
            )
            observation_id = "_".join(
                self._safe_component(part)
                for part in (country_code, indicator_code, date_text)
            )
            api_url = self._clean(observation.get("indicator", {}).get("id")) if isinstance(observation.get("indicator"), dict) else ""
            api_url = api_url or endpoint
            evidence_metadata: Dict[str, Any] = {
                "provider": "World Bank",
                "indicator_code": indicator_code,
                "indicator_name": indicator_name,
                "country_code": country_code,
                "country_name": country_name,
                "observation_date": date_text,
                "value": value,
                "unit": unit,
                "source_note": self._clean(observation.get("note")),
                "source_organization": "World Bank",
                "api_url": api_url,
                "text_limitation": "World Bank indicator observations are structured data and do not independently verify a natural-language claim.",
            }
            for key in ("decimal", "obs_status", "status", "capitalCity", "region", "incomeLevel", "lendingType"):
                if observation.get(key) not in (None, "", {}):
                    evidence_metadata[key] = observation[key]
            if metadata:
                evidence_metadata["api_metadata"] = metadata
            results.append(
                ExternalEvidence(
                    evidence_id=f"worldbank_{observation_id}",
                    text=text,
                    source="World Bank",
                    title=indicator_name,
                    url=endpoint,
                    domain="finance",
                    publication_date=observation_date,
                    retrieved_date=retrieved_date,
                    language=language,
                    source_type="economic_indicator",
                    authority_score=WORLD_BANK_AUTHORITY_SCORE,
                    metadata=evidence_metadata,
                )
            )
        return results
    supported_domains = frozenset({"political", "finance"})
