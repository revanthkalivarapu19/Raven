"""Bounded IPU Parline structured-data provider.

Parline is a structured parliamentary/electoral dataset, not a general claim
search service. Queries must be mappings (or JSON objects) describing a
supported resource and bounded filters.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import urljoin

import requests

from .base_source import BaseExternalSource
from .source_schema import ExternalEvidence

logger = logging.getLogger(__name__)

IPU_PARLINE_AUTHORITY_SCORE = 8.5
SUPPORTED_RESOURCES = {"countries", "parliaments", "chambers", "elections", "explorer"}
MAX_PAGE_SIZE = 20


class IPUParlineSource(BaseExternalSource):
    """Retrieve bounded observations from the official IPU Parline API."""

    def __init__(self, timeout: int = 30, max_page_size: int = MAX_PAGE_SIZE) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_page_size < 1:
            raise ValueError("max_page_size must be positive")
        self.name = "ipu_parline"
        self.timeout = timeout
        self.max_page_size = min(max_page_size, MAX_PAGE_SIZE)
        self.base_url = "https://api.data.ipu.org/v1"

    @staticmethod
    def _structured_query(query: Any) -> Optional[Dict[str, Any]]:
        if isinstance(query, Mapping):
            return dict(query)
        if not isinstance(query, str) or not query.strip():
            return None
        try:
            parsed = json.loads(query)
        except (TypeError, ValueError):
            return None
        return dict(parsed) if isinstance(parsed, Mapping) else None

    @staticmethod
    def _parse_date(value: Any) -> Optional[datetime]:
        if not isinstance(value, str) or not value.strip():
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            return None

    @staticmethod
    def _unwrap(value: Any, language: str = "en") -> Any:
        if isinstance(value, dict) and "value" in value:
            value = value["value"]
        if isinstance(value, dict):
            if language in value:
                return value[language]
            if "en" in value:
                return value["en"]
            return "; ".join(f"{key}={IPUParlineSource._unwrap(item, language)}" for key, item in value.items())
        if isinstance(value, list):
            return ", ".join(str(IPUParlineSource._unwrap(item, language)) for item in value)
        return value

    @classmethod
    def _record_attributes(cls, record: Any) -> Dict[str, Any]:
        if not isinstance(record, dict):
            return {}
        attrs = record.get("attributes")
        if isinstance(attrs, dict):
            return attrs
        return record

    @classmethod
    def _records(cls, data: Any) -> List[Dict[str, Any]]:
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if isinstance(data, dict):
            if all(str(key).isdigit() for key in data):
                return [item for item in data.values() if isinstance(item, dict)]
            return [data]
        raise RuntimeError("Unexpected IPU Parline response structure: data is not an object or list")

    @classmethod
    def _stable_id(cls, record: Dict[str, Any], attrs: Dict[str, Any], index: int) -> str:
        for field in ("id", "chamber_code", "parliament_code", "election_code", "specialized_body_code", "country_code"):
            value = record.get(field, attrs.get(field))
            value = cls._unwrap(value)
            if value not in (None, ""):
                return f"ipu_parline_{field}_{value}"
        digest = hashlib.sha256(json.dumps(record, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        return f"ipu_parline_observation_{index}_{digest}"

    def _params(self, spec: Dict[str, Any], top_k: int, language: Optional[str]) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        resource = spec["resource"]
        if resource == "explorer":
            fields = spec.get("fields")
            if not isinstance(fields, str) or not fields.strip():
                raise ValueError("IPU explorer queries require a non-empty fields string")
            params["fields"] = ",".join(field.strip() for field in fields.split(",") if field.strip())
            if not params["fields"]:
                raise ValueError("IPU explorer fields cannot be empty")
            for key in ("region", "structure_of_parliament", "struct_parl_status", "subregion", "country_code", "other_groupings", "keyword", "date_range[from]", "date_range[to]", "sort"):
                if spec.get(key) not in (None, ""):
                    params[key] = spec[key]
            if "sort" not in params:
                raise ValueError("IPU explorer queries require sort")
        else:
            filters = spec.get("filters", {})
            if filters and not isinstance(filters, Mapping):
                raise ValueError("IPU filters must be an object")
            if isinstance(filters, Mapping):
                for key, value in filters.items():
                    if isinstance(key, str) and value not in (None, ""):
                        params.setdefault("filter[]", []).append(f"{key}:{value}")
            for key in ("sort",):
                if spec.get(key) not in (None, ""):
                    params[key] = spec[key]
        params["page[size]"] = min(int(spec.get("page_size", top_k)), self.max_page_size)
        params["page[number]"] = max(1, int(spec.get("page", 1)))
        if language:
            params["accept-language"] = language
        return params

    def search(
        self,
        query: Any,
        domain: Optional[str] = None,
        top_k: int = 5,
        language: Optional[str] = None,
        country: Optional[str] = None,
    ) -> List[ExternalEvidence]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        spec = self._structured_query(query)
        if spec is None:
            return []
        resource = spec.get("resource")
        if resource not in SUPPORTED_RESOURCES:
            return []
        if country and resource == "explorer":
            spec.setdefault("country_code", country.upper())
        identifier = spec.get("identifier")
        path = f"/{resource}/"
        if identifier:
            safe_identifier = str(identifier).strip()
            if not safe_identifier.replace("-", "").replace("_", "").isalnum():
                raise ValueError("IPU identifier contains unsupported characters")
            path = f"/{resource}/{safe_identifier}"
        endpoint = urljoin(self.base_url + "/", path.lstrip("/"))
        params = self._params(spec, top_k, language)
        try:
            response = requests.get(
                endpoint,
                params=params,
                headers={"Accept": "application/json", "User-Agent": "MERIDIAN/1.0 (IPU Parline provider)"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(f"IPU Parline API request timed out: {exc}") from exc
        except requests.exceptions.HTTPError as exc:
            status = getattr(response, "status_code", "unknown")
            if status in (401, 403):
                raise RuntimeError(f"IPU Parline API authentication/access error: {status}") from exc
            raise RuntimeError(f"IPU Parline API request failed: {exc}") from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"IPU Parline API request failed: {exc}") from exc
        except ValueError as exc:
            raise RuntimeError(f"Invalid JSON response from IPU Parline API: {exc}") from exc

        if not isinstance(payload, dict) or "data" not in payload:
            raise RuntimeError("Unexpected IPU Parline response structure: missing data")
        records = self._records(payload["data"])
        if not records:
            return []
        meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
        retrieved = datetime.utcnow()
        results: List[ExternalEvidence] = []
        for index, record in enumerate(records[: min(top_k, self.max_page_size)]):
            attrs = self._record_attributes(record)
            fields = []
            for key, raw_value in attrs.items():
                if key in {"annotation", "field_entity", "entity_code", "missing_reason"}:
                    continue
                value = self._unwrap(raw_value, language or "en")
                if value not in (None, "", [], {}):
                    fields.append(f"{key}: {value}")
            if not fields:
                continue
            identifier = self._stable_id(record, attrs, index)
            label = next((str(self._unwrap(attrs.get(key), language or "en")) for key in ("country_name", "chamber_name", "parliament_name", "election_name") if attrs.get(key) is not None), resource)
            record_url = record.get("links", {}).get("self") if isinstance(record.get("links"), dict) else None
            url = record_url or response.url
            metadata = {
                "resource": resource,
                "query": spec,
                "observation": record,
                "api_metadata": meta,
                "text_limitation": "IPU Parline structured observation; not a natural-language claim verdict.",
            }
            ev = ExternalEvidence(
                evidence_id=identifier,
                text=f"IPU Parline structured observation for {label}: " + "; ".join(fields),
                source="IPU Parline",
                title=f"IPU Parline {resource}: {label}",
                url=url,
                domain=domain or "political",
                retrieved_date=retrieved,
                metadata=metadata,
                publication_date=None,
                language=language,
                source_type="ipu_parline",
                authority_score=IPU_PARLINE_AUTHORITY_SCORE,
            )
            results.append(ev)
        return results
    supported_domains = frozenset({"political"})
