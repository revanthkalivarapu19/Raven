from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from rag_ingestion.external_retrieval.source_manager import classify_provider_failure
from rag_ingestion.external_retrieval.news_api_source import NewsAPISource
from rag_ingestion.external_retrieval.fact_check_source import FactCheckSource
from rag_ingestion.external_retrieval.gdelt_source import GDELTSource
from rag_ingestion.external_retrieval.evidence_normalizer import ExternalEvidenceNormalizer
from rag_ingestion.external_retrieval.web_retriever import WebSourceRetriever
from rag_ingestion.retrieval.evidence_fusion import EvidenceFusionManager

QUERIES = (
    "Delhi AQI 400 Diwali 2026",
    "Akshayakalpa Farms HMWSB 20000",
    "Marco Rubio Iran talks November 2026",
)


def main() -> int:
    providers = (
        ("NewsAPI", NewsAPISource(timeout=6)),
        ("GDELT", GDELTSource(timeout=5, max_retries=0)),
        ("Google Fact Check", FactCheckSource(timeout=6)),
    )
    output = []
    first_news_candidates = []
    for query_index, query in enumerate(QUERIES):
        for name, provider in providers:
            started = time.perf_counter()
            status = None
            candidates = []
            failure = None
            try:
                candidates = provider.search(query, domain="political", top_k=5, language="en")
                status = getattr(provider, "last_http_status", None)
                if not candidates:
                    failure = "zero_results"
            except Exception as exc:
                status = getattr(provider, "last_http_status", None)
                failure = classify_provider_failure(exc)
            elapsed = round((time.perf_counter() - started) * 1000)
            titles = [item.title for item in candidates[:3] if item.title]
            output.append({
                "provider": name,
                "query": query,
                "http_status": status,
                "runtime_ms": elapsed,
                "candidate_count": len(candidates),
                "first_three_titles": titles,
                "failure_category": failure,
            })
            if query_index == 0 and name == "NewsAPI" and candidates:
                first_news_candidates = candidates[:3]

    page_checks = []
    if first_news_candidates:
        normalizer = ExternalEvidenceNormalizer()
        retriever = WebSourceRetriever(timeout=8)
        for candidate in first_news_candidates[:3]:
            fetched = retriever.retrieve(candidate.url)
            item = {
                "candidate_title": candidate.title,
                "source_url": candidate.url,
                "fetched": bool(fetched.text and not fetched.error),
                "http_status": fetched.status_code,
                "extracted_characters": len(fetched.text or ""),
                "quality_accepted": False,
                "fusion_accepted": False,
                "quality_rejection_type": None,
                "fused_count": 0,
            }
            if fetched.text and not fetched.error:
                try:
                    normalized = normalizer.normalize(candidate, web_result=fetched)
                    item["quality_accepted"] = True
                    accepted = EvidenceFusionManager().fuse(
                        local_evidence=[], external_evidence=[normalized], top_k=3,
                        claim=QUERIES[0], domain="political",
                    )
                    item["fusion_accepted"] = bool(accepted)
                    item["fused_count"] = len(accepted)
                except Exception as exc:
                    item["quality_rejection_type"] = type(exc).__name__
            page_checks.append(item)
    page_check = {
        "fetches_attempted": len(page_checks),
        "full_page_accepted_evidence": any(item["fetched"] and item["quality_accepted"] and item["fusion_accepted"] for item in page_checks),
        "candidates": page_checks,
    }
    print(json.dumps({"provider_results": output, "direct_page_quality_fusion_check": page_check}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


