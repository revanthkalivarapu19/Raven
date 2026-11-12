from __future__ import annotations

import json
import sys
import time

import requests

BASE_URL = "http://127.0.0.1:8000/api/verify"
CASES = {
    "delhi": "Thick smog and hazardous air quality (AQI crossing 400 in several areas and over 200 firecracker timing violation cases filed) have blanketed New Delhi following Diwali celebrations",
    "telangana": "Telangana Chief Minister announced that the radial road from Shamshabad to Kodangal will be completed by December 2027, HMWS&SB fined Akshayakalpa Farms ₹20,000 for wasting water, and Hyderabad firefighters battled multiple Diwali fire incidents across the state.",
    "rubio_trump": "Marco Rubio noted that US-Iran talks have not been productive, calling Iranian demands unacceptable, while Donald Trump stated that renewed US strikes on Iran are highly unlikely.",
    "earth_sun": "The Earth orbits the Sun once every year.",
    "smoking": "Smoking increases the risk of lung cancer.",
    "earth_centered": "The Earth is the center of the solar system and the Sun orbits Earth.",
}


def summarize_provider(row):
    return {
        "provider": row.get("provider"),
        "query_id": row.get("query_id"),
        "provider_query": row.get("provider_query") or row.get("query"),
        "http_status": row.get("http_status"),
        "runtime_ms": row.get("runtime_ms"),
        "candidate_count": row.get("candidate_count"),
        "accepted_count": row.get("accepted"),
        "failure_category": row.get("failure_category"),
    }


def main():
    session = requests.Session()
    session.trust_env = False
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    ready = False
    for _ in range(60):
        try:
            if session.get("http://127.0.0.1:8000/api/health", timeout=2).status_code == 200:
                ready = True
                break
        except requests.RequestException:
            time.sleep(0.5)
    if not ready:
        raise RuntimeError("FastAPI did not become ready on 127.0.0.1:8000")
    for name, claim in CASES.items():
        started = time.perf_counter()
        try:
            response = session.post(BASE_URL, json={"text": claim}, timeout=180)
            elapsed = round(time.perf_counter() - started, 2)
            body = response.json()
        except Exception as exc:
            print(json.dumps({
                "case": name,
                "runtime_seconds": round(time.perf_counter() - started, 2),
                "transport_exception_type": type(exc).__name__,
            }, ensure_ascii=True), flush=True)
            continue

        components = []
        for component in body.get("claim_components", []):
            diag = component.get("retrieval_diagnostics") or {}
            components.append({
                "component_id": component.get("id"),
                "component_text": component.get("text"),
                "queries": component.get("search_queries", []),
                "providers": [summarize_provider(row) for row in diag.get("providers", [])],
                "fetched_pages": diag.get("fetched_pages", 0),
                "quality_rejected": diag.get("quality_rejected", 0),
                "final_evidence_count": diag.get("final_evidence_count", diag.get("ranked_evidence", len(component.get("evidence", [])))),
                "verifier_relationship": component.get("relationship"),
                "verifier_confidence": component.get("confidence"),
                "evidence_count": len(component.get("evidence", [])),
            })
        print(json.dumps({
            "case": name,
            "claim": claim,
            "http_status": response.status_code,
            "runtime_seconds": elapsed,
            "api_verdict": body.get("verdict"),
            "api_confidence": body.get("confidence"),
            "api_status": body.get("status"),
            "fused_evidence_count": len(body.get("evidence", [])),
            "evidence": [
                {"source": item.get("source"), "stance": item.get("stance"), "url": item.get("url")}
                for item in body.get("evidence", [])
            ],
            "components": components,
        }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
