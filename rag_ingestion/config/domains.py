"""Canonical Box 6 domain names and safe compatibility aliases."""

CANONICAL_DOMAINS = frozenset({"medical", "political", "science", "finance", "general"})
DOMAIN_ALIASES = {"politics": "political"}


def canonical_domain(domain: str) -> str:
    if not isinstance(domain, str) or not domain.strip():
        raise ValueError("domain must be a non-empty string")
    value = domain.strip().lower()
    return DOMAIN_ALIASES.get(value, value)


def validate_domain(domain: str) -> str:
    value = canonical_domain(domain)
    if value not in CANONICAL_DOMAINS:
        raise ValueError(
            f"Unsupported Box 6 domain '{domain}'. Expected one of: "
            + ", ".join(sorted(CANONICAL_DOMAINS - {"general"}))
        )
    return value
