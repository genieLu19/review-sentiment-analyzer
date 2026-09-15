import re


def clean_text(text: str) -> str:
    """Normalize review text before vectorization."""
    if text is None:
        return ""

    normalized = str(text).lower().replace("ё", "е")
    normalized = re.sub(r"[^a-zа-я0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized
