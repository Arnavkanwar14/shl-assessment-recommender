"""TF-IDF based semantic retrieval over the SHL catalog."""

import json
import re
from pathlib import Path
from typing import Optional

CATALOG_PATH = Path(__file__).parent / "data" / "shl_catalog.json"

KEYS_TO_CODE = {
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Ability & Aptitude": "A",
    "Simulations": "S",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Motivation": "M",
}


def _load_catalog():
    with open(CATALOG_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    items = []
    for entry in raw:
        name = entry.get("name", "").strip()
        link = entry.get("link", "").strip()
        if not name or not link:
            continue

        keys = entry.get("keys", [])
        codes = sorted(set(KEYS_TO_CODE.get(k, "") for k in keys if k in KEYS_TO_CODE))
        test_type = ",".join(c for c in codes if c)

        description = (entry.get("description") or "").strip()
        job_levels = entry.get("job_levels") or []
        languages = entry.get("languages") or []
        duration = (entry.get("duration") or "").strip()
        remote = entry.get("remote", "yes")
        adaptive = entry.get("adaptive", "no")

        text_parts = [name, name]  # weight name higher
        if description:
            text_parts.append(description)
        if keys:
            text_parts.append(" ".join(keys))
        if job_levels:
            text_parts.append(" ".join(job_levels))
        if languages:
            text_parts.append(" ".join(languages[:5]))

        items.append({
            "name": name,
            "url": link,
            "test_type": test_type,
            "keys": keys,
            "job_levels": job_levels,
            "languages": languages,
            "duration": duration,
            "remote": remote,
            "adaptive": adaptive,
            "description": description,
            "text": " ".join(text_parts),
        })
    return items


# Lazy globals
_catalog: Optional[list] = None
_vectorizer = None
_matrix = None


def _ensure_loaded():
    global _catalog, _vectorizer, _matrix
    if _catalog is not None:
        return

    from sklearn.feature_extraction.text import TfidfVectorizer

    _catalog = _load_catalog()
    texts = [item["text"] for item in _catalog]

    _vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=20000,
        sublinear_tf=True,
    )
    _matrix = _vectorizer.fit_transform(texts)


def search(query: str, top_k: int = 20) -> list[dict]:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    _ensure_loaded()

    qvec = _vectorizer.transform([query])
    scores = cosine_similarity(qvec, _matrix)[0]
    top_idxs = np.argsort(scores)[::-1][:top_k]
    return [_catalog[i] for i in top_idxs if scores[i] > 0]


def get_all() -> list[dict]:
    _ensure_loaded()
    return list(_catalog)
