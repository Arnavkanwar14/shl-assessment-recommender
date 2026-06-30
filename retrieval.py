"""FAISS-based semantic retrieval over the SHL catalog."""

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

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

        # Build a rich text blob for embedding
        text_parts = [name]
        if description:
            text_parts.append(description)
        if keys:
            text_parts.append("Categories: " + ", ".join(keys))
        if job_levels:
            text_parts.append("Job levels: " + ", ".join(job_levels))
        if languages:
            text_parts.append("Languages: " + ", ".join(languages[:5]))
        if duration:
            text_parts.append("Duration: " + duration)

        items.append(
            {
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
                "text": " | ".join(text_parts),
            }
        )
    return items


# ── lazy globals ──────────────────────────────────────────────────────────────
_catalog: Optional[list] = None
_index = None
_model = None


def _ensure_loaded():
    global _catalog, _index, _model
    if _catalog is not None:
        return

    from sentence_transformers import SentenceTransformer
    import faiss

    _catalog = _load_catalog()
    _model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [item["text"] for item in _catalog]
    embeddings = _model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)

    dim = embeddings.shape[1]
    _index = faiss.IndexFlatIP(dim)
    # Normalise for cosine similarity
    faiss.normalize_L2(embeddings)
    _index.add(embeddings)


def search(query: str, top_k: int = 20) -> list[dict]:
    """Return up to top_k catalog items most relevant to query."""
    _ensure_loaded()

    import faiss

    qvec = _model.encode([query], show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(qvec)
    _, idxs = _index.search(qvec, top_k)

    results = []
    for i in idxs[0]:
        if i < 0 or i >= len(_catalog):
            continue
        results.append(_catalog[i])
    return results


def get_all() -> list[dict]:
    _ensure_loaded()
    return list(_catalog)
