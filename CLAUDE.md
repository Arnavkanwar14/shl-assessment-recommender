# SHL Assessment Recommender

Assignment project: a conversational agent that recommends SHL assessments
from a scraped catalog. FastAPI + Groq (Llama-3.3-70B) + TF-IDF retrieval,
deployed on Railway.

## Run
- `uvicorn main:app --reload` (needs `GROQ_API_KEY` in the environment —
  Railway sets it in prod; there is no .env file here).
- Health check: `GET /health`. Main endpoint: `POST /chat`.
- No test suite; verify by hitting `/chat` with a real conversation.

## Structure
- `main.py` — FastAPI app + request/response models.
- `agent.py` — Groq conversation agent (system prompt lives here).
- `retrieval.py` — TF-IDF search over `data/shl_catalog.json`.

## Decisions already made — don't relitigate
- Gemini was abandoned for Groq because of free-tier 429s (see git log).
- TF-IDF over embeddings: zero-dependency, fast Railway cold starts.
- Hard product rules: clarify-first conversation, 5–10 recommendations,
  8-turn conversation cap (enforced in main.py).

## Gotchas
- Groq free tier ≈ 14,400 requests/day — fine for demo traffic, don't add
  retry loops that burn it.
- `retrieval._ensure_loaded()` preloads the index at startup; keep it, the
  first request times out on Railway without it.
- The shortsmaker video pipeline is a SEPARATE project at
  `..\shortsmaker` with its own repo — don't mix work between them.
