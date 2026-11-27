# CineMatch — Movie Recommender & Conversational Agent

Content-based movie discovery engine across **4,800** unique films, featuring a **FastAPI (Render) + Static Web UI (Vercel)** deployment, a **Streamlit** UI, and a tool-calling **conversational movie agent** with persistent memory, semantic embedding search, and an automated evaluation harness.

## Live Deployment

- **Frontend (Vercel):** https://movierecommender-frontend.vercel.app/
- **Backend API (Render):** https://movie-recommender-api-oo2s.onrender.com

> Note: The Render free tier spins down after ~15 minutes of inactivity; the first request after idle may take ~30 seconds to wake up.

## Architecture

```text
+-----------------------------------+
|  Vercel Static UI (frontend/)     |
|  - Search & Similar Movie Cards   |
|  - Chat Panel + Memory Strip      |
+-----------------+-----------------+
                  | HTTPS (JSON)
                  v
+-----------------------------------+       +-------------------------------+
|  FastAPI Backend (backend/main.py)|------>| OpenAI-Compatible LLM (httpx) |
|  - GET  /health, /movies          |       | Groq / OpenAI / custom base   |
|  - GET  /movie/{id}, /recommend   |       +-------------------------------+
|  - POST /chat (rate-limited)      |
|  - GET  /chat/{id}/preferences    |       +-------------------------------+
+---+-------------+-------------+---+------>| OMDb / TMDB Poster Metadata   |
    |             |             |           +-------------------------------+
    v             v             v
+--------+  +-----------+  +--------------------+
| Top-25 |  | Semantic  |  | SQLAlchemy Memory  |
| Cosine |  | fastembed |  | sessions, messages |
| (.npz) |  | float16   |  | user_preferences   |
+--------+  +-----------+  +--------------------+
                           | SQLite (local) /
                           | Render Postgres
```

## API Endpoints

- `GET /health` — Readiness check (`movies_count`, poster API configuration).
- `GET /movies?q=Inception&limit=12` — Substring title/genre search.
- `GET /movie/{movie_id}` — Enriched metadata for one movie.
- `GET /recommend/{movie_id}?limit=10` — Content-similarity recommendations.
- `POST /chat` (`{ "session_id"?: str, "message": str }`) — Rate-limited conversational agent returning `{ session_id, reply: { picks, follow_up_question, message }, trace, preferences }`.
- `GET /chat/{session_id}/preferences` — Stored taste profile (`liked_genres`, `disliked_genres`, `liked_movies`, `constraints`).

## Project Structure

- `recommender.py` — `MovieRecommender` engine, `.env` loader, and OMDb/TMDB metadata client.
- `agent/` — `llm.py` (`httpx` OpenAI-compatible client), `schemas.py` (Pydantic models), `prompts.py` (`v1`/`v2` system prompts & extraction prompt), `tools.py`, `agent.py` (5-step tool loop, JSON retry, hallucination guard), `memory.py` (SQLAlchemy persistence), `semantic.py` (lazy `fastembed` cosine search).
- `data/` — `movies_clean.json.gz`, `similarity_topk.npz`, and `movie_embeddings.npz` (float16 `BAAI/bge-small-en-v1.5`).
- `backend/main.py` — FastAPI app for Render (`backend/requirements.txt`, `render.yaml`).
- `frontend/` — Vanilla JS web UI for Vercel (`index.html`, `styles.css`, `app.js`, `config.js`, `vercel.json`).
- `app.py` — Streamlit UI.
- `scripts/` — `build_artifacts.py` and `build_embeddings.py`.
- `eval/` — `cases.jsonl` (30 cases) and `run_eval.py`.
- `tests/` — Offline `unittest` suite (no network or API keys required).

## Environment Variables

Copy `.env.example` to `.env` for local runs:

| Variable | Default | Purpose |
|---|---|---|
| `OMDB_API_KEY` | — | Live poster & rating lookups (recommended) |
| `TMDB_API_KEY` | — | Optional TMDB v3 key or v4 Bearer token |
| `LLM_API_KEY` | — | API key for OpenAI-compatible `/chat/completions` |
| `LLM_BASE_URL` | `https://api.groq.com/openai/v1` | OpenAI-compatible API base URL |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Chat-completions model name |
| `DATABASE_URL` | `sqlite:///./agent.db` | SQLAlchemy DB URL (Postgres on Render) |
| `ENABLE_SEMANTIC` | `1` | Enable (`1`) or disable (`0`) semantic search tool |
| `CHAT_RATE_LIMIT` | `20/minute` | Per-IP rate limit on `POST /chat` |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |

## Local Quickstart & Tests

```bash
pip install -r requirements.txt
cp .env.example .env

# Run the FastAPI backend + static frontend
uvicorn backend.main:app --reload --port 8000
python -m http.server 3000 --directory frontend

# Or run the Streamlit app
streamlit run app.py

# Run all offline tests (no network or API keys required)
python -m unittest discover -s tests -v
```

## Evaluation Harness

Run the 30-case evaluation suite against a live LLM (`LLM_API_KEY` required):

```bash
python eval/run_eval.py --prompt-version v1
python eval/run_eval.py --prompt-version v2 --model llama-3.3-70b-versatile
```

Each run writes a detailed report to `eval/results/<timestamp>.json` and appends a summary row to [EVAL_RESULTS.md](EVAL_RESULTS.md).

### Results

See [EVAL_RESULTS.md](EVAL_RESULTS.md) for recorded benchmark runs (initialized as `"not yet run"` until executed with a live `LLM_API_KEY`).

## Deploying to Render (Backend) + Vercel (Frontend)

1. **Render:** Sync `render.yaml` as a Blueprint (provisions the free web service + free Postgres `cinematch-db` wired to `DATABASE_URL`) and set `OMDB_API_KEY` and `LLM_API_KEY` in Environment settings.
2. **Vercel:** Import this repository with **Root Directory** set to `frontend`. `frontend/config.js` points to the Render API (`API_BASE_URL`).
