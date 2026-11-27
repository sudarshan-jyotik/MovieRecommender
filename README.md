# Movie Recommender System

Content-based movie recommendation engine across 4,800 TMDB films, available both as a **Streamlit app** and as a **FastAPI (Render) + Web UI (Vercel)** deployment.

## Project Structure

- `recommender.py` — Shared recommendation engine and resilient TMDB API client.
- `data/movies_clean.json.gz` & `data/similarity_topk.npz` — Deduplicated movie metadata and compact precomputed top-25 cosine similarity matrix (~1.2 MB total).
- `app.py` — Streamlit UI.
- `backend/main.py` — FastAPI service for Render (`/health`, `/movies`, `/movie/{movie_id}`, `/recommend/{movie_id}`).
- `frontend/` — Static web UI ready for Vercel (`index.html`, `styles.css`, `app.js`, `config.js`).
- `scripts/build_artifacts.py` — Rebuilds `data/` artifacts from `movie_dict.pkl`, `similarity.pkl`, and `tmdb_5000_movies.csv`.

## Local Quickstart

```bash
pip install -r requirements.txt
# Optional: set OMDB_API_KEY (recommended) or TMDB_API_KEY for live movie posters
streamlit run app.py
```

Or run the FastAPI + Vercel frontend locally:

```bash
uvicorn backend.main:app --reload --port 8000
# Then serve frontend/ (e.g. python -m http.server 3000 --directory frontend)
```

## Deploying to Render (Backend) + Vercel (Frontend)

1. **Render:** Create a Blueprint or Web Service from this repo using `render.yaml`, and add your `OMDB_API_KEY` (from https://www.omdbapi.com/apikey.aspx) or `TMDB_API_KEY` environment variable.
2. **Vercel:** Set the project Root Directory to `frontend` and update `frontend/config.js` (`API_BASE_URL`) to your Render service URL.
3. **(Optional) CORS:** Set `ALLOWED_ORIGINS` on Render to your Vercel domain (e.g. `https://your-app.vercel.app`).

