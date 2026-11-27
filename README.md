# CineMatch — Movie Recommender System

Content-based movie recommendation engine across **4,800** unique films, available as a **FastAPI (Render) + Web UI (Vercel)** deployment and as a **Streamlit** app.

## Live Deployment

- **Frontend (Vercel):** https://movierecommender-frontend.vercel.app/
- **Backend API (Render):** https://movie-recommender-api-oo2s.onrender.com
  - Health check: `GET /health`
  - Search movies: `GET /movies?q=Inception&limit=12`
  - Movie details: `GET /movie/{movie_id}`
  - Recommendations: `GET /recommend/{movie_id}?limit=10`

> Note: The Render free tier spins down after ~15 minutes of inactivity; the first request after idle may take ~30 seconds to wake up.

## Project Structure

- `recommender.py` — Shared recommendation engine (`MovieRecommender`), `.env` loader, and resilient OMDb/TMDB metadata client with offline & SVG poster fallbacks.
- `data/movies_clean.json.gz` & `data/similarity_topk.npz` — Deduplicated movie metadata (year, rating, runtime, genres, tagline, overview) and compact precomputed top-25 cosine similarity matrix (~1.2 MB total).
- `backend/main.py` — FastAPI service for Render (`backend/requirements.txt`, `render.yaml`).
- `frontend/` — Static web UI deployed on Vercel (`index.html`, `styles.css`, `app.js`, `config.js`, `vercel.json`).
- `app.py` — Streamlit UI.
- `scripts/build_artifacts.py` — Rebuilds `data/` artifacts from `movie_dict.pkl`, `similarity.pkl`, and `tmdb_5000_movies.csv`.
- `tests/test_recommender.py` — Automated unit and integration tests for the recommender, OMDb/TMDB fallbacks, FastAPI endpoints, and Streamlit UI.

## Local Quickstart

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. *(Optional)* Copy `.env.example` to `.env` and add a free `OMDB_API_KEY` (from https://www.omdbapi.com/apikey.aspx) or `TMDB_API_KEY` for live movie posters:

   ```bash
   cp .env.example .env
   ```

3. Run **FastAPI + Web UI**:

   ```bash
   uvicorn backend.main:app --reload --port 8000
   # Serve frontend/ and point it to your local API via ?api=http://localhost:8000
   python -m http.server 3000 --directory frontend
   ```

   Or run the **Streamlit app**:

   ```bash
   streamlit run app.py
   ```

## Running Tests

```bash
python -m unittest discover -s tests -v
```

## Deploying to Render (Backend) + Vercel (Frontend)

1. **Render:** Create a Blueprint or Web Service from this repository using `render.yaml`, and set `OMDB_API_KEY` (and optionally `TMDB_API_KEY`) in Render's Environment settings.
2. **Vercel:** Import this repository with **Root Directory** set to `frontend`. `frontend/config.js` points to the Render backend URL (`API_BASE_URL`).
3. **CORS:** Set `ALLOWED_ORIGINS` on Render to your Vercel origin (e.g. `https://movierecommender-frontend.vercel.app`).
