"""Streamlit UI for the content-based Movie Recommender."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import streamlit as st

from recommender import (
    MovieMetadata,
    MovieRecommender,
    build_placeholder_poster,
    fetch_movie_metadata,
)

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
)


def resolve_secret(name: str) -> str:
    """Read an API key from environment or Streamlit secrets."""
    env_val = os.getenv(name, "").strip()
    if env_val:
        return env_val
    try:
        return str(st.secrets.get(name, "")).strip()
    except Exception:
        return ""


@st.cache_resource
def get_recommender() -> MovieRecommender:
    """Load the recommender once per server process."""
    return MovieRecommender()


@st.cache_data(ttl=3600, show_spinner=False)
def get_movie_metadata(movie_id: int, tmdb_key: str, omdb_key: str) -> MovieMetadata:
    """Cache OMDb/TMDB metadata lookups merged with offline dataset details."""
    rec = get_recommender()
    fallback = rec.get_movie(movie_id)
    return fetch_movie_metadata(
        movie_id=movie_id,
        title=str(fallback.get("title") or ""),
        release_year=str(fallback.get("release_year") or ""),
        tmdb_api_key=tmdb_key,
        omdb_api_key=omdb_key,
        fallback=fallback,
    )


def fetch_metadata_batch(
    movie_ids: list[int], tmdb_key: str, omdb_key: str
) -> dict[int, MovieMetadata]:
    """Fetch metadata for multiple movies concurrently."""
    rec = get_recommender()
    if not tmdb_key and not omdb_key:
        return {
            mid: fetch_movie_metadata(
                movie_id=mid, tmdb_api_key="", omdb_api_key="", fallback=rec.get_movie(mid)
            )
            for mid in movie_ids
        }
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(movie_ids)))) as pool:
        results = list(
            pool.map(lambda mid: get_movie_metadata(mid, tmdb_key, omdb_key), movie_ids)
        )
    return {meta.movie_id: meta for meta in results}


recommender = get_recommender()
tmdb_api_key = resolve_secret("TMDB_API_KEY")
omdb_api_key = resolve_secret("OMDB_API_KEY")

st.title("🎬 Movie Recommender System")
st.caption("Content-based recommendations across 4,800 movies.")

if not tmdb_api_key and not omdb_api_key:
    st.info(
        "Set `OMDB_API_KEY` (or `TMDB_API_KEY`) in your environment or Streamlit secrets to load "
        "live posters. Offline movie details and fallback poster cards are shown meanwhile."
    )

movie_ids = [int(m["movie_id"]) for m in recommender.movies]

col_select, col_count = st.columns([3, 1])
with col_select:
    selected_movie_id = st.selectbox(
        "Select a movie:",
        options=movie_ids,
        format_func=recommender.format_movie_label,
    )
with col_count:
    num_recommendations = st.slider(
        "Recommendations",
        min_value=5,
        max_value=min(15, recommender.max_k),
        value=5,
        step=1,
    )

if st.button("Recommend", type="primary"):
    with st.spinner("Finding similar movies..."):
        recommendations = recommender.recommend(selected_movie_id, limit=num_recommendations)
        meta_by_id = fetch_metadata_batch(
            [rec["movie_id"] for rec in recommendations],
            tmdb_key=tmdb_api_key,
            omdb_key=omdb_api_key,
        )

    st.subheader(
        f"Top {len(recommendations)} picks similar to {recommender.format_movie_label(selected_movie_id)}"
    )

    for start in range(0, len(recommendations), 5):
        row_recs = recommendations[start : start + 5]
        cols = st.columns(5)
        for col, rec in zip(cols, row_recs):
            meta = meta_by_id[rec["movie_id"]]
            poster = meta.poster_url or build_placeholder_poster(rec["title"])
            subtitle_parts = [f"{rec['match_percent']}% match"]
            if meta.release_year:
                subtitle_parts.append(meta.release_year)
            if meta.rating is not None:
                subtitle_parts.append(f"⭐ {meta.rating}")

            with col:
                st.image(poster, use_container_width=True)
                st.markdown(f"**[{rec['title']}]({meta.tmdb_url})**")
                st.caption(" · ".join(subtitle_parts))
                if meta.genres:
                    st.caption(" • ".join(meta.genres[:3]))
                if meta.overview:
                    with st.expander("Overview"):
                        st.write(meta.overview)
