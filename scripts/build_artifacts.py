"""Build compact, deduplicated movie and top-K similarity artifacts."""

from __future__ import annotations

import ast
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
TOP_K = 25


def _parse_genres(raw: object) -> list[str]:
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        items = ast.literal_eval(raw)
        return [str(item["name"]) for item in items if isinstance(item, dict) and "name" in item]
    except (ValueError, SyntaxError):
        return []


def build_artifacts(
    movie_dict_path: Path = ROOT / "movie_dict.pkl",
    similarity_path: Path = ROOT / "similarity.pkl",
    tmdb_movies_csv: Path = ROOT / "tmdb_5000_movies.csv",
    output_dir: Path = DATA_DIR,
    top_k: int = TOP_K,
) -> tuple[Path, Path]:
    """Deduplicate movies by movie_id, enrich offline metadata, and save top-K neighbors."""
    with open(movie_dict_path, "rb") as f:
        raw_dict = pickle.load(f)
    df = pd.DataFrame(raw_dict)

    with open(similarity_path, "rb") as f:
        sim = pickle.load(f)

    # Preserve first occurrence of each unique movie_id and align similarity rows/cols
    keep_mask = ~df["movie_id"].duplicated(keep="first")
    keep_indices = np.flatnonzero(keep_mask.to_numpy())

    clean_df = df.iloc[keep_indices].reset_index(drop=True)
    clean_sim = sim[np.ix_(keep_indices, keep_indices)].astype(np.float32, copy=False)

    # Enrich with offline metadata from tmdb_5000_movies.csv when available
    meta_by_id: dict[int, dict[str, object]] = {}
    if tmdb_movies_csv.exists():
        tmdb_df = pd.read_csv(tmdb_movies_csv)
        for row in tmdb_df.itertuples(index=False):
            mid = int(row.id)
            release_date = str(getattr(row, "release_date", "") or "")
            release_year = release_date[:4] if len(release_date) >= 4 and release_date[:4].isdigit() else ""
            vote_avg = getattr(row, "vote_average", None)
            rating = round(float(vote_avg), 1) if pd.notna(vote_avg) and float(vote_avg) > 0 else None
            runtime_val = getattr(row, "runtime", None)
            runtime = int(round(float(runtime_val))) if pd.notna(runtime_val) and float(runtime_val) > 0 else None
            overview = str(getattr(row, "overview", "") or "").strip()
            if overview == "nan":
                overview = ""
            tagline = str(getattr(row, "tagline", "") or "").strip()
            if tagline == "nan":
                tagline = ""
            meta_by_id[mid] = {
                "release_year": release_year,
                "rating": rating,
                "runtime": runtime,
                "genres": _parse_genres(getattr(row, "genres", "")),
                "tagline": tagline,
                "overview": overview,
            }

    records: list[dict[str, object]] = []
    for row in clean_df.itertuples(index=False):
        mid = int(row.movie_id)
        extra = meta_by_id.get(mid, {})
        records.append(
            {
                "movie_id": mid,
                "title": str(row.title),
                "release_year": extra.get("release_year", ""),
                "rating": extra.get("rating"),
                "runtime": extra.get("runtime"),
                "genres": extra.get("genres", []),
                "tagline": extra.get("tagline", ""),
                "overview": extra.get("overview", ""),
            }
        )

    n = len(records)
    k = min(top_k, n - 1)

    # Zero out self-similarity so a movie never recommends itself
    np.fill_diagonal(clean_sim, -np.inf)

    part_idx = np.argpartition(clean_sim, -k, axis=1)[:, -k:]
    part_scores = np.take_along_axis(clean_sim, part_idx, axis=1)
    order = np.argsort(-part_scores, axis=1)

    top_indices = np.take_along_axis(part_idx, order, axis=1).astype(np.uint16)
    top_scores = np.take_along_axis(part_scores, order, axis=1).astype(np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    movies_out = output_dir / "movies_clean.json.gz"
    sim_out = output_dir / "similarity_topk.npz"

    import gzip

    with gzip.open(movies_out, "wt", encoding="utf-8") as gz:
        json.dump(records, gz, ensure_ascii=False, separators=(",", ":"))

    np.savez_compressed(sim_out, indices=top_indices, scores=top_scores)

    legacy_pkl = output_dir / "movies_clean.pkl"
    if legacy_pkl.exists():
        legacy_pkl.unlink()

    return movies_out, sim_out


if __name__ == "__main__":
    m_path, s_path = build_artifacts()
    print(f"Created {m_path} ({m_path.stat().st_size:,} bytes)")
    print(f"Created {s_path} ({s_path.stat().st_size:,} bytes)")
