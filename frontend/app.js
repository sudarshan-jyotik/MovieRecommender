(() => {
  const params = new URLSearchParams(window.location.search);
  const overrideApi = params.get("api");
  if (overrideApi) {
    localStorage.setItem("cineMatchApiUrl", overrideApi.replace(/\/+$/, ""));
  }

  const configuredUrl =
    overrideApi ||
    localStorage.getItem("cineMatchApiUrl") ||
    (window.MOVIE_RECOMMENDER_CONFIG && window.MOVIE_RECOMMENDER_CONFIG.API_BASE_URL) ||
    "http://localhost:8000";

  const API_BASE = configuredUrl.replace(/\/+$/, "");

  const searchInput = document.getElementById("movieSearch");
  const dropdown = document.getElementById("searchDropdown");
  const limitSelect = document.getElementById("limitSelect");
  const recommendBtn = document.getElementById("recommendBtn");
  const alertBanner = document.getElementById("alertBanner");
  const selectedSection = document.getElementById("selectedSection");
  const resultsSection = document.getElementById("resultsSection");
  const resultsHeading = document.getElementById("resultsHeading");
  const resultsMeta = document.getElementById("resultsMeta");
  const resultsGrid = document.getElementById("resultsGrid");
  const apiStatus = document.getElementById("apiStatus");
  const apiStatusText = document.getElementById("apiStatusText");

  let selectedMovieId = 19995;
  let searchTimer = null;

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text == null ? "" : String(text);
    return div.innerHTML;
  }

  function makeFallbackPoster(title) {
    const safe = escapeHtml((title || "Movie").slice(0, 42));
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="400" height="600" viewBox="0 0 400 600">
      <defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="#1e293b"/><stop offset="100%" stop-color="#0f172a"/>
      </linearGradient></defs>
      <rect width="400" height="600" fill="url(#g)"/>
      <rect x="20" y="20" width="360" height="560" fill="none" stroke="#334155" stroke-width="2" rx="12"/>
      <text x="200" y="265" text-anchor="middle" fill="#64748b" font-family="sans-serif" font-size="16" font-weight="600">NO POSTER</text>
      <text x="200" y="310" text-anchor="middle" fill="#e2e8f0" font-family="sans-serif" font-size="20" font-weight="700">${safe}</text>
    </svg>`;
    return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
  }

  function showAlert(message) {
    if (!message) {
      alertBanner.classList.add("hidden");
      alertBanner.textContent = "";
      return;
    }
    alertBanner.textContent = message;
    alertBanner.classList.remove("hidden");
  }

  async function checkHealth() {
    if (window.location.protocol === "https:" && API_BASE.startsWith("http://localhost")) {
      apiStatus.classList.add("offline");
      apiStatusText.textContent = "Backend URL not configured";
      showAlert(
        "Update frontend/config.js with your live Render API URL (or append ?api=https://your-service.onrender.com) to connect."
      );
      return;
    }

    try {
      const res = await fetch(`${API_BASE}/health`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      apiStatus.classList.remove("offline");
      apiStatus.classList.add("online");
      apiStatusText.textContent = `${data.movies_count.toLocaleString()} movies ready`;
    } catch {
      apiStatus.classList.remove("online");
      apiStatus.classList.add("offline");
      apiStatusText.textContent = "API waking or offline";
    }
  }

  async function searchMovies(query) {
    try {
      const res = await fetch(
        `${API_BASE}/movies?q=${encodeURIComponent(query)}&limit=12`
      );
      if (!res.ok) throw new Error("Search failed");
      const data = await res.json();
      renderDropdown(data.movies || []);
    } catch {
      dropdown.classList.add("hidden");
    }
  }

  function renderDropdown(movies) {
    dropdown.innerHTML = "";
    if (!movies.length) {
      dropdown.classList.add("hidden");
      return;
    }
    for (const movie of movies) {
      const li = document.createElement("li");
      li.innerHTML = `
        <span>${escapeHtml(movie.label)}</span>
        <span class="dropdown-meta">${escapeHtml(movie.release_year || "")}${
        movie.rating ? ` · ★ ${movie.rating}` : ""
      }</span>
      `;
      li.addEventListener("click", () => {
        selectedMovieId = movie.movie_id;
        searchInput.value = movie.label;
        dropdown.classList.add("hidden");
        loadRecommendations(movie.movie_id);
      });
      dropdown.appendChild(li);
    }
    dropdown.classList.remove("hidden");
  }

  function renderSelectedMovie(movie) {
    const poster = movie.poster_url || makeFallbackPoster(movie.title);
    const genresHtml = (movie.genres || [])
      .slice(0, 4)
      .map((g) => `<span class="badge">${escapeHtml(g)}</span>`)
      .join("");

    selectedSection.innerHTML = `
      <img class="selected-poster" src="${poster}" alt="${escapeHtml(movie.title)} poster" />
      <div class="selected-info">
        <h2>${escapeHtml(movie.title)}</h2>
        ${movie.tagline ? `<p class="tagline">“${escapeHtml(movie.tagline)}”</p>` : ""}
        <div class="badges">
          ${movie.release_year ? `<span class="badge">${escapeHtml(movie.release_year)}</span>` : ""}
          ${movie.rating ? `<span class="badge rating">★ ${escapeHtml(movie.rating)}</span>` : ""}
          ${movie.runtime ? `<span class="badge">${escapeHtml(movie.runtime)} min</span>` : ""}
          ${genresHtml}
        </div>
        ${movie.overview ? `<p class="overview">${escapeHtml(movie.overview)}</p>` : ""}
        <a class="tmdb-link" href="${escapeHtml(movie.tmdb_url)}" target="_blank" rel="noopener noreferrer">
          View on TMDB ↗
        </a>
      </div>
    `;
    selectedSection.classList.remove("hidden");
  }

  function renderRecommendations(sourceMovie, recommendations) {
    resultsHeading.textContent = `Similar to ${sourceMovie.title}`;
    resultsMeta.textContent = `${recommendations.length} recommendations`;
    resultsGrid.innerHTML = "";

    for (const rec of recommendations) {
      const poster = rec.poster_url || makeFallbackPoster(rec.title);
      const subParts = [];
      if (rec.release_year) subParts.push(rec.release_year);
      if (rec.rating) subParts.push(`★ ${rec.rating}`);
      if (rec.genres && rec.genres.length) subParts.push(rec.genres.slice(0, 2).join(", "));

      const card = document.createElement("article");
      card.className = "movie-card";
      card.innerHTML = `
        <div class="card-poster-wrap">
          <img class="card-poster" src="${poster}" alt="${escapeHtml(rec.title)} poster" loading="lazy" />
          <span class="card-match-pill">${escapeHtml(rec.match_percent)}% match</span>
        </div>
        <div class="card-body">
          <h3 class="card-title">${escapeHtml(rec.title)}</h3>
          <p class="card-sub">${escapeHtml(subParts.join(" · "))}</p>
          ${rec.overview ? `<p class="card-overview">${escapeHtml(rec.overview)}</p>` : ""}
          <div class="card-actions">
            <button class="btn-ghost explore-btn" type="button">Similar</button>
            <a class="tmdb-link" href="${escapeHtml(rec.tmdb_url)}" target="_blank" rel="noopener noreferrer">
              TMDB ↗
            </a>
          </div>
        </div>
      `;

      card.querySelector(".explore-btn").addEventListener("click", () => {
        selectedMovieId = rec.movie_id;
        searchInput.value = rec.label || rec.title;
        loadRecommendations(rec.movie_id);
        window.scrollTo({ top: 0, behavior: "smooth" });
      });

      resultsGrid.appendChild(card);
    }

    resultsSection.classList.remove("hidden");
  }

  async function loadRecommendations(movieId) {
    showAlert("");
    recommendBtn.disabled = true;
    recommendBtn.textContent = "Loading…";
    try {
      const limit = Number(limitSelect.value) || 10;
      const res = await fetch(`${API_BASE}/recommend/${movieId}?limit=${limit}`);
      if (!res.ok) throw new Error(`API returned ${res.status}`);
      const data = await res.json();
      renderSelectedMovie(data.movie);
      renderRecommendations(data.movie, data.recommendations || []);
      apiStatus.classList.remove("offline");
      apiStatus.classList.add("online");
      apiStatusText.textContent = "Connected";
    } catch (err) {
      showAlert(
        `Could not reach backend at ${API_BASE}. If Render free tier was sleeping, wait ~30 seconds and click Recommend again.`
      );
    } finally {
      recommendBtn.disabled = false;
      recommendBtn.textContent = "Recommend";
    }
  }

  searchInput.addEventListener("input", () => {
    clearTimeout(searchTimer);
    const q = searchInput.value.trim();
    if (!q) {
      dropdown.classList.add("hidden");
      return;
    }
    searchTimer = setTimeout(() => searchMovies(q), 180);
  });

  document.addEventListener("click", (e) => {
    if (!e.target.closest(".search-box")) {
      dropdown.classList.add("hidden");
    }
  });

  recommendBtn.addEventListener("click", () => {
    loadRecommendations(selectedMovieId);
  });

  limitSelect.addEventListener("change", () => {
    loadRecommendations(selectedMovieId);
  });

  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      selectedMovieId = Number(chip.dataset.movieId);
      searchInput.value = chip.textContent.trim();
      dropdown.classList.add("hidden");
      loadRecommendations(selectedMovieId);
    });
  });

  checkHealth();
  searchInput.value = "Avatar";
  loadRecommendations(selectedMovieId);
})();

