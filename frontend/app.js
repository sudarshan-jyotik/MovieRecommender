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
  const memoryTags = document.getElementById("memoryTags");
  const chatMessages = document.getElementById("chatMessages");
  const chatError = document.getElementById("chatError");
  const chatForm = document.getElementById("chatForm");
  const chatInput = document.getElementById("chatInput");
  const chatSendBtn = document.getElementById("chatSendBtn");
  const newChatBtn = document.getElementById("newChatBtn");

  let selectedMovieId = 19995;
  let searchTimer = null;
  let chatSessionId = loadSessionId();

  function loadSessionId() {
    try {
      return localStorage.getItem("cineMatchSessionId") || null;
    } catch {
      return null;
    }
  }

  function saveSessionId(id) {
    chatSessionId = id;
    try {
      if (id) {
        localStorage.setItem("cineMatchSessionId", id);
      } else {
        localStorage.removeItem("cineMatchSessionId");
      }
    } catch {
      // Ignore storage errors in private browsing or restricted contexts
    }
  }

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

  function showChatError(message) {
    if (!message) {
      chatError.classList.add("hidden");
      chatError.textContent = "";
      return;
    }
    chatError.textContent = message;
    chatError.classList.remove("hidden");
  }

  function renderPreferences(prefs) {
    if (!memoryTags) return;
    const pills = [];
    const addPills = (items, prefix, cls = "") => {
      for (const item of items || []) {
        pills.push(`<span class="memory-pill ${cls}">${escapeHtml(prefix)}: ${escapeHtml(item)}</span>`);
      }
    };
    addPills(prefs && prefs.liked_genres, "Likes");
    addPills(prefs && prefs.liked_movies, "Loved");
    addPills(prefs && prefs.disliked_genres, "Avoid", "avoid");
    addPills(prefs && prefs.constraints, "Rule", "avoid");

    if (!pills.length) {
      memoryTags.innerHTML = '<span class="memory-empty">No preferences saved yet — start chatting!</span>';
      return;
    }
    memoryTags.innerHTML = pills.join("");
  }

  function appendChatMessage(role, text, picks = [], followUp = null) {
    const wrapper = document.createElement("div");
    wrapper.className = `chat-msg ${role}`;
    let html = `<p class="chat-text">${escapeHtml(text)}</p>`;

    if (picks && picks.length) {
      html += '<div class="chat-picks">';
      for (const pick of picks) {
        html += `
          <div class="chat-pick-card">
            <div class="chat-pick-title">${escapeHtml(pick.title)}</div>
            <p class="chat-pick-reason">${escapeHtml(pick.reason)}</p>
            <button class="btn-ghost pick-explore-btn" type="button" data-movie-id="${escapeHtml(pick.movie_id)}" data-title="${escapeHtml(pick.title)}">
              Explore similar
            </button>
          </div>
        `;
      }
      html += "</div>";
    }

    if (followUp) {
      html += `<p class="chat-followup">${escapeHtml(followUp)}</p>`;
    }

    wrapper.innerHTML = html;
    wrapper.querySelectorAll(".pick-explore-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const mid = Number(btn.dataset.movieId);
        selectedMovieId = mid;
        searchInput.value = btn.dataset.title || "";
        loadRecommendations(mid);
        window.scrollTo({ top: 0, behavior: "smooth" });
      });
    });

    chatMessages.appendChild(wrapper);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return wrapper;
  }

  async function loadStoredPreferences() {
    if (!chatSessionId) return;
    try {
      const res = await fetch(`${API_BASE}/chat/${encodeURIComponent(chatSessionId)}/preferences`);
      if (res.status === 404) {
        saveSessionId(null);
        renderPreferences(null);
        return;
      }
      if (!res.ok) return;
      const data = await res.json();
      renderPreferences(data.preferences);
    } catch {
      // Ignore silent background preference check errors
    }
  }

  async function sendChatMessage(rawText) {
    const message = rawText.trim();
    if (!message) return;

    showChatError("");
    appendChatMessage("user", message);
    chatInput.value = "";
    chatSendBtn.disabled = true;
    chatSendBtn.textContent = "Thinking…";
    const loadingEl = appendChatMessage("assistant loading", "Searching catalogue and picking movies…");

    try {
      const payload = { message };
      if (chatSessionId) payload.session_id = chatSessionId;

      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.detail || `Chat API returned HTTP ${res.status}`);
      }

      if (data.session_id) {
        saveSessionId(data.session_id);
      }
      loadingEl.remove();
      const reply = data.reply || {};
      appendChatMessage(
        "assistant",
        reply.message || "Here are my recommendations.",
        reply.picks || [],
        reply.follow_up_question || null
      );
      if (data.preferences) {
        renderPreferences(data.preferences);
      }
    } catch (err) {
      loadingEl.remove();
      showChatError(
        err && err.message
          ? `Chat failed: ${err.message}`
          : "Chat request failed. Check that the backend and LLM_API_KEY are configured."
      );
    } finally {
      chatSendBtn.disabled = false;
      chatSendBtn.textContent = "Send";
    }
  }

  document.querySelectorAll(".chip[data-movie-id]").forEach((chip) => {
    chip.addEventListener("click", () => {
      selectedMovieId = Number(chip.dataset.movieId);
      searchInput.value = chip.textContent.trim();
      dropdown.classList.add("hidden");
      loadRecommendations(selectedMovieId);
    });
  });

  document.querySelectorAll(".chat-starter").forEach((btn) => {
    btn.addEventListener("click", () => {
      sendChatMessage(btn.textContent.trim());
    });
  });

  if (chatForm) {
    chatForm.addEventListener("submit", (e) => {
      e.preventDefault();
      sendChatMessage(chatInput.value);
    });
  }

  if (newChatBtn) {
    newChatBtn.addEventListener("click", () => {
      saveSessionId(null);
      renderPreferences(null);
      showChatError("");
      chatMessages.innerHTML =
        '<div class="chat-msg assistant"><p class="chat-text">Started a fresh chat session. What kind of movie are you looking for?</p></div>';
    });
  }

  checkHealth();
  loadStoredPreferences();
  searchInput.value = "Avatar";
  loadRecommendations(selectedMovieId);
})();

