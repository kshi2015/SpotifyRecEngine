/**
 * app.js — Spotify Rec Engine Frontend
 *
 * Vanilla JS, no framework, no bundler.
 * Intentionally procedural so it's easy to read top-to-bottom.
 *
 * Flow:
 *   1. loadUsers()         — fetch all users, render the sidebar
 *   2. selectUser(userId)  — fetch history + recommendations, render both
 *   3. renderHistory()     — show what the user has been listening to
 *   4. renderRecommendations() — show 5 algorithm columns side-by-side
 */

const API = "http://localhost:8000/api";

// Track which user is currently selected (so we can highlight sidebar item)
let selectedUserId = null;

// ── Algorithm metadata (display names, descriptions, CSS tag class) ──────────
const ALGORITHMS = [
  {
    key: "user_cf",
    name: "User-CF",
    tagClass: "tag-cf",
    description: "Finds users with similar taste and recommends what they love.",
  },
  {
    key: "item_cf",
    name: "Item-CF",
    tagClass: "tag-cf2",
    description: "Finds songs similar to ones you've already played.",
  },
  {
    key: "svd",
    name: "SVD",
    tagClass: "tag-svd",
    description: "Discovers hidden taste factors via matrix factorization.",
  },
  {
    key: "content",
    name: "Content",
    tagClass: "tag-content",
    description: "Matches songs by audio features: energy, tempo, mood.",
  },
  {
    key: "hybrid",
    name: "Hybrid",
    tagClass: "tag-hybrid",
    description: "Weighted blend of all four algorithms.",
  },
];

// ── Startup: load users into sidebar ─────────────────────────────────────────

async function loadUsers() {
  const listEl = document.getElementById("user-list");
  listEl.innerHTML = '<div class="loading"><div class="spinner"></div></div>';

  let users;
  try {
    const res = await fetch(`${API}/users`);
    users = await res.json();
  } catch (err) {
    listEl.innerHTML = '<div style="padding:16px;color:#f44;font-size:13px;">Could not connect to server.<br>Is uvicorn running?</div>';
    return;
  }

  listEl.innerHTML = "";
  for (const user of users) {
    const item = document.createElement("div");
    item.className = "user-item";
    item.dataset.userId = user.user_id;

    // Avatar: first letter of name
    const initials = user.name.charAt(0).toUpperCase();

    item.innerHTML = `
      <div class="user-avatar">${initials}</div>
      <div class="user-info">
        <div class="user-name">${user.name}</div>
        <div class="user-taste">${user.taste_profile}</div>
      </div>
    `;

    item.addEventListener("click", () => selectUser(user.user_id));
    listEl.appendChild(item);
  }
}

// ── Select a user: fetch data + render ───────────────────────────────────────

async function selectUser(userId) {
  // Highlight the selected sidebar item
  document.querySelectorAll(".user-item").forEach(el => {
    el.classList.toggle("active", el.dataset.userId === userId);
  });
  selectedUserId = userId;

  // Show loading state, hide both empty state and user content
  document.getElementById("empty-state").style.display = "none";
  document.getElementById("user-content").style.display = "none";
  document.getElementById("loading-state").style.display = "flex";

  try {
    // Fetch history and recommendations in parallel (both are GET requests)
    const [historyRes, recsRes] = await Promise.all([
      fetch(`${API}/users/${userId}/history`),
      fetch(`${API}/users/${userId}/recommendations?n=10`),
    ]);

    const history = await historyRes.json();
    const recs = await recsRes.json();

    // Update user name + taste profile in the header
    document.getElementById("user-name-display").textContent = recs.user_name;
    document.getElementById("user-taste-display").textContent = recs.taste_profile;

    renderHistory(history);
    renderRecommendations(recs);

    // Hide loading, show content
    document.getElementById("loading-state").style.display = "none";
    document.getElementById("user-content").style.display = "block";

  } catch (err) {
    document.getElementById("loading-state").innerHTML = `
      <span style="color:#f44;">Error: ${err.message}</span>
    `;
  }
}

// ── Render listening history ──────────────────────────────────────────────────

function renderHistory(history) {
  const grid = document.getElementById("history-grid");
  grid.innerHTML = "";

  if (!history || history.length === 0) {
    grid.innerHTML = '<span style="color:var(--text-muted);font-size:13px;">No listening history.</span>';
    return;
  }

  // Show all history items as pills
  for (const item of history) {
    const pill = document.createElement("div");
    pill.className = "history-pill";
    pill.innerHTML = `
      <div class="history-pill-info">
        <div class="history-pill-title">${item.song.title}</div>
        <div class="history-pill-sub">${item.song.artist} · ${item.song.genre}</div>
      </div>
      <div class="play-count">×${item.play_count}</div>
    `;
    grid.appendChild(pill);
  }
}

// ── Render 5 recommendation columns ──────────────────────────────────────────

function renderRecommendations(recs) {
  const grid = document.getElementById("recs-grid");
  grid.innerHTML = "";

  for (const algo of ALGORITHMS) {
    const songs = recs[algo.key] || [];

    // Find the max score in this column for normalizing the score bars
    const maxScore = songs.reduce((max, s) => Math.max(max, s.score), 0);

    const col = document.createElement("div");
    col.className = "algo-column";

    col.innerHTML = `
      <div class="algo-header">
        <div class="algo-tag ${algo.tagClass}">${algo.name}</div>
        <div class="algo-description">${algo.description}</div>
      </div>
      <div class="algo-cards" id="cards-${algo.key}"></div>
    `;
    grid.appendChild(col);

    const cardsEl = col.querySelector(`#cards-${algo.key}`);
    songs.forEach((rec, index) => {
      const card = createSongCard(rec, index + 1, maxScore);
      cardsEl.appendChild(card);
    });

    if (songs.length === 0) {
      cardsEl.innerHTML = '<div style="padding:12px;color:var(--text-muted);font-size:12px;">No recommendations available.</div>';
    }
  }
}

// ── Create a single song recommendation card ──────────────────────────────────

function createSongCard(rec, rank, maxScore) {
  const card = document.createElement("div");
  card.className = "song-card";

  // Normalize score to [0, 1] for the score bar width
  const barWidth = maxScore > 0 ? Math.round((rec.score / maxScore) * 100) : 0;

  card.innerHTML = `
    <div class="song-rank">#${rank}</div>
    <div class="song-title">${rec.song.title}</div>
    <div class="song-meta">${rec.song.artist}</div>
    <div>
      <span class="genre-chip">${rec.song.genre}</span>
    </div>
    <div class="score-bar-track">
      <div class="score-bar-fill" style="width: ${barWidth}%"></div>
    </div>
    <div class="song-explanation">${rec.explanation}</div>
  `;

  return card;
}

// ── Boot ─────────────────────────────────────────────────────────────────────

loadUsers();
