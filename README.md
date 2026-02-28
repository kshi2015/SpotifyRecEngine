# Spotify Recommendation Engine — A Learning Project

A from-scratch implementation of the five core algorithms behind modern music recommendation systems, built as an educational exercise.

Select a listener from the sidebar and see how each algorithm generates a different set of recommendations — side by side.

![Architecture: FastAPI backend + vanilla JS frontend]

---

## What This Teaches You

| Concept | Where to find it |
|---|---|
| User-Item interaction matrix | `data/mock_data.py` → `LISTEN_MATRIX` |
| Cosine similarity | `algorithms/user_cf.py` |
| Adjusted cosine (item-based) | `algorithms/item_cf.py` |
| Singular Value Decomposition | `algorithms/matrix_factorization.py` |
| Audio feature matching | `algorithms/content_based.py` |
| Hybrid ensemble + normalization | `algorithms/hybrid.py` |
| Cold start problem | Compare Content-Based vs CF for new users |
| Filter bubble effect | Notice how Content-Based never surprises you |

---

## The 5 Algorithms

### 1. User-Based Collaborative Filtering (`user_cf.py`)
> "Users with similar taste also liked..."

Finds your nearest neighbors (users who listened to similar things), then recommends what they loved that you haven't heard. Uses **Pearson correlation** (mean-centered cosine similarity) to avoid bias toward heavy vs. casual listeners.

**Strength:** Serendipity — can recommend unexpected genres if a neighbor happens to bridge them.
**Weakness:** Cold start (new users have no neighbors), scales poorly to millions of users.

### 2. Item-Based Collaborative Filtering (`item_cf.py`)
> "Because you played X, you might like Y..."

Computes song-to-song similarity from co-listening patterns. More stable than User-CF because song similarities don't change once the song is out. Amazon's original recommendation system used this approach.

Uses **adjusted cosine similarity** — centers by user mean before comparing songs, removing the bias from users who generally rate everything high or low.

**Strength:** Stable, highly explainable, good "more of this" feel.
**Weakness:** Low serendipity — tends to stay in your current genre.

### 3. Matrix Factorization / SVD (`matrix_factorization.py`)
> "Hidden latent taste factors predict your affinity..."

Decomposes the user-item matrix M ≈ U × Σ × Vᵀ using **Truncated SVD**. U gives each user a k-dimensional latent taste vector; Vᵀ gives each song a k-dimensional feature vector. Predicted rating = dot product of the two vectors.

The k=20 latent factors implicitly discover things like "electronic," "acoustic," "tempo preference" without being told what features to look for.

**Strength:** Most accurate in practice, handles sparsity well.
**Weakness:** Less interpretable, cold start problem, needs periodic retraining.

### 4. Content-Based Filtering (`content_based.py`)
> "Songs that match your audio feature profile..."

Builds your taste profile as a weighted average of the audio features of songs you've played (weighted by play count), then recommends songs with the most similar feature vector. Uses **Spotify-style audio features**: energy, danceability, valence, acousticness, instrumentalness, tempo.

**Strength:** Works immediately for new users (just one listened song is enough), no dependence on other users.
**Weakness:** Filter bubble — if you've only heard EDM, you'll only get EDM back.

### 5. Hybrid Ensemble (`hybrid.py`)
> "The best of all four algorithms..."

Normalizes scores from all four algorithms to [0,1] (min-max), then combines with weights:
- User-CF: 25%
- Item-CF: 25%
- SVD: 30%
- Content: 20%

Songs that appear in multiple algorithms' lists naturally get a boost. This is the basic structure of real-world production recommendation systems.

---

## Project Structure

```
SpotifyRecEngine/
├── data/
│   ├── schemas.py          # Pydantic models: Song, User, AudioFeatures
│   └── mock_data.py        # 30 users, 100 songs, listening history
├── algorithms/
│   ├── user_cf.py          # User-Based CF
│   ├── item_cf.py          # Item-Based CF
│   ├── matrix_factorization.py  # SVD
│   ├── content_based.py    # Audio feature similarity
│   └── hybrid.py           # Weighted ensemble
├── engine/
│   └── recommender.py      # RecEngine class — wires everything together
├── api/
│   ├── main.py             # FastAPI app
│   └── routes.py           # /api/* endpoints
└── static/
    ├── index.html           # Single-page UI
    ├── style.css            # Spotify dark theme
    └── app.js               # Vanilla JS frontend
```

---

## Setup & Running

```bash
# 1. Clone and enter the project
cd SpotifyRecEngine

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Start the server (from the SpotifyRecEngine/ directory)
uvicorn api.main:app --reload --port 8000

# 4. Open the UI
open http://localhost:8000/static/index.html
```

The interactive API docs (auto-generated) are at: `http://localhost:8000/docs`

---

## Experimenting & Learning

**Change algorithm weights** in `algorithms/hybrid.py`:
```python
DEFAULT_WEIGHTS = {"user_cf": 0.25, "item_cf": 0.25, "svd": 0.30, "content": 0.20}
```

**Change the number of latent factors** in `algorithms/matrix_factorization.py`:
```python
SVDRecommender(k=5)   # too few → generic
SVDRecommender(k=20)  # sweet spot
SVDRecommender(k=40)  # starts overfitting
```

**Add a new user** to `data/mock_data.py`:
```python
User(user_id="u031", name="Zoe", taste_profile="Metal Fan"),
# Add to LISTEN_MATRIX with some metal songs:
"u031": {"s091": 15, "s093": 12, "s095": 10, "s032": 5},
```

---

## Key Concepts Glossary

**User-Item Matrix** — A table where rows = users, columns = songs, values = play counts. Most entries are 0 (sparse) because no one has heard every song.

**Cosine Similarity** — A measure of angle between two vectors. Used to compare users or songs. Score of 1 = identical direction, 0 = orthogonal (no relationship), -1 = opposite.

**Latent Factors** — Hidden dimensions discovered by matrix factorization. They correspond to real but unlabeled characteristics like "energy level" or "mainstream-ness."

**Cold Start Problem** — The challenge of recommending to a new user (no history) or recommending a new song (no co-play data). Content-based filtering solves the song cold start; onboarding questionnaires solve the user cold start.

**Filter Bubble** — When a recommendation algorithm only shows you things similar to what you already know, limiting discovery. Hybrid systems mitigate this.

**Hybrid System** — A recommendation system that combines multiple algorithms to get the best of each.

---

## Mock Data

- **30 users** across 10 taste archetypes: Pop Lover, EDM Fan, Indie Listener, Hip-Hop Head, Jazz Aficionado, Rock Enthusiast, Classical Fan, Acoustic Chill, Party Mixer, Eclectic Listener
- **100 songs** across 10 genres, each with Spotify-style audio features
- **Listening history**: each user has 10-15 songs in their history, primarily matching their taste archetype with some cross-genre noise — making the recommendation differences between algorithms visible and meaningful
