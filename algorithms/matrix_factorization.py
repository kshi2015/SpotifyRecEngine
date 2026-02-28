"""
matrix_factorization.py — Matrix Factorization via Truncated SVD

HOW IT WORKS:
  Collaborative filtering (User-CF and Item-CF) directly compares users and items
  in the original play-count space. Matrix Factorization does something more
  powerful: it learns a compressed representation of both users and songs in a
  shared "latent factor" space.

  Intuition:
    Imagine every song can be described by 20 hidden dimensions — things like
    "how energetic," "how danceable," "how mainstream." No human labels these;
    the algorithm discovers them automatically from listening patterns.
    Similarly, every user gets a 20-dimensional vector that says how much they
    weight each latent factor.
    To predict whether user u would like song s:
        predicted_rating = user_vector[u] · song_vector[s]  (dot product)

  The math — Singular Value Decomposition (SVD):
    Given a user-item matrix M (shape: n_users × n_songs):
        M ≈ U × Σ × Vᵀ
    where:
        U  [n_users × k]  — each user's latent taste vector
        Σ  [k]            — how important each latent factor is (singular values)
        Vᵀ [k × n_songs]  — each song's latent feature vector
        k  (= 20 here)    — the number of latent dimensions

    Reconstructed matrix M̂ = U × Σ × Vᵀ gives predicted ratings for every
    (user, song) pair, including songs the user has never heard!

CHOOSING k (number of latent factors):
  - k = 2: Only captures the broadest genre split (e.g., electronic vs acoustic).
           Under-fits. Recommendations are too generic.
  - k = 20: Captures nuanced sub-genres and taste facets. Good for 100 songs.
  - k = 50: At 100 songs, this starts memorizing noise. Over-fits.

  For Spotify's production system (millions of songs), k is typically in the
  hundreds or thousands. They use ALS (Alternating Least Squares) rather than
  SVD because SVD requires a dense matrix, which is infeasible at scale.

STRENGTHS:
  - Captures latent structure that pure similarity measures miss
  - Can model nuanced taste (not just genre, but tempo, mood, era, etc.)
  - Generally more accurate than User-CF or Item-CF alone
  - Handles sparsity well (most users haven't heard most songs)

WEAKNESSES:
  - Less interpretable than CF (hard to say *why* a song was recommended)
  - Training is more compute-intensive (though SVD is still fast for small data)
  - Cold start: new users or songs have no latent vector until the model retrains
"""

from __future__ import annotations
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

from data.schemas import RecommendationResult


class SVDRecommender:
    """Matrix Factorization recommender using Truncated SVD."""

    def __init__(self, k: int = 20):
        """
        Args:
            k: Number of latent factors. This is the key hyperparameter.
               Increasing k improves expressiveness but risks overfitting.
               For 100 songs and 30 users, k=20 is a good starting point.
        """
        self.k = k
        self._M_hat: Optional[np.ndarray] = None  # reconstructed matrix (n_users × n_songs)
        self._user_index: Dict[str, int] = {}
        self._song_index: Dict[str, int] = {}
        self._M_original: Optional[np.ndarray] = None  # to know what was already heard

    def fit(self, interaction_df: pd.DataFrame) -> "SVDRecommender":
        """
        Factorize the user-item matrix and precompute M̂ = U Σ Vᵀ.

        Args:
            interaction_df: DataFrame with [user_id, song_id, play_count]
        """
        # Step 1: Build the user × song matrix
        matrix = interaction_df.pivot_table(
            index="user_id",
            columns="song_id",
            values="play_count",
            fill_value=0,
        )
        self._user_index = {uid: i for i, uid in enumerate(matrix.index)}
        self._song_index = {sid: i for i, sid in enumerate(matrix.columns)}

        M = matrix.values.astype(float)
        self._M_original = M

        # Step 2: Apply Truncated SVD
        # scipy.sparse.linalg.svds is more memory-efficient than np.linalg.svd
        # (it only computes the top-k singular values instead of all of them)
        # We clamp k to avoid requesting more factors than matrix rank
        actual_k = min(self.k, min(M.shape) - 1)
        U, sigma, Vt = svds(M, k=actual_k)

        # svds returns singular values in ASCENDING order — we reverse to descending
        U = U[:, ::-1]
        sigma = sigma[::-1]
        Vt = Vt[::-1, :]

        # Step 3: Reconstruct M̂ — predicted ratings for all (user, song) pairs
        # M̂[u][s] = predicted play count for user u × song s
        self._M_hat = U @ np.diag(sigma) @ Vt

        return self

    def recommend(
        self, user_id: str, n: int = 10
    ) -> List[RecommendationResult]:
        """
        Recommend top-N songs by looking up the user's row in M̂.

        The reconstructed matrix M̂ gives a predicted "affinity" for every
        (user, song) pair. We sort these predictions, exclude heard songs,
        and return the top-N.

        Args:
            user_id: Target user ID
            n:       Number of recommendations to return
        """
        if self._M_hat is None:
            raise RuntimeError("Call .fit() before .recommend()")
        if user_id not in self._user_index:
            return []

        user_idx = self._user_index[user_id]

        # Get this user's row from the reconstructed matrix
        predicted_ratings = self._M_hat[user_idx].copy()

        # Zero out songs the user has already heard
        heard_mask = self._M_original[user_idx] > 0
        predicted_ratings[heard_mask] = -np.inf

        # Sort by predicted rating and take top-N
        top_indices = np.argsort(predicted_ratings)[::-1][:n]

        idx_to_song = {v: k for k, v in self._song_index.items()}

        results = []
        for song_col_idx in top_indices:
            score = predicted_ratings[song_col_idx]
            if score == -np.inf:
                break
            song_id = idx_to_song[song_col_idx]
            explanation = (
                f"Predicted affinity {score:.2f} based on {self.k} latent "
                f"taste factors discovered from your listening patterns"
            )
            results.append(
                RecommendationResult(
                    song_id=song_id, score=float(score), explanation=explanation
                )
            )

        return results
