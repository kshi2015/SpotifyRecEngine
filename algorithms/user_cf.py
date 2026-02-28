"""
user_cf.py — User-Based Collaborative Filtering

HOW IT WORKS:
  "Find users who listen to similar things as you, then recommend what they like."

  Steps:
    1. Build a user-item matrix: rows = users, columns = songs, values = play counts
    2. Mean-center each user's row (Pearson correlation trick — see below)
    3. For the target user, compute cosine similarity with every other user
    4. Take the top-K most similar users (K=10 by default)
    5. Weighted sum of their play history → scores for all songs
    6. Filter out songs the target user already heard → return top-N

WHY MEAN-CENTERING (Pearson vs raw cosine)?
  Imagine two users:
    - Alice plays everything she hears 1 time (low engagement)
    - Bob plays his favorites 20 times (high engagement)
  If Alice and Bob have the same taste, raw cosine similarity would score them low
  because Alice's play counts are small (her vector has small magnitude).
  Mean-centering removes the "base rate" of each user before comparing, so
  Alice and Bob with identical relative taste end up with high similarity.
  This is equivalent to Pearson correlation on the user vectors.

STRENGTHS:
  - Simple and interpretable
  - Works well when users are opinionated (clear taste profiles)
  - Good "serendipity" — can recommend unexpected things from neighbor taste

WEAKNESSES:
  - Cold start: new users with few listens have weak similarity signals
  - Scales poorly: O(n_users²) similarity computation
  - "Popularity bias": popular songs get recommended to many users

Real-world usage: Item-CF is usually preferred over User-CF at scale
(song similarities are more stable than user similarities), but User-CF
is easier to understand as a teaching example.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from data.schemas import RecommendationResult


class UserCFRecommender:
    """User-Based Collaborative Filtering recommender."""

    def __init__(self, k_neighbors: int = 10):
        """
        Args:
            k_neighbors: Number of most-similar users to use for scoring.
                         More neighbors → smoother scores, less personalized.
                         Fewer neighbors → more personalized but noisier.
        """
        self.k = k_neighbors
        self._user_item_matrix: Optional[pd.DataFrame] = None  # rows=users, cols=songs
        self._user_index: Dict[str, int] = {}   # user_id → row index
        self._song_index: Dict[str, int] = {}   # song_id → col index
        self._similarity_matrix: Optional[np.ndarray] = None

    def fit(self, interaction_df: pd.DataFrame) -> "UserCFRecommender":
        """
        Build the user-item matrix and precompute user-user similarities.

        Args:
            interaction_df: DataFrame with columns [user_id, song_id, play_count]

        Returns:
            self (for chaining: UserCFRecommender().fit(df))
        """
        # Step 1: Pivot long-format interactions into a user × song matrix
        # Missing entries (no plays) are filled with 0
        matrix = interaction_df.pivot_table(
            index="user_id",
            columns="song_id",
            values="play_count",
            fill_value=0,
        )
        self._user_item_matrix = matrix
        self._user_index = {uid: i for i, uid in enumerate(matrix.index)}
        self._song_index = {sid: i for i, sid in enumerate(matrix.columns)}

        # Step 2: Mean-center each user's row (Pearson correlation trick)
        # This removes per-user "bias" (some users rate everything high, some low)
        M = matrix.values.astype(float)
        # Only subtract mean from non-zero entries (don't penalize unheard songs)
        row_means = np.where(M > 0, M, np.nan)
        row_means = np.nanmean(row_means, axis=1, keepdims=True)
        row_means = np.nan_to_num(row_means)  # users with no history → mean = 0
        M_centered = np.where(M > 0, M - row_means, 0)

        # Step 3: Compute cosine similarity between all user pairs
        # Shape: (n_users, n_users), sim[i][j] ∈ [−1, 1]
        self._similarity_matrix = cosine_similarity(M_centered)
        self._M = M  # store original (non-centered) for scoring
        return self

    def recommend(
        self,
        user_id: str,
        n: int = 10,
        user_names: Optional[Dict[str, str]] = None,
    ) -> List[RecommendationResult]:
        """
        Recommend top-N songs for a user using User-Based CF.

        Args:
            user_id:    Target user ID (e.g. "u001")
            n:          Number of recommendations to return
            user_names: Optional dict {user_id: name} for richer explanations

        Returns:
            List of RecommendationResult, sorted by score descending
        """
        if self._user_item_matrix is None:
            raise RuntimeError("Call .fit() before .recommend()")
        if user_id not in self._user_index:
            return []

        user_idx = self._user_index[user_id]

        # Step 4: Find top-K similar users (excluding the target user itself)
        similarities = self._similarity_matrix[user_idx].copy()
        similarities[user_idx] = -1  # exclude self

        # Argsort descending → indices of most similar users
        top_k_indices = np.argsort(similarities)[::-1][: self.k]
        top_k_sims = similarities[top_k_indices]

        # Step 5: For each song the target user has NOT heard,
        # compute a weighted score = sum of (similarity × play_count) across neighbors
        user_row = self._M[user_idx]
        heard_mask = user_row > 0  # True for already-heard songs

        scores: Dict[int, float] = {}
        neighbor_counts: Dict[int, int] = {}  # how many neighbors liked this song

        for rank, (neighbor_idx, sim) in enumerate(
            zip(top_k_indices, top_k_sims)
        ):
            if sim <= 0:
                continue  # only use positively correlated neighbors
            neighbor_row = self._M[neighbor_idx]
            for song_col_idx, play_count in enumerate(neighbor_row):
                if play_count > 0 and not heard_mask[song_col_idx]:
                    scores[song_col_idx] = (
                        scores.get(song_col_idx, 0.0) + sim * play_count
                    )
                    neighbor_counts[song_col_idx] = (
                        neighbor_counts.get(song_col_idx, 0) + 1
                    )

        if not scores:
            return []

        # Step 6: Sort by score, return top-N with explanations
        idx_to_song = {v: k for k, v in self._song_index.items()}
        idx_to_user = {v: k for k, v in self._user_index.items()}

        results = []
        for song_col_idx, score in sorted(
            scores.items(), key=lambda x: x[1], reverse=True
        )[:n]:
            song_id = idx_to_song[song_col_idx]
            n_neighbors = neighbor_counts[song_col_idx]

            # Build explanation using the most similar neighbor who liked this song
            most_similar_neighbor_uid = idx_to_user[top_k_indices[0]]
            if user_names:
                neighbor_name = user_names.get(
                    most_similar_neighbor_uid, most_similar_neighbor_uid
                )
            else:
                neighbor_name = most_similar_neighbor_uid

            explanation = (
                f"Liked by {n_neighbors} user(s) with similar taste "
                f"(e.g. {neighbor_name})"
            )
            results.append(
                RecommendationResult(
                    song_id=song_id, score=score, explanation=explanation
                )
            )

        return results
