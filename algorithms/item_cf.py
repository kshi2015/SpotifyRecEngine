"""
item_cf.py — Item-Based Collaborative Filtering

HOW IT WORKS:
  "Find songs that are similar to songs you've already listened to."

  Steps:
    1. Build a user-item matrix (same as User-CF)
    2. Compute item-item similarity using *adjusted cosine similarity*
       (center by user mean before comparing — see below)
    3. For the target user, compute a score for each unheard song:
       score(candidate) = Σ sim(candidate, heard_song) × play_count(heard_song)
    4. Return top-N unheard songs by score

ADJUSTED COSINE SIMILARITY (why not raw cosine?):
  Consider two songs, A and B.
  - User 1 rates both A and B = 5 (loves them)
  - User 2 rates both A = 5, B = 3 (likes A more)
  - User 3 rates both A = 3, B = 5 (likes B more)

  Raw cosine sees A and B as similar (both have high ratings).
  Adjusted cosine subtracts each user's mean rating first:
  - User 1: both become 5 − 4.3 = +0.7 (loves them equally above average)
  - User 2: A = 5−4 = +1, B = 3−4 = −1 (A above average, B below)
  - User 3: A = 3−4 = −1, B = 5−4 = +1 (opposite)
  Now A and B look dissimilar, which is correct.

WHY ITEM-CF IS MORE STABLE THAN USER-CF:
  Song similarities change slowly. "Bohemian Rhapsody" will always be
  similar to other classic rock songs, regardless of which new users
  join the platform. But user similarity is volatile — a user who starts
  exploring new genres can dramatically change who their "neighbors" are.
  This stability makes Item-CF more reliable in production systems.
  Amazon's original recommendation system used Item-CF for this reason.

STRENGTHS:
  - More stable than User-CF (item similarities change slowly)
  - Excellent explainability: "Because you liked X"
  - Scales better: O(n_items²) not O(n_users²), and items << users in practice

WEAKNESSES:
  - Can't recommend items very different from what user already heard
    (less serendipity than User-CF)
  - Cold start for new items: a brand-new song has no co-play history
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from data.schemas import RecommendationResult


class ItemCFRecommender:
    """Item-Based Collaborative Filtering recommender."""

    def __init__(self):
        self._user_item_matrix: Optional[pd.DataFrame] = None
        self._item_similarity: Optional[np.ndarray] = None  # (n_songs, n_songs)
        self._user_index: Dict[str, int] = {}
        self._song_index: Dict[str, int] = {}
        self._M: Optional[np.ndarray] = None

    def fit(self, interaction_df: pd.DataFrame) -> "ItemCFRecommender":
        """
        Build item-item similarity matrix using adjusted cosine similarity.

        Args:
            interaction_df: DataFrame with [user_id, song_id, play_count]
        """
        # Step 1: User × Song matrix (rows=users, cols=songs)
        matrix = interaction_df.pivot_table(
            index="user_id",
            columns="song_id",
            values="play_count",
            fill_value=0,
        )
        self._user_item_matrix = matrix
        self._user_index = {uid: i for i, uid in enumerate(matrix.index)}
        self._song_index = {sid: i for i, sid in enumerate(matrix.columns)}

        M = matrix.values.astype(float)
        self._M = M

        # Step 2: Adjusted cosine — subtract each USER's mean before comparing items
        # This is the key difference from raw cosine:
        # We center by user mean (row mean) rather than item mean (col mean)
        row_means = np.where(M > 0, M, np.nan)
        row_means = np.nanmean(row_means, axis=1, keepdims=True)
        row_means = np.nan_to_num(row_means)
        M_adjusted = np.where(M > 0, M - row_means, 0)

        # Item-item similarity: cosine on COLUMNS (songs) of the adjusted matrix
        # cosine_similarity expects rows = items, so we transpose
        self._item_similarity = cosine_similarity(M_adjusted.T)  # (n_songs, n_songs)
        return self

    def recommend(
        self,
        user_id: str,
        n: int = 10,
        song_titles: Optional[Dict[str, str]] = None,
    ) -> List[RecommendationResult]:
        """
        Recommend top-N songs for a user using Item-Based CF.

        For each unheard song, score = Σ sim(candidate, heard_song) × play_count
        This weights the candidate's similarity by how much the user liked each
        known song — songs the user played 20 times matter more than 1-time plays.

        Args:
            user_id:     Target user ID
            n:           Number of recommendations to return
            song_titles: Optional dict {song_id: title} for explanations
        """
        if self._M is None:
            raise RuntimeError("Call .fit() before .recommend()")
        if user_id not in self._user_index:
            return []

        user_idx = self._user_index[user_id]
        user_row = self._M[user_idx]  # play counts for this user
        heard_mask = user_row > 0

        idx_to_song = {v: k for k, v in self._song_index.items()}

        # Identify which songs the user has heard and their play counts
        heard_indices = np.where(heard_mask)[0]
        heard_play_counts = user_row[heard_indices]

        # For each unheard song: weighted similarity sum across heard songs
        scores: Dict[int, float] = {}
        # Also track which heard song contributed most (for explanation)
        best_contributor: Dict[int, Tuple[int, float]] = {}  # song_idx → (heard_idx, sim)

        n_songs = self._item_similarity.shape[0]
        for candidate_idx in range(n_songs):
            if heard_mask[candidate_idx]:
                continue  # skip songs already heard

            score = 0.0
            best_sim = 0.0
            best_heard_idx = heard_indices[0] if len(heard_indices) > 0 else 0

            for heard_idx, play_count in zip(heard_indices, heard_play_counts):
                sim = self._item_similarity[candidate_idx][heard_idx]
                if sim > 0:
                    score += sim * play_count
                    if sim > best_sim:
                        best_sim = sim
                        best_heard_idx = heard_idx

            if score > 0:
                scores[candidate_idx] = score
                best_contributor[candidate_idx] = (best_heard_idx, best_sim)

        if not scores:
            return []

        results = []
        for song_col_idx, score in sorted(
            scores.items(), key=lambda x: x[1], reverse=True
        )[:n]:
            song_id = idx_to_song[song_col_idx]
            heard_idx, sim = best_contributor[song_col_idx]
            heard_song_id = idx_to_song[heard_idx]
            heard_play_count = int(user_row[heard_idx])

            if song_titles:
                heard_title = song_titles.get(heard_song_id, heard_song_id)
            else:
                heard_title = heard_song_id

            explanation = (
                f"Similar to \"{heard_title}\" which you've played "
                f"{heard_play_count}× (similarity: {sim:.2f})"
            )
            results.append(
                RecommendationResult(
                    song_id=song_id, score=score, explanation=explanation
                )
            )

        return results
