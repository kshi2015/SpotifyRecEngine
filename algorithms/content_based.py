"""
content_based.py — Content-Based Filtering

HOW IT WORKS:
  "Recommend songs whose audio features match your listening taste."

  Unlike collaborative filtering (which only uses who listened to what),
  content-based filtering uses the *properties of the songs themselves*:
  energy, danceability, valence, acousticness, instrumentalness, and tempo.

  Steps:
    1. Build a song × feature matrix F (100 songs × 6 audio features)
    2. Normalize features with StandardScaler (so no one feature dominates)
    3. For the target user, build a "taste profile" vector:
       profile = weighted average of F[heard_song], weighted by play_count
       (songs played 20× matter more than songs played once)
    4. Compute cosine similarity between the profile and every unheard song
    5. Return top-N by similarity

THE USER TASTE PROFILE:
  If Alice has played:
    - "Golden Hour" (pop, high valence, mid energy) × 18
    - "Heartbeat Rush" (pop, high valence, mid energy) × 14
    - "Slow Motion" (r&b, medium valence, medium energy) × 7
  Her profile is: 18/39 × F["Golden Hour"] + 14/39 × F["Heartbeat Rush"] + 7/39 × F["Slow Motion"]
  This gives a centroid in feature space that reflects her overall taste.

FEATURE STANDARDIZATION:
  Without scaling, tempo_normalized might dominate (BPM varies a lot),
  while instrumentalness barely matters (most pop songs have 0.04 or 0.05).
  StandardScaler makes each feature contribute equally to similarity.

STRENGTHS:
  - Cold start for new users: works with as few as 1 played song
  - Cold start for new songs: a brand-new song can be recommended immediately
    based on its audio features (no need for co-play history)
  - No dependence on other users — fully private/personalized
  - Highly interpretable: "matched because energy = 0.9 (your avg: 0.88)"

WEAKNESSES:
  - Filter bubble: only recommends songs *similar* to what you've heard.
    No serendipity. If you've only heard EDM, you'll only get EDM back.
  - Limited by feature quality: if the features don't capture what makes
    songs appealing, recommendations are poor (Spotify has 100+ audio features;
    we use 6 here for clarity)
  - Doesn't leverage community wisdom (unlike CF approaches)
"""

from __future__ import annotations
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from data.schemas import RecommendationResult, Song


# The 6 audio features we use — in order (same order as feature vectors)
FEATURE_NAMES = [
    "energy",
    "danceability",
    "valence",
    "acousticness",
    "instrumentalness",
    "tempo_normalized",
]


class ContentBasedRecommender:
    """Content-Based Filtering recommender using audio feature similarity."""

    def __init__(self, songs: List[Song]):
        """
        Args:
            songs: Full list of Song objects (provides the audio feature matrix)
        """
        self._songs = songs
        self._song_index: Dict[str, int] = {s.song_id: i for i, s in enumerate(songs)}
        self._feature_matrix: Optional[np.ndarray] = None  # (n_songs × 6), scaled
        self._M_original: Optional[np.ndarray] = None  # user × song play counts

    def fit(self, interaction_df: pd.DataFrame) -> "ContentBasedRecommender":
        """
        Build and scale the song feature matrix.
        Also store the user-item matrix to know what each user has heard.

        Args:
            interaction_df: DataFrame with [user_id, song_id, play_count]
        """
        # Step 1: Build the raw song × feature matrix
        raw_features = np.array([
            [
                s.features.energy,
                s.features.danceability,
                s.features.valence,
                s.features.acousticness,
                s.features.instrumentalness,
                s.features.tempo_normalized,
            ]
            for s in self._songs
        ])  # shape: (n_songs, 6)

        # Step 2: Standardize features (mean=0, std=1 across songs)
        # This prevents high-variance features from dominating similarity
        scaler = StandardScaler()
        self._feature_matrix = scaler.fit_transform(raw_features)
        self._scaler = scaler

        # Step 3: Store the user-item matrix to know listening history
        matrix = interaction_df.pivot_table(
            index="user_id",
            columns="song_id",
            values="play_count",
            fill_value=0,
        )
        self._user_index = {uid: i for i, uid in enumerate(matrix.index)}
        self._M_original = matrix.values.astype(float)
        self._matrix_cols = list(matrix.columns)  # song_id ordering in M_original

        return self

    def _build_user_profile(self, user_id: str) -> Optional[np.ndarray]:
        """
        Build a weighted average feature vector representing the user's taste.

        Returns None if the user has no listening history.
        """
        if user_id not in self._user_index:
            return None
        user_idx = self._user_index[user_id]
        user_play_counts = self._M_original[user_idx]  # indexed by matrix_cols order

        # Find which songs this user has heard and their play counts
        weighted_sum = np.zeros(len(FEATURE_NAMES))
        total_weight = 0.0

        for song_col_idx, play_count in enumerate(user_play_counts):
            if play_count > 0:
                song_id = self._matrix_cols[song_col_idx]
                if song_id in self._song_index:
                    song_feature_idx = self._song_index[song_id]
                    weighted_sum += self._feature_matrix[song_feature_idx] * play_count
                    total_weight += play_count

        if total_weight == 0:
            return None
        return weighted_sum / total_weight  # shape: (6,)

    def recommend(
        self, user_id: str, n: int = 10
    ) -> List[RecommendationResult]:
        """
        Recommend top-N songs by audio feature similarity to user taste profile.

        Args:
            user_id: Target user ID
            n:       Number of recommendations to return
        """
        if self._feature_matrix is None:
            raise RuntimeError("Call .fit() before .recommend()")

        # Build this user's taste profile vector
        user_profile = self._build_user_profile(user_id)
        if user_profile is None:
            return []

        # Determine which songs this user has already heard
        user_idx = self._user_index[user_id]
        heard_song_ids = set(
            self._matrix_cols[i]
            for i, pc in enumerate(self._M_original[user_idx])
            if pc > 0
        )

        # Compute cosine similarity between user profile and all songs
        # user_profile shape: (6,) → reshape to (1, 6) for sklearn
        similarities = cosine_similarity(
            user_profile.reshape(1, -1), self._feature_matrix
        )[0]  # shape: (n_songs,)

        # Sort by similarity descending, filter out heard songs
        sorted_indices = np.argsort(similarities)[::-1]

        # Compute user's raw (unscaled) average features for the explanation
        user_raw_profile = self._compute_raw_profile(user_id)

        results = []
        for song_idx in sorted_indices:
            song = self._songs[song_idx]
            if song.song_id in heard_song_ids:
                continue

            sim = similarities[song_idx]
            explanation = self._build_explanation(song, user_raw_profile)

            results.append(
                RecommendationResult(
                    song_id=song.song_id, score=float(sim), explanation=explanation
                )
            )
            if len(results) >= n:
                break

        return results

    def _compute_raw_profile(self, user_id: str) -> Dict[str, float]:
        """Compute unscaled weighted average features for explanation text."""
        if user_id not in self._user_index:
            return {}
        user_idx = self._user_index[user_id]
        user_play_counts = self._M_original[user_idx]

        raw_features_arr = np.array([
            [
                s.features.energy,
                s.features.danceability,
                s.features.valence,
                s.features.acousticness,
                s.features.instrumentalness,
                s.features.tempo_normalized,
            ]
            for s in self._songs
        ])

        weighted_sum = np.zeros(len(FEATURE_NAMES))
        total_weight = 0.0

        for song_col_idx, play_count in enumerate(user_play_counts):
            if play_count > 0:
                song_id = self._matrix_cols[song_col_idx]
                if song_id in self._song_index:
                    song_feat_idx = self._song_index[song_id]
                    weighted_sum += raw_features_arr[song_feat_idx] * play_count
                    total_weight += play_count

        if total_weight == 0:
            return {}
        avg = weighted_sum / total_weight
        return dict(zip(FEATURE_NAMES, avg))

    def _build_explanation(self, song: Song, user_profile: Dict[str, float]) -> str:
        """
        Build an explanation citing the feature with the highest absolute match.
        """
        if not user_profile:
            return "Matches your audio feature profile"

        song_features = {
            "energy": song.features.energy,
            "danceability": song.features.danceability,
            "valence": song.features.valence,
            "acousticness": song.features.acousticness,
            "instrumentalness": song.features.instrumentalness,
            "tempo_normalized": song.features.tempo_normalized,
        }

        # Find the feature where song is closest to user's profile
        best_feature = min(
            FEATURE_NAMES,
            key=lambda f: abs(song_features[f] - user_profile.get(f, 0.5)),
        )
        feature_label = best_feature.replace("_", " ").title()
        song_val = song_features[best_feature]
        user_val = user_profile.get(best_feature, 0.5)

        return (
            f"Matches your taste profile — {feature_label}: "
            f"{song_val:.2f} (your avg: {user_val:.2f})"
        )
