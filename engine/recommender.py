"""
recommender.py — The RecEngine: single entry point for the API layer.

This class owns all five fitted algorithm objects and exposes a clean
interface that the API routes can call without knowing anything about
the individual algorithms.

Design principle: The API layer should never import from `algorithms/` directly.
All recommendation logic goes through RecEngine.
"""

from __future__ import annotations
from typing import Dict, List, Tuple

from data.mock_data import USERS, SONGS, build_interaction_df
from data.schemas import RecommendationResult, Song, User
from algorithms.user_cf import UserCFRecommender
from algorithms.item_cf import ItemCFRecommender
from algorithms.matrix_factorization import SVDRecommender
from algorithms.content_based import ContentBasedRecommender
from algorithms.hybrid import HybridRecommender


class RecEngine:
    """
    Initializes all five recommendation algorithms from shared mock data.

    Called once at server startup. All models are held in memory (no database).
    At 30 users and 100 songs, everything fits comfortably in RAM and
    all algorithms run in < 1ms per request.
    """

    def __init__(self):
        # Load data
        self.users: List[User] = USERS
        self.songs: List[Song] = SONGS
        self.songs_by_id: Dict[str, Song] = {s.song_id: s for s in SONGS}
        self.users_by_id: Dict[str, User] = {u.user_id: u for u in USERS}

        # Build helper lookup dicts for algorithm explanations
        self._user_names: Dict[str, str] = {u.user_id: u.name for u in USERS}
        self._song_titles: Dict[str, str] = {s.song_id: s.title for s in SONGS}

        # Build the interaction DataFrame (long-format: user_id, song_id, play_count)
        self.interaction_df = build_interaction_df()

        # Fit all five algorithms
        print("Fitting User-CF...", end=" ", flush=True)
        self._user_cf = UserCFRecommender(k_neighbors=10).fit(self.interaction_df)
        print("done")

        print("Fitting Item-CF...", end=" ", flush=True)
        self._item_cf = ItemCFRecommender().fit(self.interaction_df)
        print("done")

        print("Fitting SVD...", end=" ", flush=True)
        self._svd = SVDRecommender(k=20).fit(self.interaction_df)
        print("done")

        print("Fitting Content-Based...", end=" ", flush=True)
        self._content = ContentBasedRecommender(SONGS).fit(self.interaction_df)
        print("done")

        print("Assembling Hybrid...", end=" ", flush=True)
        self._hybrid = HybridRecommender(
            user_cf=self._user_cf,
            item_cf=self._item_cf,
            svd=self._svd,
            content=self._content,
        )
        print("done")
        print("RecEngine ready.")

    def get_user_history(self, user_id: str) -> List[Tuple[Song, int]]:
        """
        Return a list of (Song, play_count) pairs for a user, sorted by play count.

        Args:
            user_id: e.g. "u001"

        Returns:
            List of (Song, play_count), sorted descending by play_count.
            Empty list if user not found or has no history.
        """
        user_rows = self.interaction_df[self.interaction_df["user_id"] == user_id]
        if user_rows.empty:
            return []

        result = []
        for _, row in user_rows.iterrows():
            song = self.songs_by_id.get(row["song_id"])
            if song:
                result.append((song, int(row["play_count"])))

        return sorted(result, key=lambda x: x[1], reverse=True)

    def get_all_recommendations(
        self, user_id: str, n: int = 10
    ) -> Dict[str, List[RecommendationResult]]:
        """
        Run all five algorithms and return their recommendations in one dict.

        Args:
            user_id: Target user ID
            n:       Number of recommendations per algorithm

        Returns:
            Dict with keys: "user_cf", "item_cf", "svd", "content", "hybrid"
            Each value is a list of RecommendationResult, sorted by score.
        """
        return {
            "user_cf": self._user_cf.recommend(
                user_id, n=n, user_names=self._user_names
            ),
            "item_cf": self._item_cf.recommend(
                user_id, n=n, song_titles=self._song_titles
            ),
            "svd": self._svd.recommend(user_id, n=n),
            "content": self._content.recommend(user_id, n=n),
            "hybrid": self._hybrid.recommend(
                user_id,
                n=n,
                user_names=self._user_names,
                song_titles=self._song_titles,
            ),
        }
