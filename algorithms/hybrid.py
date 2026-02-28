"""
hybrid.py — Hybrid Recommendation System (Weighted Ensemble)

HOW IT WORKS:
  "Combine scores from all four algorithms into a single ranked list."

  Each algorithm has different strengths:
    - User-CF:  Good for serendipity, community wisdom
    - Item-CF:  Good for "more like this", stable and explainable
    - SVD:      Good for latent structure, generally most accurate
    - Content:  Good for cold start, pure taste matching

  A hybrid system lets us get the best of all four.

THE NORMALIZATION PROBLEM:
  Before combining scores, we must normalize them to the same range.
  Why? Each algorithm produces scores in a different scale:
    - User-CF:  scores ~ 10-500 (weighted sum of play counts × similarities)
    - Item-CF:  scores ~ 5-200 (similar)
    - SVD:      scores ~ 0.1-5 (reconstructed rating)
    - Content:  scores ~ 0.7-0.99 (cosine similarity, bounded by [0,1])

  If we just average these raw scores, SVD would be drowned out by CF.
  Solution: min-max normalize each algorithm's scores to [0, 1] first,
  then apply weights, then sum.

WEIGHTS (tunable):
    User-CF:  0.25
    Item-CF:  0.25
    SVD:      0.30  ← slightly higher because MF is generally most accurate
    Content:  0.20  ← lower because it's most likely to create filter bubbles

  Songs that appear in multiple algorithms' lists get a natural boost.
  Songs that only appear in one list can still make it if their normalized
  score in that algorithm is very high.

REAL-WORLD HYBRID SYSTEMS:
  Spotify's actual system is much more complex:
  - Multiple matrix factorization models (ALS for implicit feedback)
  - Deep learning on audio features (audio CNNs on Mel spectrograms)
  - NLP on playlists (word2vec-style models treating songs as "words")
  - Contextual signals (time of day, device, recent skip rate)
  - Reinforcement learning for long-term engagement
  But the core idea of combining multiple signals is the same.
"""

from __future__ import annotations
from typing import Dict, List

import numpy as np

from data.schemas import RecommendationResult
from algorithms.user_cf import UserCFRecommender
from algorithms.item_cf import ItemCFRecommender
from algorithms.matrix_factorization import SVDRecommender
from algorithms.content_based import ContentBasedRecommender

DEFAULT_WEIGHTS = {
    "user_cf": 0.25,
    "item_cf": 0.25,
    "svd": 0.30,
    "content": 0.20,
}


def _normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """
    Min-max normalize a dict of {song_id: score} to [0, 1].
    Returns an empty dict if all scores are identical.
    """
    if not scores:
        return {}
    values = list(scores.values())
    min_v, max_v = min(values), max(values)
    if max_v == min_v:
        return {k: 0.5 for k in scores}
    return {k: (v - min_v) / (max_v - min_v) for k, v in scores.items()}


class HybridRecommender:
    """Weighted ensemble of User-CF, Item-CF, SVD, and Content-Based."""

    def __init__(
        self,
        user_cf: UserCFRecommender,
        item_cf: ItemCFRecommender,
        svd: SVDRecommender,
        content: ContentBasedRecommender,
        weights: Dict[str, float] = DEFAULT_WEIGHTS,
    ):
        """
        Args:
            user_cf:  Fitted UserCFRecommender
            item_cf:  Fitted ItemCFRecommender
            svd:      Fitted SVDRecommender
            content:  Fitted ContentBasedRecommender
            weights:  How much to weight each algorithm (should sum to 1.0)
        """
        self._user_cf = user_cf
        self._item_cf = item_cf
        self._svd = svd
        self._content = content
        self._weights = weights

    def recommend(
        self,
        user_id: str,
        n: int = 10,
        user_names: Dict[str, str] = None,
        song_titles: Dict[str, str] = None,
        candidate_n: int = 50,
    ) -> List[RecommendationResult]:
        """
        Recommend top-N songs by combining all four algorithm scores.

        Args:
            user_id:     Target user ID
            n:           Final number of recommendations to return
            user_names:  {user_id: name} for User-CF explanations
            song_titles: {song_id: title} for Item-CF explanations
            candidate_n: Number of candidates to request from each algorithm
                         (a larger pool gives the ensemble more to work with)
        """
        # Step 1: Collect raw recommendations from each algorithm
        user_cf_recs = self._user_cf.recommend(user_id, n=candidate_n, user_names=user_names)
        item_cf_recs = self._item_cf.recommend(user_id, n=candidate_n, song_titles=song_titles)
        svd_recs = self._svd.recommend(user_id, n=candidate_n)
        content_recs = self._content.recommend(user_id, n=candidate_n)

        # Step 2: Convert each recommendation list to {song_id: score} dicts
        user_cf_scores = {r.song_id: r.score for r in user_cf_recs}
        item_cf_scores = {r.song_id: r.score for r in item_cf_recs}
        svd_scores = {r.song_id: r.score for r in svd_recs}
        content_scores = {r.song_id: r.score for r in content_recs}

        # Step 3: Min-max normalize each algorithm's scores to [0, 1]
        user_cf_norm = _normalize(user_cf_scores)
        item_cf_norm = _normalize(item_cf_scores)
        svd_norm = _normalize(svd_scores)
        content_norm = _normalize(content_scores)

        # Step 4: Combine — all songs from any algorithm are candidates
        all_songs = (
            set(user_cf_norm)
            | set(item_cf_norm)
            | set(svd_norm)
            | set(content_norm)
        )

        combined: Dict[str, float] = {}
        for song_id in all_songs:
            combined[song_id] = (
                self._weights["user_cf"] * user_cf_norm.get(song_id, 0.0)
                + self._weights["item_cf"] * item_cf_norm.get(song_id, 0.0)
                + self._weights["svd"] * svd_norm.get(song_id, 0.0)
                + self._weights["content"] * content_norm.get(song_id, 0.0)
            )

        # Step 5: Sort by combined score, return top-N with multi-signal explanation
        results = []
        for song_id, score in sorted(
            combined.items(), key=lambda x: x[1], reverse=True
        )[:n]:
            cf_score = user_cf_norm.get(song_id, 0.0)
            item_score = item_cf_norm.get(song_id, 0.0)
            svd_score = svd_norm.get(song_id, 0.0)
            cb_score = content_norm.get(song_id, 0.0)

            explanation = (
                f"Hybrid score {score:.2f} — "
                f"User-CF: {cf_score:.2f}, "
                f"Item-CF: {item_score:.2f}, "
                f"SVD: {svd_score:.2f}, "
                f"Content: {cb_score:.2f}"
            )
            results.append(
                RecommendationResult(
                    song_id=song_id, score=score, explanation=explanation
                )
            )

        return results
