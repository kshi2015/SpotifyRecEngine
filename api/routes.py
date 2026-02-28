"""
routes.py — All /api/* endpoints for the Spotify Rec Engine.

Three endpoints:
  GET /api/users                            — list all 30 users
  GET /api/users/{user_id}/history          — a user's listening history
  GET /api/users/{user_id}/recommendations  — recs from all 5 algorithms
"""

from __future__ import annotations
from typing import Dict, List

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from data.schemas import AudioFeatures, Song, User

router = APIRouter()


# ---------------------------------------------------------------------------
# Response schemas (what the API returns as JSON)
# ---------------------------------------------------------------------------

class UserSummary(BaseModel):
    user_id: str
    name: str
    taste_profile: str


class SongWithCount(BaseModel):
    song: Song
    play_count: int


class SongRecommendation(BaseModel):
    song: Song
    score: float
    explanation: str


class AllRecommendations(BaseModel):
    user_id: str
    user_name: str
    taste_profile: str
    user_cf: List[SongRecommendation]
    item_cf: List[SongRecommendation]
    svd: List[SongRecommendation]
    content: List[SongRecommendation]
    hybrid: List[SongRecommendation]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/users", response_model=List[UserSummary])
def list_users(request: Request) -> List[UserSummary]:
    """Return all 30 users for populating the sidebar."""
    engine = request.app.state.engine
    return [
        UserSummary(
            user_id=u.user_id,
            name=u.name,
            taste_profile=u.taste_profile,
        )
        for u in engine.users
    ]


@router.get("/users/{user_id}/history", response_model=List[SongWithCount])
def get_user_history(user_id: str, request: Request) -> List[SongWithCount]:
    """
    Return a user's listening history sorted by play count.

    The history shows which songs this user has actually played,
    which is what the algorithms use to generate recommendations.
    """
    engine = request.app.state.engine
    if user_id not in engine.users_by_id:
        raise HTTPException(status_code=404, detail=f"User {user_id!r} not found")

    history = engine.get_user_history(user_id)
    return [SongWithCount(song=song, play_count=count) for song, count in history]


@router.get("/users/{user_id}/recommendations", response_model=AllRecommendations)
def get_recommendations(
    user_id: str, request: Request, n: int = 10
) -> AllRecommendations:
    """
    Run all 5 recommendation algorithms for a user and return results.

    Query params:
      n (int, default=10): Number of recommendations per algorithm
    """
    engine = request.app.state.engine
    user = engine.users_by_id.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User {user_id!r} not found")

    all_recs = engine.get_all_recommendations(user_id, n=n)

    def to_response(recs):
        return [
            SongRecommendation(
                song=engine.songs_by_id[r.song_id],
                score=r.score,
                explanation=r.explanation,
            )
            for r in recs
            if r.song_id in engine.songs_by_id
        ]

    return AllRecommendations(
        user_id=user_id,
        user_name=user.name,
        taste_profile=user.taste_profile,
        user_cf=to_response(all_recs["user_cf"]),
        item_cf=to_response(all_recs["item_cf"]),
        svd=to_response(all_recs["svd"]),
        content=to_response(all_recs["content"]),
        hybrid=to_response(all_recs["hybrid"]),
    )
