"""
schemas.py — Core data types for the Spotify Recommendation Engine.

All data in this project flows through these types. Pydantic models are used
so FastAPI can automatically serialize them to JSON with zero boilerplate.

Key design decision: ListenHistory is NOT stored on the User object.
Instead, it lives in a separate LISTEN_MATRIX dict (in mock_data.py).
This forces the algorithms to build the user-item interaction matrix explicitly,
which is the right mental model for recommendation systems.
"""

from __future__ import annotations
from dataclasses import dataclass
from pydantic import BaseModel


class AudioFeatures(BaseModel):
    """
    Audio features for a song, mimicking Spotify's Audio Features API.

    All values are in [0.0, 1.0] to make cosine similarity meaningful.
    Tempo is normalized by dividing by 250 BPM (a practical maximum).

    These features are what Content-Based Filtering uses to compare songs.
    Collaborative Filtering ignores these entirely — it only uses who listened
    to what, not what the songs sound like.
    """
    energy: float           # 0 = calm/quiet, 1 = loud/energetic
    danceability: float     # 0 = non-danceable, 1 = very danceable
    valence: float          # 0 = sad/dark/angry, 1 = happy/cheerful/euphoric
    acousticness: float     # 0 = electric/produced, 1 = acoustic/unplugged
    instrumentalness: float # 0 = lots of vocals, 1 = purely instrumental
    tempo_normalized: float # BPM / 250; 0.4 ≈ 100 BPM, 0.6 ≈ 150 BPM


class Song(BaseModel):
    """A song with metadata and audio features."""
    song_id: str        # e.g. "s001"
    title: str
    artist: str
    genre: str          # "pop", "edm", "indie", "hip-hop", "jazz", etc.
    features: AudioFeatures


class User(BaseModel):
    """A user with a human-readable taste profile label."""
    user_id: str        # e.g. "u001"
    name: str
    taste_profile: str  # e.g. "Pop Lover", "EDM Fan", "Jazz Aficionado"


@dataclass
class RecommendationResult:
    """
    A single recommendation from any algorithm.

    The explanation field is crucial for learning: it tells you *why*
    this song was recommended, in plain English. Each algorithm generates
    a different kind of explanation, which makes the differences visible.
    """
    song_id: str
    score: float
    explanation: str
