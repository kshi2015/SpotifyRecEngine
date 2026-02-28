"""
mock_data.py — Deterministic mock dataset for the Spotify Recommendation Engine.

Contains 30 users across 10 taste archetypes, 100 songs across 10 genres,
and a realistic sparse listening history.

Why hardcoded instead of randomly generated?
  Random data produces different results every run, making it impossible to
  understand *why* an algorithm made a recommendation. Hardcoded data means
  every recommendation is traceable back to the data that caused it.

Structure exported by this module:
  USERS            — List[User], 30 users
  SONGS            — List[Song], 100 songs
  LISTEN_MATRIX    — Dict[user_id, Dict[song_id, play_count]], sparse
  build_interaction_df() — returns a pandas DataFrame for algorithm input
"""

from __future__ import annotations
from typing import Dict, List
import pandas as pd

from data.schemas import AudioFeatures, Song, User

# ---------------------------------------------------------------------------
# Genre audio feature centroids
# Each song's features are these centroid values ± small hardcoded offsets.
# This makes songs within a genre sound similar but not identical.
# ---------------------------------------------------------------------------
#
# Genre     | energy | dance | valence | acoustic | instrum | tempo_norm
# ----------|--------|-------|---------|----------|---------|----------
# edm       | 0.92   | 0.88  | 0.65    | 0.04     | 0.12    | 0.72
# pop       | 0.70   | 0.78  | 0.75    | 0.22     | 0.05    | 0.58
# hip-hop   | 0.72   | 0.82  | 0.55    | 0.15     | 0.08    | 0.60
# rock      | 0.85   | 0.52  | 0.45    | 0.10     | 0.15    | 0.62
# indie     | 0.55   | 0.60  | 0.60    | 0.48     | 0.10    | 0.50
# acoustic  | 0.35   | 0.48  | 0.72    | 0.82     | 0.05    | 0.40
# jazz      | 0.40   | 0.55  | 0.65    | 0.68     | 0.60    | 0.45
# classical | 0.25   | 0.22  | 0.50    | 0.88     | 0.95    | 0.30
# r&b       | 0.62   | 0.75  | 0.68    | 0.30     | 0.05    | 0.52
# metal     | 0.96   | 0.35  | 0.25    | 0.05     | 0.25    | 0.74

# ---------------------------------------------------------------------------
# 100 Songs — 10 per genre, with realistic audio features
# ---------------------------------------------------------------------------

SONGS: List[Song] = [
    # --- EDM (s001–s010) ---
    Song(song_id="s001", title="Midnight Pulse", artist="Neon Cascade", genre="edm",
         features=AudioFeatures(energy=0.95, danceability=0.90, valence=0.68, acousticness=0.03, instrumentalness=0.15, tempo_normalized=0.74)),
    Song(song_id="s002", title="Electric Dawn", artist="Synthwave City", genre="edm",
         features=AudioFeatures(energy=0.91, danceability=0.86, valence=0.62, acousticness=0.05, instrumentalness=0.10, tempo_normalized=0.70)),
    Song(song_id="s003", title="Bass Drop", artist="DJ Horizon", genre="edm",
         features=AudioFeatures(energy=0.94, danceability=0.92, valence=0.70, acousticness=0.02, instrumentalness=0.20, tempo_normalized=0.76)),
    Song(song_id="s004", title="Neon Fever", artist="Pulse Engine", genre="edm",
         features=AudioFeatures(energy=0.89, danceability=0.85, valence=0.60, acousticness=0.06, instrumentalness=0.08, tempo_normalized=0.68)),
    Song(song_id="s005", title="Rave Nation", artist="Neon Cascade", genre="edm",
         features=AudioFeatures(energy=0.93, danceability=0.91, valence=0.67, acousticness=0.03, instrumentalness=0.18, tempo_normalized=0.73)),
    Song(song_id="s006", title="Ultraviolet", artist="Synthwave City", genre="edm",
         features=AudioFeatures(energy=0.90, danceability=0.87, valence=0.63, acousticness=0.04, instrumentalness=0.12, tempo_normalized=0.71)),
    Song(song_id="s007", title="Grid Runner", artist="DJ Horizon", genre="edm",
         features=AudioFeatures(energy=0.96, danceability=0.89, valence=0.66, acousticness=0.02, instrumentalness=0.22, tempo_normalized=0.75)),
    Song(song_id="s008", title="Flux", artist="Pulse Engine", genre="edm",
         features=AudioFeatures(energy=0.88, danceability=0.84, valence=0.58, acousticness=0.07, instrumentalness=0.09, tempo_normalized=0.69)),
    Song(song_id="s009", title="Cyber Rush", artist="Neon Cascade", genre="edm",
         features=AudioFeatures(energy=0.92, danceability=0.88, valence=0.64, acousticness=0.04, instrumentalness=0.14, tempo_normalized=0.72)),
    Song(song_id="s010", title="Aurora Protocol", artist="Synthwave City", genre="edm",
         features=AudioFeatures(energy=0.87, danceability=0.83, valence=0.61, acousticness=0.06, instrumentalness=0.11, tempo_normalized=0.70)),

    # --- Pop (s011–s020) ---
    Song(song_id="s011", title="Golden Hour", artist="Luna Blake", genre="pop",
         features=AudioFeatures(energy=0.72, danceability=0.80, valence=0.78, acousticness=0.20, instrumentalness=0.04, tempo_normalized=0.60)),
    Song(song_id="s012", title="Heartbeat Rush", artist="The Sparks", genre="pop",
         features=AudioFeatures(energy=0.68, danceability=0.76, valence=0.72, acousticness=0.25, instrumentalness=0.03, tempo_normalized=0.56)),
    Song(song_id="s013", title="Shimmer", artist="Stella Rose", genre="pop",
         features=AudioFeatures(energy=0.74, danceability=0.82, valence=0.80, acousticness=0.18, instrumentalness=0.06, tempo_normalized=0.62)),
    Song(song_id="s014", title="Summer Glow", artist="Luna Blake", genre="pop",
         features=AudioFeatures(energy=0.66, danceability=0.74, valence=0.74, acousticness=0.28, instrumentalness=0.02, tempo_normalized=0.54)),
    Song(song_id="s015", title="Neon Lights", artist="The Sparks", genre="pop",
         features=AudioFeatures(energy=0.71, danceability=0.79, valence=0.77, acousticness=0.22, instrumentalness=0.05, tempo_normalized=0.59)),
    Song(song_id="s016", title="Dreamer", artist="Stella Rose", genre="pop",
         features=AudioFeatures(energy=0.69, danceability=0.77, valence=0.73, acousticness=0.24, instrumentalness=0.04, tempo_normalized=0.57)),
    Song(song_id="s017", title="Rise Up", artist="Luna Blake", genre="pop",
         features=AudioFeatures(energy=0.75, danceability=0.81, valence=0.79, acousticness=0.19, instrumentalness=0.05, tempo_normalized=0.61)),
    Song(song_id="s018", title="Cascades", artist="The Sparks", genre="pop",
         features=AudioFeatures(energy=0.67, danceability=0.75, valence=0.71, acousticness=0.26, instrumentalness=0.03, tempo_normalized=0.55)),
    Song(song_id="s019", title="Vertigo", artist="Stella Rose", genre="pop",
         features=AudioFeatures(energy=0.73, danceability=0.80, valence=0.76, acousticness=0.21, instrumentalness=0.06, tempo_normalized=0.60)),
    Song(song_id="s020", title="Afterglow", artist="Luna Blake", genre="pop",
         features=AudioFeatures(energy=0.70, danceability=0.78, valence=0.74, acousticness=0.23, instrumentalness=0.04, tempo_normalized=0.58)),

    # --- Hip-Hop (s021–s030) ---
    Song(song_id="s021", title="Block Party", artist="King Verse", genre="hip-hop",
         features=AudioFeatures(energy=0.74, danceability=0.85, valence=0.58, acousticness=0.12, instrumentalness=0.06, tempo_normalized=0.62)),
    Song(song_id="s022", title="Crown Heights", artist="Metro Flow", genre="hip-hop",
         features=AudioFeatures(energy=0.70, danceability=0.80, valence=0.52, acousticness=0.18, instrumentalness=0.05, tempo_normalized=0.58)),
    Song(song_id="s023", title="Grind Season", artist="King Verse", genre="hip-hop",
         features=AudioFeatures(energy=0.76, danceability=0.84, valence=0.56, acousticness=0.10, instrumentalness=0.08, tempo_normalized=0.64)),
    Song(song_id="s024", title="Midnight Grind", artist="Apex MC", genre="hip-hop",
         features=AudioFeatures(energy=0.68, danceability=0.78, valence=0.50, acousticness=0.20, instrumentalness=0.04, tempo_normalized=0.56)),
    Song(song_id="s025", title="City Moves", artist="Metro Flow", genre="hip-hop",
         features=AudioFeatures(energy=0.73, danceability=0.83, valence=0.57, acousticness=0.14, instrumentalness=0.07, tempo_normalized=0.61)),
    Song(song_id="s026", title="Flex Zone", artist="King Verse", genre="hip-hop",
         features=AudioFeatures(energy=0.75, danceability=0.86, valence=0.60, acousticness=0.11, instrumentalness=0.09, tempo_normalized=0.63)),
    Song(song_id="s027", title="Uptown", artist="Apex MC", genre="hip-hop",
         features=AudioFeatures(energy=0.71, danceability=0.81, valence=0.54, acousticness=0.16, instrumentalness=0.06, tempo_normalized=0.59)),
    Song(song_id="s028", title="No Ceiling", artist="Metro Flow", genre="hip-hop",
         features=AudioFeatures(energy=0.77, danceability=0.83, valence=0.55, acousticness=0.13, instrumentalness=0.07, tempo_normalized=0.62)),
    Song(song_id="s029", title="Street Gospel", artist="King Verse", genre="hip-hop",
         features=AudioFeatures(energy=0.69, danceability=0.79, valence=0.51, acousticness=0.19, instrumentalness=0.05, tempo_normalized=0.57)),
    Song(song_id="s030", title="Hustle Hard", artist="Apex MC", genre="hip-hop",
         features=AudioFeatures(energy=0.74, danceability=0.84, valence=0.58, acousticness=0.12, instrumentalness=0.08, tempo_normalized=0.62)),

    # --- Rock (s031–s040) ---
    Song(song_id="s031", title="Thunder Road", artist="Iron Veil", genre="rock",
         features=AudioFeatures(energy=0.87, danceability=0.54, valence=0.47, acousticness=0.08, instrumentalness=0.17, tempo_normalized=0.64)),
    Song(song_id="s032", title="Shattered Glass", artist="The Riff Kings", genre="rock",
         features=AudioFeatures(energy=0.83, danceability=0.50, valence=0.43, acousticness=0.12, instrumentalness=0.13, tempo_normalized=0.60)),
    Song(song_id="s033", title="Highway Burn", artist="Iron Veil", genre="rock",
         features=AudioFeatures(energy=0.89, danceability=0.56, valence=0.49, acousticness=0.07, instrumentalness=0.19, tempo_normalized=0.66)),
    Song(song_id="s034", title="Raw Power", artist="Granite Wave", genre="rock",
         features=AudioFeatures(energy=0.85, danceability=0.52, valence=0.45, acousticness=0.10, instrumentalness=0.15, tempo_normalized=0.62)),
    Song(song_id="s035", title="Overdrive", artist="The Riff Kings", genre="rock",
         features=AudioFeatures(energy=0.82, danceability=0.48, valence=0.42, acousticness=0.13, instrumentalness=0.12, tempo_normalized=0.59)),
    Song(song_id="s036", title="Steel & Fire", artist="Iron Veil", genre="rock",
         features=AudioFeatures(energy=0.88, danceability=0.55, valence=0.48, acousticness=0.09, instrumentalness=0.16, tempo_normalized=0.65)),
    Song(song_id="s037", title="Last Stand", artist="Granite Wave", genre="rock",
         features=AudioFeatures(energy=0.84, danceability=0.51, valence=0.44, acousticness=0.11, instrumentalness=0.14, tempo_normalized=0.61)),
    Song(song_id="s038", title="Rogue Signal", artist="The Riff Kings", genre="rock",
         features=AudioFeatures(energy=0.86, danceability=0.53, valence=0.46, acousticness=0.09, instrumentalness=0.18, tempo_normalized=0.63)),
    Song(song_id="s039", title="Broken Crown", artist="Iron Veil", genre="rock",
         features=AudioFeatures(energy=0.81, danceability=0.49, valence=0.41, acousticness=0.14, instrumentalness=0.11, tempo_normalized=0.58)),
    Song(song_id="s040", title="Wildfire", artist="Granite Wave", genre="rock",
         features=AudioFeatures(energy=0.90, danceability=0.57, valence=0.50, acousticness=0.06, instrumentalness=0.20, tempo_normalized=0.67)),

    # --- Indie (s041–s050) ---
    Song(song_id="s041", title="Paper Planes", artist="Silver Lining", genre="indie",
         features=AudioFeatures(energy=0.57, danceability=0.62, valence=0.63, acousticness=0.46, instrumentalness=0.09, tempo_normalized=0.52)),
    Song(song_id="s042", title="Wandering Eye", artist="Moth & Flame", genre="indie",
         features=AudioFeatures(energy=0.53, danceability=0.58, valence=0.58, acousticness=0.51, instrumentalness=0.08, tempo_normalized=0.48)),
    Song(song_id="s043", title="Overgrown", artist="Silver Lining", genre="indie",
         features=AudioFeatures(energy=0.59, danceability=0.64, valence=0.65, acousticness=0.44, instrumentalness=0.11, tempo_normalized=0.54)),
    Song(song_id="s044", title="Dusk", artist="The Velvet Hours", genre="indie",
         features=AudioFeatures(energy=0.51, danceability=0.56, valence=0.56, acousticness=0.53, instrumentalness=0.07, tempo_normalized=0.46)),
    Song(song_id="s045", title="Echo Chamber", artist="Moth & Flame", genre="indie",
         features=AudioFeatures(energy=0.56, danceability=0.61, valence=0.61, acousticness=0.49, instrumentalness=0.10, tempo_normalized=0.51)),
    Song(song_id="s046", title="Fluorescent", artist="Silver Lining", genre="indie",
         features=AudioFeatures(energy=0.58, danceability=0.63, valence=0.64, acousticness=0.47, instrumentalness=0.09, tempo_normalized=0.53)),
    Song(song_id="s047", title="Glass Bones", artist="The Velvet Hours", genre="indie",
         features=AudioFeatures(energy=0.52, danceability=0.57, valence=0.57, acousticness=0.52, instrumentalness=0.08, tempo_normalized=0.47)),
    Song(song_id="s048", title="Northern Lights", artist="Moth & Flame", genre="indie",
         features=AudioFeatures(energy=0.60, danceability=0.65, valence=0.66, acousticness=0.43, instrumentalness=0.12, tempo_normalized=0.55)),
    Song(song_id="s049", title="Pale Blue", artist="Silver Lining", genre="indie",
         features=AudioFeatures(energy=0.54, danceability=0.59, valence=0.59, acousticness=0.50, instrumentalness=0.07, tempo_normalized=0.49)),
    Song(song_id="s050", title="Salt & Sea", artist="The Velvet Hours", genre="indie",
         features=AudioFeatures(energy=0.55, danceability=0.60, valence=0.62, acousticness=0.48, instrumentalness=0.10, tempo_normalized=0.50)),

    # --- Acoustic (s051–s060) ---
    Song(song_id="s051", title="Morning Coffee", artist="River & Reed", genre="acoustic",
         features=AudioFeatures(energy=0.37, danceability=0.50, valence=0.74, acousticness=0.84, instrumentalness=0.04, tempo_normalized=0.42)),
    Song(song_id="s052", title="Porch Song", artist="Hollow Pines", genre="acoustic",
         features=AudioFeatures(energy=0.33, danceability=0.46, valence=0.70, acousticness=0.88, instrumentalness=0.03, tempo_normalized=0.38)),
    Song(song_id="s053", title="Bare Feet", artist="River & Reed", genre="acoustic",
         features=AudioFeatures(energy=0.39, danceability=0.52, valence=0.76, acousticness=0.82, instrumentalness=0.05, tempo_normalized=0.44)),
    Song(song_id="s054", title="Still Water", artist="The Wanderers", genre="acoustic",
         features=AudioFeatures(energy=0.31, danceability=0.44, valence=0.68, acousticness=0.90, instrumentalness=0.02, tempo_normalized=0.36)),
    Song(song_id="s055", title="Wildflower", artist="Hollow Pines", genre="acoustic",
         features=AudioFeatures(energy=0.36, danceability=0.49, valence=0.73, acousticness=0.85, instrumentalness=0.04, tempo_normalized=0.41)),
    Song(song_id="s056", title="Ember Days", artist="River & Reed", genre="acoustic",
         features=AudioFeatures(energy=0.38, danceability=0.51, valence=0.75, acousticness=0.83, instrumentalness=0.05, tempo_normalized=0.43)),
    Song(song_id="s057", title="Dusty Roads", artist="The Wanderers", genre="acoustic",
         features=AudioFeatures(energy=0.32, danceability=0.45, valence=0.69, acousticness=0.89, instrumentalness=0.03, tempo_normalized=0.37)),
    Song(song_id="s058", title="Old Cabin", artist="Hollow Pines", genre="acoustic",
         features=AudioFeatures(energy=0.34, danceability=0.47, valence=0.71, acousticness=0.87, instrumentalness=0.02, tempo_normalized=0.39)),
    Song(song_id="s059", title="Sundown", artist="River & Reed", genre="acoustic",
         features=AudioFeatures(energy=0.40, danceability=0.53, valence=0.77, acousticness=0.81, instrumentalness=0.06, tempo_normalized=0.45)),
    Song(song_id="s060", title="Lantern Light", artist="The Wanderers", genre="acoustic",
         features=AudioFeatures(energy=0.35, danceability=0.48, valence=0.72, acousticness=0.86, instrumentalness=0.03, tempo_normalized=0.40)),

    # --- Jazz (s061–s070) ---
    Song(song_id="s061", title="Blue Smoke", artist="The Cool Quartet", genre="jazz",
         features=AudioFeatures(energy=0.42, danceability=0.57, valence=0.67, acousticness=0.70, instrumentalness=0.62, tempo_normalized=0.47)),
    Song(song_id="s062", title="Midnight Club", artist="Vera Miles Trio", genre="jazz",
         features=AudioFeatures(energy=0.38, danceability=0.53, valence=0.63, acousticness=0.74, instrumentalness=0.58, tempo_normalized=0.43)),
    Song(song_id="s063", title="Brushwork", artist="The Cool Quartet", genre="jazz",
         features=AudioFeatures(energy=0.44, danceability=0.59, valence=0.69, acousticness=0.68, instrumentalness=0.64, tempo_normalized=0.49)),
    Song(song_id="s064", title="Slow Burn", artist="Harlem Sounds", genre="jazz",
         features=AudioFeatures(energy=0.36, danceability=0.51, valence=0.61, acousticness=0.76, instrumentalness=0.56, tempo_normalized=0.41)),
    Song(song_id="s065", title="Bebop Kitchen", artist="Vera Miles Trio", genre="jazz",
         features=AudioFeatures(energy=0.41, danceability=0.56, valence=0.66, acousticness=0.71, instrumentalness=0.61, tempo_normalized=0.46)),
    Song(song_id="s066", title="Smoke & Keys", artist="The Cool Quartet", genre="jazz",
         features=AudioFeatures(energy=0.43, danceability=0.58, valence=0.68, acousticness=0.69, instrumentalness=0.63, tempo_normalized=0.48)),
    Song(song_id="s067", title="East Village", artist="Harlem Sounds", genre="jazz",
         features=AudioFeatures(energy=0.37, danceability=0.52, valence=0.62, acousticness=0.75, instrumentalness=0.57, tempo_normalized=0.42)),
    Song(song_id="s068", title="Upright Walk", artist="Vera Miles Trio", genre="jazz",
         features=AudioFeatures(energy=0.45, danceability=0.60, valence=0.70, acousticness=0.67, instrumentalness=0.65, tempo_normalized=0.50)),
    Song(song_id="s069", title="Minor Third", artist="The Cool Quartet", genre="jazz",
         features=AudioFeatures(energy=0.39, danceability=0.54, valence=0.64, acousticness=0.73, instrumentalness=0.59, tempo_normalized=0.44)),
    Song(song_id="s070", title="After Hours", artist="Harlem Sounds", genre="jazz",
         features=AudioFeatures(energy=0.40, danceability=0.55, valence=0.65, acousticness=0.72, instrumentalness=0.60, tempo_normalized=0.45)),

    # --- Classical (s071–s080) ---
    Song(song_id="s071", title="Adagio in Blue", artist="Vienna String Collective", genre="classical",
         features=AudioFeatures(energy=0.27, danceability=0.24, valence=0.52, acousticness=0.90, instrumentalness=0.97, tempo_normalized=0.32)),
    Song(song_id="s072", title="Nocturne No. 3", artist="The Chamber Players", genre="classical",
         features=AudioFeatures(energy=0.23, danceability=0.20, valence=0.48, acousticness=0.94, instrumentalness=0.93, tempo_normalized=0.28)),
    Song(song_id="s073", title="Prelude in Rain", artist="Vienna String Collective", genre="classical",
         features=AudioFeatures(energy=0.29, danceability=0.26, valence=0.54, acousticness=0.88, instrumentalness=0.99, tempo_normalized=0.34)),
    Song(song_id="s074", title="Sonata Movement", artist="Baroque Revival", genre="classical",
         features=AudioFeatures(energy=0.21, danceability=0.18, valence=0.46, acousticness=0.96, instrumentalness=0.91, tempo_normalized=0.26)),
    Song(song_id="s075", title="Fugue in E Minor", artist="The Chamber Players", genre="classical",
         features=AudioFeatures(energy=0.26, danceability=0.23, valence=0.51, acousticness=0.91, instrumentalness=0.96, tempo_normalized=0.31)),
    Song(song_id="s076", title="Waltz for Two", artist="Vienna String Collective", genre="classical",
         features=AudioFeatures(energy=0.28, danceability=0.25, valence=0.53, acousticness=0.89, instrumentalness=0.98, tempo_normalized=0.33)),
    Song(song_id="s077", title="Etude No. 7", artist="Baroque Revival", genre="classical",
         features=AudioFeatures(energy=0.22, danceability=0.19, valence=0.47, acousticness=0.95, instrumentalness=0.92, tempo_normalized=0.27)),
    Song(song_id="s078", title="Coda", artist="The Chamber Players", genre="classical",
         features=AudioFeatures(energy=0.30, danceability=0.27, valence=0.55, acousticness=0.87, instrumentalness=0.94, tempo_normalized=0.35)),
    Song(song_id="s079", title="Intermezzo", artist="Vienna String Collective", genre="classical",
         features=AudioFeatures(energy=0.24, danceability=0.21, valence=0.49, acousticness=0.93, instrumentalness=0.96, tempo_normalized=0.29)),
    Song(song_id="s080", title="Largo", artist="Baroque Revival", genre="classical",
         features=AudioFeatures(energy=0.25, danceability=0.22, valence=0.50, acousticness=0.92, instrumentalness=0.95, tempo_normalized=0.30)),

    # --- R&B (s081–s090) ---
    Song(song_id="s081", title="Slow Motion", artist="Jade Rivers", genre="r&b",
         features=AudioFeatures(energy=0.64, danceability=0.77, valence=0.70, acousticness=0.28, instrumentalness=0.04, tempo_normalized=0.54)),
    Song(song_id="s082", title="Silk & Chrome", artist="The Groove Society", genre="r&b",
         features=AudioFeatures(energy=0.60, danceability=0.73, valence=0.66, acousticness=0.32, instrumentalness=0.03, tempo_normalized=0.50)),
    Song(song_id="s083", title="Velvet Touch", artist="Jade Rivers", genre="r&b",
         features=AudioFeatures(energy=0.66, danceability=0.79, valence=0.72, acousticness=0.26, instrumentalness=0.05, tempo_normalized=0.56)),
    Song(song_id="s084", title="After Dark", artist="Midnight Haze", genre="r&b",
         features=AudioFeatures(energy=0.58, danceability=0.71, valence=0.64, acousticness=0.34, instrumentalness=0.02, tempo_normalized=0.48)),
    Song(song_id="s085", title="Smooth Operator", artist="The Groove Society", genre="r&b",
         features=AudioFeatures(energy=0.63, danceability=0.76, valence=0.69, acousticness=0.29, instrumentalness=0.04, tempo_normalized=0.53)),
    Song(song_id="s086", title="Tender", artist="Jade Rivers", genre="r&b",
         features=AudioFeatures(energy=0.65, danceability=0.78, valence=0.71, acousticness=0.27, instrumentalness=0.05, tempo_normalized=0.55)),
    Song(song_id="s087", title="Night Drive", artist="Midnight Haze", genre="r&b",
         features=AudioFeatures(energy=0.59, danceability=0.72, valence=0.65, acousticness=0.33, instrumentalness=0.03, tempo_normalized=0.49)),
    Song(song_id="s088", title="Body Language", artist="The Groove Society", genre="r&b",
         features=AudioFeatures(energy=0.67, danceability=0.80, valence=0.73, acousticness=0.25, instrumentalness=0.06, tempo_normalized=0.57)),
    Song(song_id="s089", title="Gravity", artist="Jade Rivers", genre="r&b",
         features=AudioFeatures(energy=0.61, danceability=0.74, valence=0.67, acousticness=0.31, instrumentalness=0.03, tempo_normalized=0.51)),
    Song(song_id="s090", title="Champagne Nights", artist="Midnight Haze", genre="r&b",
         features=AudioFeatures(energy=0.62, danceability=0.75, valence=0.68, acousticness=0.30, instrumentalness=0.04, tempo_normalized=0.52)),

    # --- Metal (s091–s100) ---
    Song(song_id="s091", title="Iron Throne", artist="Black Citadel", genre="metal",
         features=AudioFeatures(energy=0.98, danceability=0.37, valence=0.27, acousticness=0.04, instrumentalness=0.27, tempo_normalized=0.76)),
    Song(song_id="s092", title="Void Walker", artist="Storm Hammer", genre="metal",
         features=AudioFeatures(energy=0.94, danceability=0.33, valence=0.23, acousticness=0.06, instrumentalness=0.23, tempo_normalized=0.72)),
    Song(song_id="s093", title="Ashen Wake", artist="Black Citadel", genre="metal",
         features=AudioFeatures(energy=1.00, danceability=0.39, valence=0.29, acousticness=0.03, instrumentalness=0.29, tempo_normalized=0.78)),
    Song(song_id="s094", title="Serpent King", artist="Savage Rift", genre="metal",
         features=AudioFeatures(energy=0.92, danceability=0.31, valence=0.21, acousticness=0.07, instrumentalness=0.21, tempo_normalized=0.70)),
    Song(song_id="s095", title="Nether Realm", artist="Storm Hammer", genre="metal",
         features=AudioFeatures(energy=0.97, danceability=0.36, valence=0.26, acousticness=0.04, instrumentalness=0.26, tempo_normalized=0.75)),
    Song(song_id="s096", title="Cursed Earth", artist="Black Citadel", genre="metal",
         features=AudioFeatures(energy=0.95, danceability=0.34, valence=0.24, acousticness=0.06, instrumentalness=0.24, tempo_normalized=0.73)),
    Song(song_id="s097", title="Blood & Iron", artist="Savage Rift", genre="metal",
         features=AudioFeatures(energy=0.93, danceability=0.32, valence=0.22, acousticness=0.07, instrumentalness=0.22, tempo_normalized=0.71)),
    Song(song_id="s098", title="Cathedral of Ash", artist="Storm Hammer", genre="metal",
         features=AudioFeatures(energy=0.99, danceability=0.38, valence=0.28, acousticness=0.03, instrumentalness=0.28, tempo_normalized=0.77)),
    Song(song_id="s099", title="Hailstorm", artist="Black Citadel", genre="metal",
         features=AudioFeatures(energy=0.96, danceability=0.35, valence=0.25, acousticness=0.05, instrumentalness=0.25, tempo_normalized=0.74)),
    Song(song_id="s100", title="Ragnarok", artist="Savage Rift", genre="metal",
         features=AudioFeatures(energy=0.91, danceability=0.30, valence=0.20, acousticness=0.08, instrumentalness=0.20, tempo_normalized=0.69)),
]

# ---------------------------------------------------------------------------
# 30 Users across 10 taste archetypes (3 users per archetype)
# ---------------------------------------------------------------------------

USERS: List[User] = [
    # Pop Lovers (u001–u003)
    User(user_id="u001", name="Alice", taste_profile="Pop Lover"),
    User(user_id="u002", name="Bob", taste_profile="Pop Lover"),
    User(user_id="u003", name="Carol", taste_profile="Pop Lover"),
    # EDM Fans (u004–u006)
    User(user_id="u004", name="Dave", taste_profile="EDM Fan"),
    User(user_id="u005", name="Eve", taste_profile="EDM Fan"),
    User(user_id="u006", name="Frank", taste_profile="EDM Fan"),
    # Indie Listeners (u007–u009)
    User(user_id="u007", name="Grace", taste_profile="Indie Listener"),
    User(user_id="u008", name="Hank", taste_profile="Indie Listener"),
    User(user_id="u009", name="Iris", taste_profile="Indie Listener"),
    # Hip-Hop Heads (u010–u012)
    User(user_id="u010", name="Jake", taste_profile="Hip-Hop Head"),
    User(user_id="u011", name="Kira", taste_profile="Hip-Hop Head"),
    User(user_id="u012", name="Liam", taste_profile="Hip-Hop Head"),
    # Jazz Aficionados (u013–u015)
    User(user_id="u013", name="Mia", taste_profile="Jazz Aficionado"),
    User(user_id="u014", name="Nate", taste_profile="Jazz Aficionado"),
    User(user_id="u015", name="Olivia", taste_profile="Jazz Aficionado"),
    # Rock Enthusiasts (u016–u018)
    User(user_id="u016", name="Paul", taste_profile="Rock Enthusiast"),
    User(user_id="u017", name="Quinn", taste_profile="Rock Enthusiast"),
    User(user_id="u018", name="Rose", taste_profile="Rock Enthusiast"),
    # Classical Fans (u019–u021)
    User(user_id="u019", name="Sam", taste_profile="Classical Fan"),
    User(user_id="u020", name="Tina", taste_profile="Classical Fan"),
    User(user_id="u021", name="Umar", taste_profile="Classical Fan"),
    # Acoustic Chill (u022–u024)
    User(user_id="u022", name="Vera", taste_profile="Acoustic Chill"),
    User(user_id="u023", name="Will", taste_profile="Acoustic Chill"),
    User(user_id="u024", name="Xena", taste_profile="Acoustic Chill"),
    # Party Mixers (u025–u027)
    User(user_id="u025", name="Yara", taste_profile="Party Mixer"),
    User(user_id="u026", name="Zack", taste_profile="Party Mixer"),
    User(user_id="u027", name="Anya", taste_profile="Party Mixer"),
    # Eclectic Listeners (u028–u030)
    User(user_id="u028", name="Boris", taste_profile="Eclectic Listener"),
    User(user_id="u029", name="Cleo", taste_profile="Eclectic Listener"),
    User(user_id="u030", name="Dean", taste_profile="Eclectic Listener"),
]

# ---------------------------------------------------------------------------
# Listening history
#
# LISTEN_MATRIX[user_id][song_id] = play_count
#
# Construction rules:
#   - Pop Lovers    listen to s011–s020 (pop) heavily, some r&b crossover
#   - EDM Fans      listen to s001–s010 (edm) heavily, some pop/hip-hop crossover
#   - Indie         listen to s041–s050 (indie) heavily, some acoustic/folk crossover
#   - Hip-Hop Heads listen to s021–s030 heavily, some r&b crossover
#   - Jazz Afic.    listen to s061–s070 (jazz) heavily, some classical crossover
#   - Rock          listen to s031–s040 (rock) heavily, some metal crossover
#   - Classical     listen to s071–s080 heavily, some jazz crossover
#   - Acoustic Chill listen to s051–s060 (acoustic) heavily, some indie crossover
#   - Party Mixers  listen to s001–s010 (edm) + s011–s020 (pop) equally
#   - Eclectic      listen to a spread across all genres
#
# Each user only has ~15–25 songs in their history (sparse matrix).
# Play counts 1–20 reflect engagement level.
# ---------------------------------------------------------------------------

LISTEN_MATRIX: Dict[str, Dict[str, int]] = {
    # === Pop Lovers ===
    "u001": {  # Alice — strong pop, some r&b
        "s011": 18, "s012": 14, "s013": 16, "s015": 12, "s017": 15,
        "s019": 11, "s020": 13, "s081": 7, "s085": 5, "s089": 4,
        "s016": 9, "s018": 8,
    },
    "u002": {  # Bob — pop-focused, a bit of edm
        "s011": 10, "s013": 12, "s014": 9, "s016": 14, "s018": 11,
        "s019": 13, "s020": 8, "s001": 5, "s003": 4, "s009": 3,
        "s012": 7, "s017": 6,
    },
    "u003": {  # Carol — pop + r&b crossover
        "s012": 16, "s013": 14, "s015": 18, "s016": 12, "s017": 10,
        "s082": 8, "s083": 7, "s086": 9, "s088": 6,
        "s011": 5, "s014": 4,
    },

    # === EDM Fans ===
    "u004": {  # Dave — heavy edm, some hip-hop
        "s001": 20, "s002": 17, "s003": 18, "s005": 15, "s007": 16,
        "s009": 14, "s010": 13, "s021": 5, "s025": 4, "s026": 3,
        "s004": 12, "s006": 11,
    },
    "u005": {  # Eve — edm + pop crossover
        "s002": 15, "s003": 18, "s004": 12, "s006": 16, "s008": 14,
        "s010": 11, "s011": 6, "s015": 5, "s017": 4,
        "s001": 9, "s005": 8,
    },
    "u006": {  # Frank — pure edm lover
        "s001": 19, "s003": 16, "s005": 18, "s007": 17, "s009": 15,
        "s002": 14, "s004": 13, "s006": 12, "s008": 11, "s010": 10,
        "s026": 3,
    },

    # === Indie Listeners ===
    "u007": {  # Grace — indie + acoustic crossover
        "s041": 16, "s042": 14, "s043": 15, "s045": 12, "s048": 11,
        "s051": 7, "s055": 6, "s059": 5, "s041": 13,
        "s044": 9, "s046": 8,
    },
    "u008": {  # Hank — indie-focused
        "s042": 18, "s044": 15, "s046": 16, "s047": 13, "s049": 14,
        "s050": 12, "s043": 11, "s048": 9,
        "s041": 7, "s045": 6,
    },
    "u009": {  # Iris — indie + some r&b
        "s041": 14, "s043": 16, "s045": 15, "s046": 12, "s048": 13,
        "s050": 11, "s082": 5, "s089": 4,
        "s044": 9, "s047": 7,
    },

    # === Hip-Hop Heads ===
    "u010": {  # Jake — heavy hip-hop
        "s021": 19, "s022": 16, "s023": 18, "s025": 15, "s026": 17,
        "s028": 14, "s030": 13, "s024": 12, "s027": 11,
        "s081": 4, "s085": 3,
    },
    "u011": {  # Kira — hip-hop + r&b
        "s022": 17, "s024": 15, "s025": 16, "s027": 13, "s029": 14,
        "s081": 8, "s083": 7, "s087": 6, "s090": 5,
        "s021": 9, "s023": 8,
    },
    "u012": {  # Liam — hip-hop + edm
        "s021": 15, "s023": 17, "s026": 16, "s028": 14, "s030": 15,
        "s001": 5, "s004": 4, "s007": 3,
        "s022": 10, "s024": 9,
    },

    # === Jazz Aficionados ===
    "u013": {  # Mia — heavy jazz
        "s061": 17, "s062": 15, "s063": 16, "s065": 13, "s066": 14,
        "s068": 12, "s069": 11, "s070": 10,
        "s071": 5, "s075": 4,
    },
    "u014": {  # Nate — jazz + classical crossover
        "s062": 16, "s064": 14, "s065": 15, "s067": 12, "s069": 13,
        "s071": 8, "s073": 7, "s077": 6, "s079": 5,
        "s061": 9, "s063": 8,
    },
    "u015": {  # Olivia — jazz aficionado
        "s061": 15, "s063": 17, "s065": 16, "s067": 13, "s068": 14,
        "s070": 12, "s066": 11,
        "s062": 9, "s064": 7, "s069": 6,
    },

    # === Rock Enthusiasts ===
    "u016": {  # Paul — rock + metal crossover
        "s031": 18, "s032": 15, "s033": 17, "s035": 14, "s037": 15,
        "s039": 13, "s091": 6, "s094": 5, "s097": 4,
        "s034": 11, "s036": 10,
    },
    "u017": {  # Quinn — pure rock
        "s032": 17, "s034": 16, "s035": 18, "s036": 15, "s038": 14,
        "s040": 13, "s033": 12, "s039": 11,
        "s031": 8, "s037": 7,
    },
    "u018": {  # Rose — rock + indie
        "s031": 16, "s033": 14, "s036": 15, "s038": 13, "s040": 14,
        "s041": 6, "s044": 5, "s047": 4,
        "s032": 10, "s035": 9,
    },

    # === Classical Fans ===
    "u019": {  # Sam — heavy classical
        "s071": 18, "s072": 16, "s073": 17, "s075": 14, "s076": 15,
        "s078": 13, "s079": 12, "s080": 11,
        "s061": 4, "s062": 3,
    },
    "u020": {  # Tina — classical + jazz
        "s072": 17, "s074": 15, "s075": 16, "s077": 13, "s079": 14,
        "s080": 12, "s062": 7, "s064": 6, "s067": 5,
        "s071": 9, "s073": 8,
    },
    "u021": {  # Umar — classical aficionado
        "s071": 16, "s073": 18, "s074": 15, "s076": 17, "s078": 14,
        "s080": 15, "s075": 13,
        "s072": 9, "s077": 7, "s079": 6,
    },

    # === Acoustic Chill ===
    "u022": {  # Vera — acoustic + indie
        "s051": 17, "s052": 15, "s053": 16, "s055": 13, "s057": 14,
        "s059": 12, "s041": 6, "s044": 5, "s050": 4,
        "s054": 10, "s056": 9,
    },
    "u023": {  # Will — pure acoustic
        "s052": 18, "s054": 16, "s055": 17, "s056": 15, "s058": 14,
        "s060": 13, "s053": 12, "s059": 11,
        "s051": 8, "s057": 7,
    },
    "u024": {  # Xena — acoustic + jazz crossover
        "s051": 15, "s053": 17, "s056": 16, "s057": 13, "s059": 14,
        "s060": 12, "s061": 6, "s065": 5, "s070": 4,
        "s052": 9, "s055": 8,
    },

    # === Party Mixers (edm + pop equally) ===
    "u025": {  # Yara — edm + pop
        "s001": 12, "s003": 11, "s005": 13, "s007": 10, "s009": 9,
        "s011": 12, "s013": 11, "s015": 10, "s017": 9, "s019": 8,
        "s002": 7, "s012": 7,
    },
    "u026": {  # Zack — edm + pop + hip-hop
        "s002": 10, "s004": 11, "s006": 12, "s008": 9, "s010": 10,
        "s012": 11, "s014": 10, "s016": 9, "s018": 8,
        "s021": 5, "s025": 4,
    },
    "u027": {  # Anya — edm + r&b party vibes
        "s003": 14, "s005": 13, "s007": 12, "s009": 11, "s001": 10,
        "s081": 9, "s083": 8, "s085": 10, "s087": 7,
        "s013": 6, "s017": 5,
    },

    # === Eclectic Listeners (spread across genres) ===
    "u028": {  # Boris — a bit of everything
        "s011": 8, "s021": 7, "s031": 6, "s041": 5, "s051": 6,
        "s061": 7, "s071": 5, "s081": 6, "s091": 4, "s001": 5,
        "s065": 4, "s045": 5, "s082": 3,
    },
    "u029": {  # Cleo — eclectic with jazz/indie lean
        "s061": 9, "s062": 8, "s041": 7, "s043": 8, "s051": 6,
        "s011": 5, "s081": 6, "s031": 4, "s071": 5, "s001": 3,
        "s066": 7, "s047": 6,
    },
    "u030": {  # Dean — eclectic with rock/metal lean
        "s031": 10, "s033": 9, "s091": 8, "s094": 7, "s021": 6,
        "s001": 5, "s011": 4, "s061": 5, "s041": 4, "s051": 3,
        "s036": 8, "s095": 6,
    },
}


def build_interaction_df() -> pd.DataFrame:
    """
    Convert the LISTEN_MATRIX nested dict into a long-format DataFrame.

    Returns a DataFrame with columns: [user_id, song_id, play_count]
    containing only non-zero entries (the sparse interactions).

    This format is the canonical input for all algorithm modules because:
    - It's easy to pivot into a user-item matrix: df.pivot(user, song, play_count)
    - It's easy to filter by user: df[df.user_id == 'u001']
    - It's explicit about sparsity — missing rows mean 0 plays, not NaN
    """
    rows = []
    for user_id, song_plays in LISTEN_MATRIX.items():
        for song_id, play_count in song_plays.items():
            rows.append({
                "user_id": user_id,
                "song_id": song_id,
                "play_count": play_count,
            })
    return pd.DataFrame(rows, columns=["user_id", "song_id", "play_count"])
