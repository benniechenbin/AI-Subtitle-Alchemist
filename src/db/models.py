from dataclasses import dataclass
from typing import Optional


@dataclass
class MovieMeta:
    movie_name: str
    poster_url: Optional[str] = None
    release_year: Optional[int] = None
    highlight_status: int = 0
    line_count: int = 0


@dataclass
class GoldenQuote:
    content: str
    timestamp: str
    reason: str
    movie_name: Optional[str] = None


@dataclass
class BgmAsset:
    track_name: str
    artist: str
    valence: float
    energy: float
    tempo: int
    tags: str
    source: str = "llm_prior"
    confidence: float = 0.5
    user_verified: int = 0
    updated_at: Optional[str] = None


@dataclass
class Tag:
    id: int
    name: str
    tag_type: str
    confidence: float = 1.0
    source: str = "manual"
