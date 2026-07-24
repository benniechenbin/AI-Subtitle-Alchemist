from dataclasses import dataclass


@dataclass
class MovieMeta:
    movie_name: str
    poster_url: str | None = None
    release_year: int | None = None
    highlight_status: int = 0
    line_count: int = 0


@dataclass
class GoldenQuote:
    content: str
    timestamp: str
    reason: str
    movie_name: str | None = None


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
    updated_at: str | None = None


@dataclass
class Tag:
    id: int
    name: str
    tag_type: str
    confidence: float = 1.0
    source: str = "manual"
