from src.services import tmdb_service


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_fetch_tmdb_poster_prefers_matching_release_year(monkeypatch):
    def fake_get(_url, params, timeout):
        assert params["query"] == "Example Movie"
        assert timeout == 10
        return FakeResponse(
            {
                "results": [
                    {
                        "media_type": "movie",
                        "poster_path": "/wrong-year.jpg",
                        "release_date": "2020-01-01",
                    },
                    {
                        "media_type": "tv",
                        "poster_path": "/right-year.jpg",
                        "first_air_date": "2021-02-03",
                    },
                ]
            }
        )

    monkeypatch.setattr(tmdb_service.requests, "get", fake_get)

    poster_url = tmdb_service.fetch_tmdb_poster(
        "Example Movie",
        api_key="fake-key",
        release_year=2021,
    )

    assert poster_url == "https://image.tmdb.org/t/p/w500/right-year.jpg"


def test_fetch_tmdb_poster_rejects_wrong_release_year(monkeypatch):
    def fake_get(_url, params, timeout):
        return FakeResponse(
            {
                "results": [
                    {
                        "media_type": "movie",
                        "poster_path": "/fallback.jpg",
                        "release_date": "2020-01-01",
                    },
                    {
                        "media_type": "movie",
                        "poster_path": "/later.jpg",
                        "release_date": "2021-01-01",
                    },
                ]
            }
        )

    monkeypatch.setattr(tmdb_service.requests, "get", fake_get)

    poster_url = tmdb_service.fetch_tmdb_poster(
        "Example Movie",
        api_key="fake-key",
        release_year=1999,
    )

    assert poster_url is None


def test_search_filters_people_and_normalizes_tmdb_identity(monkeypatch):
    def fake_get(_url, params, timeout):
        return FakeResponse(
            {
                "results": [
                    {
                        "id": 1,
                        "media_type": "person",
                        "name": "Example Person",
                    },
                    {
                        "id": 42,
                        "media_type": "tv",
                        "name": "中文剧名",
                        "original_name": "Original Series",
                        "first_air_date": "2025-01-02",
                        "poster_path": "/series.jpg",
                    },
                ]
            }
        )

    monkeypatch.setattr(tmdb_service.requests, "get", fake_get)

    metadata = tmdb_service.search_tmdb_metadata("Original Series", api_key="fake-key")

    assert metadata["title"] == "中文剧名"
    assert metadata["year"] == 2025
    assert metadata["tmdb_id"] == 42
    assert metadata["media_type"] == "tv"
    assert metadata["raw"]["original_name"] == "Original Series"
