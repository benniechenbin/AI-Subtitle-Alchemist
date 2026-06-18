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
                    {"poster_path": "/wrong-year.jpg", "release_date": "2020-01-01"},
                    {"poster_path": "/right-year.jpg", "first_air_date": "2021-02-03"},
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


def test_fetch_tmdb_poster_falls_back_to_first_available_poster(monkeypatch):
    def fake_get(_url, params, timeout):
        return FakeResponse(
            {
                "results": [
                    {"poster_path": None, "release_date": "2024-01-01"},
                    {"poster_path": "/fallback.jpg", "release_date": "2020-01-01"},
                    {"poster_path": "/later.jpg", "release_date": "2021-01-01"},
                ]
            }
        )

    monkeypatch.setattr(tmdb_service.requests, "get", fake_get)

    poster_url = tmdb_service.fetch_tmdb_poster(
        "Example Movie",
        api_key="fake-key",
        release_year=1999,
    )

    assert poster_url == "https://image.tmdb.org/t/p/w500/fallback.jpg"
