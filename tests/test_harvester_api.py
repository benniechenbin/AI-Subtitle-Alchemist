import importlib.util
from pathlib import Path

import pytest
import requests

MODULE_PATH = Path(__file__).parents[1] / "src" / "services" / "harvester_api.py"
SPEC = importlib.util.spec_from_file_location("harvester_api", MODULE_PATH)
harvester_api = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(harvester_api)

HarvesterApiClient = harvester_api.HarvesterApiClient
HarvesterApiError = harvester_api.HarvesterApiError


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


def test_harvester_health_uses_base_url(monkeypatch):
    calls = []

    def fake_request(method, url, json=None, params=None, timeout=None):
        calls.append((method, url, json, params, timeout))
        return FakeResponse({"status": "ok", "output_dir": "/tmp/output"})

    monkeypatch.setattr(harvester_api.requests, "request", fake_request)

    payload = HarvesterApiClient("http://127.0.0.1:8000/").health()

    assert payload["status"] == "ok"
    assert calls == [
        ("GET", "http://127.0.0.1:8000/health", None, None, 2.0)
    ]


def test_harvester_discovery_posts_payload(monkeypatch):
    calls = []

    def fake_request(method, url, json=None, params=None, timeout=None):
        calls.append((method, url, json, params, timeout))
        return FakeResponse({"candidate_count": 2})

    monkeypatch.setattr(harvester_api.requests, "request", fake_request)

    payload = HarvesterApiClient("http://harvester").run_discovery({"year": 2026})

    assert payload["candidate_count"] == 2
    assert calls[0][0:3] == ("POST", "http://harvester/discovery/run", {"year": 2026})


def test_harvester_client_methods_map_to_expected_endpoints(monkeypatch):
    calls = []

    def fake_request(method, url, json=None, params=None, timeout=None):
        calls.append((method, url, json, params))
        return FakeResponse({"ok": True})

    monkeypatch.setattr(harvester_api.requests, "request", fake_request)
    client = HarvesterApiClient("http://harvester")

    client.list_candidates("/tmp/candidates.json")
    client.curate_candidates([{"title": "Demo"}])
    client.collect_subtitles({"providers": ["assrt"]})
    client.list_manifests()
    client.get_manifest("movie_1")
    client.export_library(manifest_paths=["/tmp/manifest.json"], library_dir="/tmp/lib")

    assert calls == [
        ("GET", "http://harvester/candidates", None, {"source_path": "/tmp/candidates.json"}),
        ("POST", "http://harvester/candidates/curate", {"rows": [{"title": "Demo"}]}, None),
        ("POST", "http://harvester/subtitles/collect", {"providers": ["assrt"]}, None),
        ("GET", "http://harvester/manifests", None, None),
        ("GET", "http://harvester/manifests/movie_1", None, None),
        (
            "POST",
            "http://harvester/export/library",
            {"manifest_paths": ["/tmp/manifest.json"], "library_dir": "/tmp/lib"},
            None,
        ),
    ]


def test_harvester_http_error_uses_detail(monkeypatch):
    def fake_request(method, url, json=None, params=None, timeout=None):
        return FakeResponse({"detail": "bad request"}, status_code=400)

    monkeypatch.setattr(harvester_api.requests, "request", fake_request)

    with pytest.raises(HarvesterApiError, match="bad request"):
        HarvesterApiClient("http://harvester").list_manifests()


def test_harvester_connection_error(monkeypatch):
    def fake_request(method, url, json=None, params=None, timeout=None):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(harvester_api.requests, "request", fake_request)

    with pytest.raises(HarvesterApiError, match="无法连接"):
        HarvesterApiClient("http://harvester").health()
