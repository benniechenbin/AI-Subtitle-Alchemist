from __future__ import annotations

from typing import Any

import requests


class HarvesterApiError(RuntimeError):
    pass


class HarvesterApiClient:
    def __init__(self, base_url: str, *, timeout: float | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health", timeout=2.0)

    def run_discovery(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/discovery/run", json=payload)

    def list_candidates(self, source_path: str | None = None) -> dict[str, Any]:
        params = {"source_path": source_path} if source_path else None
        return self._request("GET", "/candidates", params=params)

    def curate_candidates(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        return self._request("POST", "/candidates/curate", json={"rows": rows})

    def collect_subtitles(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/subtitles/collect", json=payload)

    def list_manifests(self) -> dict[str, Any]:
        return self._request("GET", "/manifests")

    def get_manifest(self, media_key: str) -> dict[str, Any]:
        return self._request("GET", f"/manifests/{media_key}")

    def export_library(
        self,
        *,
        manifest_paths: list[str],
        library_dir: str,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/export/library",
            json={"manifest_paths": manifest_paths, "library_dir": library_dir},
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        try:
            response = requests.request(
                method,
                f"{self.base_url}{path}",
                json=json,
                params=params,
                timeout=timeout or self.timeout,
            )
        except requests.RequestException as exc:
            raise HarvesterApiError(f"无法连接 Harvester API：{exc}") from exc

        if response.status_code >= 400:
            raise HarvesterApiError(_error_message(response))

        try:
            payload = response.json()
        except ValueError as exc:
            raise HarvesterApiError("Harvester API 返回了无法解析的 JSON。") from exc

        if not isinstance(payload, dict):
            raise HarvesterApiError("Harvester API 返回格式异常。")
        return payload


def _error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return f"Harvester API 请求失败：HTTP {response.status_code}"

    detail = payload.get("detail") if isinstance(payload, dict) else None
    if isinstance(detail, list):
        return f"Harvester API 请求失败：{detail}"
    if detail:
        return f"Harvester API 请求失败：{detail}"
    return f"Harvester API 请求失败：HTTP {response.status_code}"
