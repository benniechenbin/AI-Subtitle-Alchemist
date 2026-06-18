import json
import posixpath
from pathlib import Path
from typing import Any

import guessit

from src.observability.logger import logger


MANIFEST_NAME = "harvester_import_manifest.json"


def normalize_manifest_path(path: str | Path) -> str:
    raw = str(path).replace("\\", "/").strip()
    if not raw:
        return ""
    normalized = posixpath.normpath(raw)
    if normalized == ".":
        return ""
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def load_harvester_manifest(input_dir: str | Path) -> dict[str, dict]:
    manifest_path = Path(input_dir) / MANIFEST_NAME
    if not manifest_path.exists():
        return {}

    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to read harvester manifest {manifest_path}: {e}")
        return {}

    if isinstance(manifest, dict):
        items = manifest.get("items", [])
    elif isinstance(manifest, list):
        items = manifest
    else:
        logger.warning(f"Unsupported harvester manifest format: {manifest_path}")
        return {}

    if not isinstance(items, list):
        logger.warning(f"Invalid harvester manifest items: {manifest_path}")
        return {}

    manifest_dir = manifest_path.parent.resolve()
    index: dict[str, dict] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        subtitle_file = item.get("subtitle_file")
        if not subtitle_file:
            continue

        rel_key = normalize_manifest_path(subtitle_file)
        if not rel_key:
            continue

        enriched_item = dict(item)
        enriched_item["_harvester_relative_path"] = rel_key

        index[rel_key] = enriched_item
        abs_key = normalize_manifest_path(manifest_dir / Path(rel_key))
        if abs_key:
            enriched_item["_harvester_absolute_path"] = abs_key
            index[abs_key] = enriched_item

    if index:
        logger.info(f"Loaded harvester manifest: {manifest_path} ({len(index)} path keys)")
    return index


def resolve_subtitle_metadata(
    file_path: str | Path,
    root_path: str | Path,
    manifest_index: dict[str, dict],
) -> dict[str, Any]:
    path = Path(file_path)
    guess_info = guessit.guessit(path.name)
    item = _find_manifest_item(path, root_path, manifest_index)

    if item:
        movie_name = item.get("title") or guess_info.get("title") or path.name
        year = item.get("year") or guess_info.get("year", 0)
        metadata = {
            "movie_name": movie_name,
            "year": year,
            "season": guess_info.get("season", 0),
            "episode": guess_info.get("episode", 0),
            "media_type": item.get("media_type"),
            "tmdb_id": item.get("tmdb_id"),
            "imdb_id": item.get("imdb_id"),
            "provider": item.get("provider"),
            "source_id": item.get("source_id"),
            "source": "harvester",
        }
        logger.info(
            "Harvester metadata matched: "
            f"{path.name} -> {movie_name} ({year}) "
            f"media_type={metadata['media_type']} "
            f"tmdb_id={metadata['tmdb_id']} imdb_id={metadata['imdb_id']} "
            f"provider={metadata['provider']} source_id={metadata['source_id']}"
        )
        return metadata

    return {
        "movie_name": guess_info.get("title") or path.name,
        "year": guess_info.get("year", 0),
        "season": guess_info.get("season", 0),
        "episode": guess_info.get("episode", 0),
        "media_type": None,
        "tmdb_id": None,
        "imdb_id": None,
        "provider": None,
        "source_id": None,
        "source": "guessit",
    }


def _find_manifest_item(
    file_path: Path,
    root_path: str | Path,
    manifest_index: dict[str, dict],
) -> dict | None:
    if not manifest_index:
        return None

    rel_key = _relative_key(file_path, root_path)
    abs_key = normalize_manifest_path(file_path.resolve())
    return manifest_index.get(rel_key) or manifest_index.get(abs_key)


def _relative_key(file_path: Path, root_path: str | Path) -> str:
    root = Path(root_path)
    try:
        return normalize_manifest_path(file_path.resolve().relative_to(root.resolve()))
    except ValueError:
        try:
            return normalize_manifest_path(file_path.relative_to(root))
        except ValueError:
            return normalize_manifest_path(file_path)
