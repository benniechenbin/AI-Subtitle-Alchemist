import json

from app.tabs.clean_import import _filter_manifest_rows
from src.services.upload_manifest_adapter import prepare_upload_analysis


class FakeUploadedFile:
    def __init__(self, name: str, raw_bytes: bytes = b""):
        self.name = name
        self._raw_bytes = raw_bytes
        self._position = 0

    def getvalue(self) -> bytes:
        return self._raw_bytes

    def read(self) -> bytes:
        data = self._raw_bytes[self._position :]
        self._position = len(self._raw_bytes)
        return data

    def seek(self, position: int) -> None:
        self._position = position

    def tell(self) -> int:
        return self._position


def _manifest_file(items, name: str = "harvester_import_manifest.json"):
    raw = json.dumps({"items": items}, ensure_ascii=False).encode("utf-8")
    return FakeUploadedFile(name, raw)


def _manifest_list_file(items, name: str = "harvester_import_manifest.json"):
    raw = json.dumps(items, ensure_ascii=False).encode("utf-8")
    return FakeUploadedFile(name, raw)


def test_plain_upload_filename_recognition_still_works():
    result = prepare_upload_analysis([FakeUploadedFile("The.Matrix.1999.srt")])

    assert [f.name for f in result.subtitle_files] == ["The.Matrix.1999.srt"]
    assert len(result.analysis_data) == 1
    assert result.analysis_data[0]["识别片名"] == "The Matrix"
    assert result.analysis_data[0]["年份"] == 1999
    assert result.analysis_data[0]["状态"] == "待确认"


def test_harvester_manifest_with_weak_relative_filename_uses_json():
    result = prepare_upload_analysis(
        [
            _manifest_file(
                [
                    {
                        "title": "正确片名",
                        "year": 2024,
                        "subtitle_file": "movie_980477/Chs.srt",
                    }
                ]
            ),
            FakeUploadedFile("movie_980477/Chs.srt"),
        ]
    )

    assert [f.name for f in result.subtitle_files] == ["movie_980477/Chs.srt"]
    assert all(
        row["原始文件名"] != "harvester_import_manifest.json"
        for row in result.analysis_data
    )
    assert result.analysis_data[0]["识别片名"] == "正确片名"
    assert result.analysis_data[0]["年份"] == 2024
    assert "Harvester JSON" in result.analysis_data[0]["状态"]


def test_harvester_manifest_with_folder_prefix_is_filtered_from_results():
    result = prepare_upload_analysis(
        [
            _manifest_file(
                [
                    {
                        "title": "正确片名",
                        "year": 2024,
                        "subtitle_file": "movie_980477/Chs.srt",
                    }
                ],
                name="staging/harvester_import_manifest.json",
            ),
            FakeUploadedFile("movie_980477/Chs.srt"),
        ]
    )

    assert [f.name for f in result.subtitle_files] == ["movie_980477/Chs.srt"]
    assert [row["原始文件名"] for row in result.analysis_data] == [
        "movie_980477/Chs.srt"
    ]
    assert result.analysis_data[0]["识别片名"] == "正确片名"


def test_clean_import_defensively_filters_stale_manifest_rows():
    rows = [
        {
            "原始文件名": "harvester_import_manifest.json",
            "识别片名": "harvester import manifest json",
        },
        {"原始文件名": "Chs.srt", "识别片名": "Chs"},
    ]

    filtered = _filter_manifest_rows(rows)

    assert filtered == [{"原始文件名": "Chs.srt", "识别片名": "Chs"}]


def test_harvester_manifest_matches_unique_basename_when_upload_loses_path():
    result = prepare_upload_analysis(
        [
            _manifest_file(
                [
                    {
                        "title": "唯一 basename 片名",
                        "year": 2025,
                        "subtitle_file": "movie_980477/Chs.srt",
                    }
                ]
            ),
            FakeUploadedFile("Chs.srt"),
        ]
    )

    assert result.analysis_data[0]["识别片名"] == "唯一 basename 片名"
    assert result.analysis_data[0]["年份"] == 2025
    assert result.analysis_data[0]["状态"] == "来自Harvester JSON"


def test_harvester_manifest_accepts_top_level_list_and_windows_paths():
    result = prepare_upload_analysis(
        [
            _manifest_list_file(
                [
                    {
                        "title": "Windows 路径片名",
                        "year": 2026,
                        "subtitle_file": r"movie_980477\Cht.srt",
                    }
                ]
            ),
            FakeUploadedFile("movie_980477/Cht.srt"),
        ]
    )

    assert result.analysis_data[0]["识别片名"] == "Windows 路径片名"
    assert result.analysis_data[0]["年份"] == 2026
    assert result.analysis_data[0]["状态"] == "来自Harvester JSON"


def test_harvester_manifest_basename_conflict_falls_back_to_filename():
    result = prepare_upload_analysis(
        [
            _manifest_file(
                [
                    {
                        "title": "第一部",
                        "year": 2021,
                        "subtitle_file": "movie_1/Chs.srt",
                    },
                    {
                        "title": "第二部",
                        "year": 2022,
                        "subtitle_file": "movie_2/Chs.srt",
                    },
                ]
            ),
            FakeUploadedFile("Chs.srt"),
        ]
    )

    assert result.analysis_data[0]["识别片名"] not in {"第一部", "第二部"}
    assert result.analysis_data[0]["状态"] == "JSON匹配冲突，请手动确认"


def test_broken_harvester_manifest_warns_and_falls_back_to_filename():
    result = prepare_upload_analysis(
        [
            FakeUploadedFile("harvester_import_manifest.json", b"{bad json"),
            FakeUploadedFile("The.Matrix.1999.srt"),
        ]
    )

    assert result.warnings
    assert result.analysis_data[0]["识别片名"] == "The Matrix"
    assert result.analysis_data[0]["年份"] == 1999
    assert result.analysis_data[0]["状态"] == "待确认"
