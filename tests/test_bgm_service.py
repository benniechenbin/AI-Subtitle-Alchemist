import pytest
from src.services.bgm_service import BgmService


@pytest.mark.manual
def test_bgm_analysis():
    """通过 LLM 先验知识分析 BGM 音乐属性的测试（需联网 / API Key）。"""
    test_songs = [
        "Hans Zimmer - Time (盗梦空间)",
        "Eminem - Lose Yourself",
        "RADWIMPS - Sparkle (你的名字)",
        "泽野弘之 - aLIEz (核爆神曲)",
    ]

    for song in test_songs:
        result = BgmService.analyze_bgm(song)
        if result:
            assert "track_name" in result
            assert "valence" in result
            assert "energy" in result
            assert "tempo" in result
            assert "tags" in result
