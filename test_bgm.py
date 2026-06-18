from src.services.bgm_service import BgmService

test_songs = [
    "Hans Zimmer - Time (盗梦空间)",
    "Eminem - Lose Yourself",
    "RADWIMPS - Sparkle (你的名字)",
    "泽野弘之 - aLIEz (核爆神曲)"
]

for song in test_songs:
    result = BgmService.analyze_bgm(song)
    if result:
        print(f"✅ 【{result['track_name']}】分析结果:")
        print(f"  📊 数值: 情绪 {result['valence']} | 能量 {result['energy']} | BPM {result['tempo']}")
        print(f"  🏷️ 标签: {result['tags']}\n")