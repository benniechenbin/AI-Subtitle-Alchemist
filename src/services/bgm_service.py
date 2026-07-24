import json
import re

from src.config.settings import settings
from src.observability.logger import logger
from src.services.llm import call_llm_api


class BgmService:
    @staticmethod
    def analyze_bgm(track_name: str, api_key: str | None = None) -> dict | None:
        """
        纯大模型架构：通过大模型的先验知识，直接对 BGM 进行剪辑维度的量化分析。
        """
        logger.info(f"🧠 正在请求大模型分析 BGM: {track_name} ...")

        system_prompt = """你是一位顶级的电影配乐指导和混剪大神。
请根据我提供的音乐/配乐名称，凭借你的知识库，精准评估它的声学特征，并必须返回纯 JSON 格式。

【JSON 必须包含的字段】
1. "artist": 艺术家/作曲家名称
2. "normalized_title": 规范化的曲目名称
3. "valence": 0.0 到 1.0 的浮点数 (0 为极度悲伤/压抑，1 为极度欢快/治愈)
4. "energy": 0.0 到 1.0 的浮点数 (0 为极度安静/舒缓，1 为极度激烈/吵闹)
5. "tempo": 估算的 BPM 节拍数
6. "tags": 3 到 4 个专门用于【视频剪辑场景】的中文标签 (例如: ["高燃踩点", "悬疑压抑", "空镜头过渡", "史诗感"])
7. "confidence": 你对该分析结果的把握程度 (0.0 到 1.0)

【严禁事项】
绝对不要输出任何 Markdown 标记（如 ```json）、不要有任何前言后记，只能输出合法的 JSON 字符串！
"""
        user_prompt = f"请分析这首 BGM：【 {track_name} 】"

        try:
            # 调用你配置好的大模型引擎
            llm_response = call_llm_api(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                api_key=api_key or settings.get_llm_api_key(),
                model_name=settings.prefs.llm_model_name,
                base_url=settings.prefs.llm_base_url,
            )

            # 暴力清洗可能带有的 Markdown 格式
            match = re.search(r"\{.*\}", llm_response, re.DOTALL)
            if match:
                clean_json_str = match.group(0)
                features = json.loads(clean_json_str)
            else:
                raise ValueError("未从大模型回复中提取到有效的 JSON 对象")

            result = {
                "track_name": track_name,
                "artist": features.get("artist", "未知艺术家"),
                "normalized_title": features.get("normalized_title", track_name),
                "valence": float(features.get("valence", 0.5)),
                "energy": float(features.get("energy", 0.5)),
                "tempo": int(features.get("tempo", 120)),
                "tags": features.get("tags", ["未分类"]),
                "source": "llm_prior",
                "confidence": float(features.get("confidence", 0.5)),
                "raw_metadata": llm_response,
            }
            logger.success(
                f"🎵 AI 分析完成: {result['normalized_title']} by {result['artist']} | 情绪: {result['valence']} | 把握度: {result['confidence']}"
            )
            return result

        except json.JSONDecodeError:
            logger.error(f"❌ 大模型返回的不是合法 JSON: {llm_response}")
            return None
        except Exception as e:  # noqa: BLE001
            logger.error(f"❌ BGM 分析过程发生未知错误: {e}")
            return None
