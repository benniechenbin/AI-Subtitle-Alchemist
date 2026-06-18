import json
import re
from src.db import get_db_connection
from src.services.llm import call_llm_api
from src.prompts import HIGHLIGHT_SYSTEM_PROMPT
from src.config.settings import settings
from src.observability.logger import logger

# 专门针对金句提取定制的 System Prompt


class HighlightService:
    @staticmethod
    def extract_movie_highlights(movie_name: str, chunk_size: int = 200, api_key: str | None = None) -> int:
        """
        对指定电影的完整字幕进行大模型分批扫描，提炼金句并持久化。

        :param movie_name: 电影/影视剧名称
        :param chunk_size: 每批喂给大模型的台词数量
        :param api_key: 可选的 API Key，若不传则从配置读取
        :return: 最终成功提炼并入库的金句数量
        """
        db_path = settings.prefs.db_path

        # 1. 拦截与状态更新：将状态标记为 1 (正在挖掘)，并清空旧数据
        conn = get_db_connection(db_path)
        c = conn.cursor()
        try:
            c.execute(
                "UPDATE movies_meta SET highlight_status = 1 WHERE movie_name = ?", 
                (movie_name,)
            )
            c.execute("DELETE FROM golden_quotes WHERE movie_name = ?", (movie_name,))
            conn.commit()
        except Exception as e:
            logger.error(f"❌ 初始化挖掘状态失败: {e}")
            conn.close()
            return 0

        # 2. 拉取该电影的所有原始字幕
        c.execute("""
            SELECT start_time, content FROM subtitles 
            WHERE movie_name = ? 
            ORDER BY id ASC
        """, (movie_name,))
        all_lines = c.fetchall()
        conn.close()

        if not all_lines:
            logger.warning(f"🗙 影视剧《{movie_name}》在数据库中未找到任何台词记录，中止挖掘。")
            HighlightService._set_status(movie_name, 0) # 重置为未挖掘
            return 0

        logger.info(f"🎬 开始对《{movie_name}》进行金句深度挖掘，总计 {len(all_lines)} 行台词，每批次 {chunk_size} 行...")

        # 获取当前配置的 LLM 参数
        final_api_key = api_key or settings.get_llm_api_key()
        model_name = settings.prefs.llm_model_name
        base_url = settings.prefs.llm_base_url

        total_saved = 0

        try:
            # 3. 核心滑动窗口分批处理
            for i in range(0, len(all_lines), chunk_size):
                chunk = all_lines[i : i + chunk_size]

                # 组装成大模型易读的文本块
                context_lines = []
                for item in chunk:
                    context_lines.append(f"[{item[0]}] {item[1]}")
                raw_context_text = "\n".join(context_lines)

                user_prompt = f"【当前影视剧】《{movie_name}》\n【待分析台词片段如下】:\n{raw_context_text}\n\n请严格按JSON格式输出提炼结果："

                try:
                    # 调用大模型
                    response_text = call_llm_api(
                        system_prompt=HIGHLIGHT_SYSTEM_PROMPT,
                        user_prompt=user_prompt,
                        api_key=final_api_key,
                        model_name=model_name,
                        base_url=base_url
                    )

                    # 清洗大模型可能自带的 ```json 脏标记
                    clean_json_str = re.sub(r"```json\s*|```", "", response_text).strip()
                    if not clean_json_str or clean_json_str == "[]":
                        continue

                    quotes_list = json.loads(clean_json_str)

                    # 4. 提取成功，准备批量写入 golden_quotes 表
                    if quotes_list:
                        insert_rows = []
                        for q in quotes_list:
                            # 防御性编程：确保大模型返回了必要的字段
                            if "content" in q and "timestamp" in q:
                                insert_rows.append((
                                    movie_name,
                                    q["content"],
                                    q["timestamp"],
                                    q.get("reason", "未提供理由")
                                ))

                        if insert_rows:
                            conn = get_db_connection(db_path)
                            c = conn.cursor()
                            c.executemany("""
                                INSERT INTO golden_quotes (movie_name, quote_content, timestamp, reason)
                                VALUES (?, ?, ?, ?)
                            """, insert_rows)
                            conn.commit()
                            conn.close()

                            total_saved += len(insert_rows)
                            logger.info(f"📦 批次 {i//chunk_size + 1} 处理完毕，成功提炼 {len(insert_rows)} 条金句。")

                except json.JSONDecodeError:
                    logger.error(f"❌ 大模型批次 {i//chunk_size + 1} 返回的不是合法的 JSON 格式，跳过该批次。原始内容: {response_text}")

            # 5. 收尾：更新状态为 2 (已挖掘)
            HighlightService._set_status(movie_name, 2)
            logger.success(f"🎉 《{movie_name}》金句挖掘战役圆满结束！共计持久化 {total_saved} 条黄金时刻台词。")

        except Exception as e:
            logger.error(f"❌ 《{movie_name}》挖掘过程中断: {e}")
            HighlightService._set_status(movie_name, 3) # 标记为失败
            raise e

        return total_saved

    @staticmethod
    def _set_status(movie_name: str, status: int):
        """内部工具：更新 highlight_status"""
        db_path = settings.prefs.db_path
        conn = get_db_connection(db_path)
        c = conn.cursor()
        c.execute(
            "UPDATE movies_meta SET highlight_status = ? WHERE movie_name = ?", 
            (status, movie_name)
        )
        conn.commit()
        conn.close()