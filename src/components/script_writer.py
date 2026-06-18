"""AI 编剧：提示词组装组件。"""

_USER_PROMPT_TEMPLATE = """
【创作风格】
{script_style}

【用户创作需求】
{director_prompt}

【可用素材库 (Reference Material)】
{context_text}

请开始编写脚本：
"""


class ScriptWriterPromptBuilder:
    """将风格、导演指令与素材库组装为 LLM 用户提示词。"""

    @staticmethod
    def build_user_prompt(
        script_style: str,
        director_prompt: str,
        context_text: str,
    ) -> str:
        return _USER_PROMPT_TEMPLATE.format(
            script_style=script_style,
            director_prompt=director_prompt,
            context_text=context_text,
        )

    @staticmethod
    def format_material_line(source_tag: str, timestamp: str, content: str) -> str:
        """单条素材的标准格式。"""
        full_tag = f"[{source_tag} {timestamp}]"
        return f"素材ID: {full_tag}\n台词内容: {content}\n\n"

    @staticmethod
    def build_context_from_results(results: list[dict]) -> str:
        """从检索结果批量生成素材库文本。"""
        lines = []
        for res in results:
            source_tag = f"《{res['movie']}》"
            season, episode = res.get("season"), res.get("episode")
            if season and episode:
                source_tag += f"S{str(season).zfill(2)}E{str(episode).zfill(2)}"
            lines.append(
                ScriptWriterPromptBuilder.format_material_line(
                    source_tag, res["time"], res["content"]
                )
            )
        return "".join(lines)
