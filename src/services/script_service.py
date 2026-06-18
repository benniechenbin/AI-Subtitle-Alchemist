from src.components import ScriptWriterPromptBuilder
from src.models import ScriptGenerationRequest, ScriptGenerationResult
from src.prompts import SCRIPT_WRITER_SYSTEM
from src.services.llm import call_llm_api
from src.services.search import search_semantic


class ScriptService:
    """AI 编剧业务编排（无 UI 依赖）。"""

    @staticmethod
    def generate(
        request: ScriptGenerationRequest,
        model,
    ) -> ScriptGenerationResult:
        results = search_semantic(
            request.prompt[:200],
            model,
            request.db_path,
            embedding_model_name=request.embedding_model,
            final_k=30,
            target_movie=request.target_movie,
            allow_duplicates=request.allow_duplicates,
        )
        if not results:
            raise ValueError("未找到相关素材，请尝试更换关键词。")

        context_text = ScriptWriterPromptBuilder.build_context_from_results(results)
        user_prompt = ScriptWriterPromptBuilder.build_user_prompt(
            request.script_style,
            request.prompt,
            context_text,
        )
        script = call_llm_api(
            SCRIPT_WRITER_SYSTEM,
            user_prompt,
            request.llm_key,
            request.llm_model_name,
            request.llm_base_url,
        )
        return ScriptGenerationResult(script=script, material_count=len(results))
