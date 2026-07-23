from openai import OpenAI
from src.observability.logger import logger


class LLMCallError(RuntimeError):
    """LLM API 调用失败的自定义异常"""

    pass


def call_llm_api(
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    model_name: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com",
) -> str:
    if not api_key and "ollama" not in base_url.lower():
        raise ValueError("未配置 API Key")

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=False,
            temperature=0.7,
            max_tokens=4000,
        )
        content = response.choices[0].message.content
        return content or ""
    except Exception as e:
        error_msg = f"LLM API 调用失败 | 模型: {model_name} | 错误: {str(e)}"
        logger.error(f"❌ {error_msg}")
        raise LLMCallError(error_msg) from e
