import json
import shutil
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# =====================================================================
# 轨 1：系统绝对路径常量（由代码动态计算，绝不写入 .env）
# =====================================================================
def find_project_root(current_path: Path, markers: tuple = ("pyproject.toml", "requirements.txt", ".git")) -> Path:
    for parent in current_path.parents:
        if any((parent / marker).exists() for marker in markers):
            return parent
    return current_path.parent

# 替换掉原来的写死层级的代码：
PROJECT_ROOT = find_project_root(Path(__file__).resolve())
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
PREFERENCES_FILE = PROJECT_ROOT / "user_preferences.json"
LEGACY_CONFIG_FILE = PROJECT_ROOT / "config.json"
DB_NAME = "subtitle_library.db"
DEFAULT_DB_PATH = DATA_DIR / DB_NAME
LEGACY_DB_PATH = PROJECT_ROOT / DB_NAME


def ensure_data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def migrate_legacy_db() -> bool:
    """将项目根目录下的旧数据库迁移到 data/。"""
    ensure_data_dir()
    if LEGACY_DB_PATH.is_file() and not DEFAULT_DB_PATH.is_file():
        shutil.move(str(LEGACY_DB_PATH), str(DEFAULT_DB_PATH))
        return True
    return False


def normalize_db_path(db_path: str | None) -> str:
    """将配置中的旧路径规范为 data/ 下的默认路径。"""
    if not db_path:
        return str(DEFAULT_DB_PATH)
    normalized = Path(db_path).expanduser()
    if normalized == LEGACY_DB_PATH or normalized == PROJECT_ROOT / DB_NAME:
        return str(DEFAULT_DB_PATH)
    return str(normalized)


# =====================================================================
# 轨 2：安全凭证类（严格映射 .env 文件，负责拦截敏感信息）
# =====================================================================
class EnvSettings(BaseSettings):
    tmdb_api_key: str = ""
    deepseek_api_key: str = ""
    openai_api_key: str = ""
    gemini_api_key: str = ""
    spotify_client_id: str = ""
    spotify_client_secret: str = ""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


# =====================================================================
# 轨 3：用户偏好类（映射 user_preferences.json，负责 WebUI 动态交互）
# =====================================================================
class UserPreferences(BaseModel):
    library_path: str = Field(
        default_factory=lambda: str(Path.home() / "Movies" / "Subtitles")
    )
    embedding_model: str = "moka-ai/m3e-base"
    db_path: str = Field(default_factory=lambda: str(DEFAULT_DB_PATH))
    llm_provider: str = "DeepSeek"
    llm_model_name: str = "deepseek-chat"
    llm_base_url: str = "https://api.deepseek.com"


# =====================================================================
# 统一调度中心
# =====================================================================
class SettingsManager:
    def __init__(self):
        ensure_data_dir()
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.env = EnvSettings()
        self.prefs = self._load_preferences()

    def _load_preferences(self) -> UserPreferences:
        if PREFERENCES_FILE.exists():
            try:
                with open(PREFERENCES_FILE, "r", encoding="utf-8") as f:
                    prefs = UserPreferences(**json.load(f))
            except Exception:
                prefs = UserPreferences()
        else:
            prefs = UserPreferences()

        normalized = normalize_db_path(prefs.db_path)
        if prefs.db_path != normalized:
            prefs = prefs.model_copy(update={"db_path": normalized})
        return prefs

    def save_preferences(self, updated_dict: dict) -> None:
        current_data = self.prefs.model_dump()
        current_data.update(updated_dict)
        if "db_path" in current_data:
            current_data["db_path"] = normalize_db_path(current_data["db_path"])
        self.prefs = UserPreferences(**current_data)
        with open(PREFERENCES_FILE, "w", encoding="utf-8") as f:
            json.dump(self.prefs.model_dump(), f, ensure_ascii=False, indent=4)

    def get_llm_api_key(self) -> str:
        """从 .env 读取 API Key。不再从 user_preferences.json 读取。"""
        provider_keys = {
            "DeepSeek": self.env.deepseek_api_key,
            "OpenAI": self.env.openai_api_key,
            "Google": self.env.gemini_api_key,
        }
        return provider_keys.get(self.prefs.llm_provider, "")

    def migrate_legacy_config(self) -> bool:
        """将旧版 config.json 合并进 user_preferences.json。"""
        if not LEGACY_CONFIG_FILE.exists():
            return False

        try:
            with open(LEGACY_CONFIG_FILE, "r", encoding="utf-8") as f:
                legacy = json.load(f)
        except Exception:
            return False

        field_map = {
            "library_path": "library_path",
            "embedding_model": "embedding_model",
            "db_path": "db_path",
            "llm_provider": "llm_provider",
            "llm_model_name": "llm_model_name",
            "llm_base_url": "llm_base_url",
        }
        updates = {
            target: legacy[source]
            for source, target in field_map.items()
            if source in legacy
        }
        if updates:
            self.save_preferences(updates)
            return True
        return False


settings = SettingsManager()
