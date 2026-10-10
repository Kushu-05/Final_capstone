import os
from typing import Optional
from dotenv import load_dotenv

# Explicitly load .env from project root directory
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.env"))
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

class AppConfig:
    """Central configuration for AI Restaurant Operations Assistant."""
    def __init__(self):
        self.reload()

    def reload(self):
        # LLM Settings
        self.anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
        model_env = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
        if model_env in ["claude-3-5-sonnet-20241022", "claude-3-5-sonnet-latest"]:
            model_env = "claude-sonnet-5-5"
        self.anthropic_model: str = model_env

        self.nvidia_api_key: Optional[str] = os.getenv("NVIDIA_API_KEY")
        nvidia_m = os.getenv("NVIDIA_MODEL", "meta/llama-3.2-11b-vision-instruct")
        if nvidia_m in ["openai/gpt-oss-20b", ""]:
            nvidia_m = "meta/llama-3.2-11b-vision-instruct"
        self.nvidia_model: str = nvidia_m
        self.nvidia_base_url: str = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

        self.openrouter_api_key: Optional[str] = os.getenv("OPENROUTER_API_KEY")
        self.openrouter_model: str = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
        self.openrouter_base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

        self.gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        self.preferred_provider: str = os.getenv("LLM_PROVIDER", "auto").lower()
        self.openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self.llm_timeout: float = float(os.getenv("LLM_TIMEOUT", "8.0"))
        self.llm_max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "1"))
        self.data_dir: str = os.getenv("DATA_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data")))

        # PostgreSQL Production Database
        self.postgres_host: str = os.getenv("POSTGRES_HOST", "127.0.0.1")
        self.postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
        self.postgres_db: str = os.getenv("POSTGRES_DB", "postgres")
        self.postgres_user: str = os.getenv("POSTGRES_USER", "bharathyadav")
        self.postgres_password: str = os.getenv("POSTGRES_PASSWORD", "")

        # Qdrant Vector Engine
        self.qdrant_url: str = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
        self.qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "yelp_reviews_bge_m3")

        # Embedding & Reranker Models
        self.embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        self.reranker_model: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

    def get_active_provider(self) -> str:
        if self.preferred_provider == "local":
            return "local"
        if self.preferred_provider in ["anthropic", "claude"] and self.anthropic_api_key:
            return "anthropic"
        if self.preferred_provider in ["nvidia"] and self.nvidia_api_key:
            return "nvidia"
        if self.preferred_provider in ["openrouter"] and self.openrouter_api_key:
            return "openrouter"
        if self.preferred_provider in ["gemini", "google"] and self.gemini_api_key:
            return "gemini"
        if self.preferred_provider in ["openai"] and self.openai_api_key:
            return "openai"

        # In 'auto' mode: 1st Claude, 2nd NVIDIA, 3rd OpenRouter, 4th Gemini, 5th OpenAI, 6th Local
        if self.anthropic_api_key:
            return "anthropic"
        if self.nvidia_api_key:
            return "nvidia"
        if self.openrouter_api_key:
            return "openrouter"
        if self.gemini_api_key:
            return "gemini"
        if self.openai_api_key:
            return "openai"
        return "local"

    def set_api_key(self, provider: str, key: str):
        p = provider.lower()
        if p in ["anthropic", "claude"]:
            self.anthropic_api_key = key
            os.environ["ANTHROPIC_API_KEY"] = key
        elif p == "nvidia":
            self.nvidia_api_key = key
            os.environ["NVIDIA_API_KEY"] = key
        elif p == "openrouter":
            self.openrouter_api_key = key
            os.environ["OPENROUTER_API_KEY"] = key
        elif p == "openai":
            self.openai_api_key = key
            os.environ["OPENAI_API_KEY"] = key
        elif p in ["gemini", "google"]:
            self.gemini_api_key = key
            os.environ["GEMINI_API_KEY"] = key

config = AppConfig()
