import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    app_name: str = "EvidenceFlow"
    app_env: str = os.getenv("APP_ENV", "development")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    default_data_dir: str = os.getenv("DATA_DIR", "data")

    @property
    def llm_enabled(self) -> bool:
        return bool(self.openai_api_key.strip())


settings = Settings()
