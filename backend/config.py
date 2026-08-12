from __future__ import annotations
import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")
    bloom_model_size: str = "0.5b"
    bloom_model_dir: str | None = None
    bloom_use_quantized: bool = False
    generator_model_path: str | None = None
    retrieval_encoder: str = "bge-small"
    offline_mode: bool = True
    data_dir: Path = ROOT / "data" / "web_sessions"
    max_upload_mb: int = 20
    max_question_chars: int = 4000
    bloom_gate_threshold: float = 0.40
    session_secret: str = "change-me-before-deployment"
    student_access_code: str = "student-local"
    teacher_access_code: str = "teacher-local"
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    def origins(self) -> list[str]: return [x.strip() for x in self.cors_origins.split(",") if x.strip()]

settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HUB_OFFLINE", "1" if settings.offline_mode else "0")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1" if settings.offline_mode else "0")
