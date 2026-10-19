"""Model configuration loading (configs/models.yaml + environment variables)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "models.yaml"


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal .env loader: KEY=VALUE lines; existing env vars win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class ModelConfig:
    name: str
    base_url: str
    model: str
    api_key_env: str
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout: float = 120
    price_in: float | None = None   # CNY / 1M input tokens (None = unknown)
    price_out: float | None = None  # CNY / 1M output tokens
    extra_body: dict | None = None  # provider-specific request fields (e.g. DeepSeek thinking mode)

    @property
    def api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise RuntimeError(
                f"Environment variable {self.api_key_env} is not set "
                f"(needed by model '{self.name}'). Put it in .env or your environment."
            )
        return key

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float | None:
        if self.price_in is None or self.price_out is None:
            return None
        return (prompt_tokens * self.price_in + completion_tokens * self.price_out) / 1e6


def load_model(name: str, path: Path | str = DEFAULT_CONFIG) -> ModelConfig:
    load_dotenv()
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    models = data.get("models", {})
    if name not in models:
        raise KeyError(f"Unknown model '{name}'. Available: {', '.join(models)}")
    merged = {**data.get("defaults", {}), **models[name]}
    return ModelConfig(name=name, **merged)
