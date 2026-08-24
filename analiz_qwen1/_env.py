"""API anahtarı yalnızca ortam değişkeni veya git'e girmeyen .env dosyasından."""
from __future__ import annotations

import os
from pathlib import Path

_ENV_NAME = "DEEPSEEK_API_KEY"


def deepseek_api_key() -> str:
    key = (os.environ.get(_ENV_NAME) or "").strip()
    if key:
        return key
    env_file = Path(__file__).resolve().parent / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            if name.strip() == _ENV_NAME:
                key = value.strip().strip('"').strip("'")
                break
    if not key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY yok. .env.example dosyasını .env olarak kopyalayıp "
            "anahtarı oraya yazın (.env commit edilmez)."
        )
    return key
