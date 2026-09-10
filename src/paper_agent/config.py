"""配置加载：config.yaml 提供平台与流水线参数，API Key 从 .env 读取。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
ENV_PATH = PROJECT_ROOT / ".env"


@dataclass
class EndpointConfig:
    """一个 OpenAI 兼容服务端点（LLM 与 embedding 共用结构）。"""

    base_url: str
    model: str
    api_key: str = ""


def load_config() -> dict:
    """加载 config.yaml；顺带加载 .env，使 api_key_env 指向的环境变量生效。"""
    load_dotenv(ENV_PATH)
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_endpoint(raw: dict) -> EndpointConfig:
    """从配置节解析端点；Key 从 api_key_env 指定的环境变量读取，缺失时返回空串。"""
    env_name = raw.get("api_key_env", "LLM_API_KEY")
    return EndpointConfig(
        base_url=raw.get("base_url", ""),
        model=raw.get("model", ""),
        api_key=os.environ.get(env_name, ""),
    )
