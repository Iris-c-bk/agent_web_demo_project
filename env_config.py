"""从环境变量读取 API Key；若已安装 python-dotenv，会加载与本文件同目录下的 .env。
Streamlit Community Cloud：在应用后台填写的 Secrets 会出现在 st.secrets，此处会同步到 os.environ。"""
import os
from pathlib import Path


def _hydrate_streamlit_secrets() -> None:
    try:
        import streamlit as st

        secrets = getattr(st, "secrets", None)
        if secrets is None:
            return
        for key in ("SILICONFLOW_API_KEY", "OPENAI_API_KEY", "TAVILY_API_KEY"):
            try:
                if key not in secrets:
                    continue
                val = secrets[key]
                if val is not None and str(val).strip():
                    os.environ.setdefault(key, str(val).strip())
            except Exception:
                continue
    except Exception:
        pass


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parent / ".env")
    except ImportError:
        pass


def load_api_key() -> str:
    _hydrate_streamlit_secrets()
    _load_dotenv_if_available()

    key = os.environ.get("SILICONFLOW_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key or not key.strip():
        raise SystemExit(
            "未设置 API Key。\n"
            "任选一种方式：\n"
            "  1) Streamlit Cloud：打开已部署应用 → 右下角「⋯ / Manage app」→ Secrets，写入 SILICONFLOW_API_KEY\n"
            "  2) 系统环境变量：SILICONFLOW_API_KEY 或 OPENAI_API_KEY\n"
            "  3) 本目录 .env：SILICONFLOW_API_KEY=你的key（需 python-dotenv）\n"
            "  Windows PowerShell 临时：$env:SILICONFLOW_API_KEY='sk-你的key'"
        )
    return key.strip()


def load_tavily_api_key() -> str:
    _hydrate_streamlit_secrets()
    _load_dotenv_if_available()

    key = os.environ.get("TAVILY_API_KEY")
    if not key or not key.strip():
        raise SystemExit(
            "未设置 TAVILY_API_KEY。\n"
            "任选一种方式：\n"
            "  1) Streamlit Cloud：Manage app → Secrets，写入 TAVILY_API_KEY\n"
            "  2) 系统环境变量：TAVILY_API_KEY\n"
            "  3) 本目录 .env：TAVILY_API_KEY=你的key（需 python-dotenv）\n"
            "  Windows PowerShell：$env:TAVILY_API_KEY='tvly-...'"
        )
    return key.strip()
