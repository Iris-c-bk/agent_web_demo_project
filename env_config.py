"""从环境变量读取 API Key；若已安装 python-dotenv，会加载与本文件同目录下的 .env。"""
import os
from pathlib import Path


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parent / ".env")
    except ImportError:
        pass


def load_api_key() -> str:
    _load_dotenv_if_available()

    key = os.environ.get("SILICONFLOW_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key or not key.strip():
        raise SystemExit(
            "未设置 API Key。\n"
            "任选一种方式：\n"
            "  1) 系统环境变量：SILICONFLOW_API_KEY 或 OPENAI_API_KEY\n"
            "  2) 在本目录创建 .env 文件，写入一行：SILICONFLOW_API_KEY=你的key\n"
            "     （需先执行：pip install python-dotenv）\n"
            "  Windows PowerShell 当前会话临时设置：\n"
            "     $env:SILICONFLOW_API_KEY='sk-你的key'"
        )
    return key.strip()


def load_tavily_api_key() -> str:
    _load_dotenv_if_available()

    key = os.environ.get("TAVILY_API_KEY")
    if not key or not key.strip():
        raise SystemExit(
            "未设置 TAVILY_API_KEY。\n"
            "任选一种方式：\n"
            "  1) 系统环境变量：TAVILY_API_KEY\n"
            "  2) 在本目录 .env 中增加一行：TAVILY_API_KEY=你的key\n"
            "     （需 pip install python-dotenv）\n"
            "  Windows PowerShell 临时设置：\n"
            "     $env:TAVILY_API_KEY='tvly-...'"
        )
    return key.strip()
