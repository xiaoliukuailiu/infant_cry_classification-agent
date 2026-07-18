"""
配置中心
"""
import os
from pathlib import Path
from dataclasses import dataclass


def _load_env():
    """手动加载 .env，不依赖 python-dotenv（避免跨进程问题）"""
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and val and key not in os.environ:
                    os.environ[key] = val


_load_env()


@dataclass
class Config:
    # ========== LLM ==========
    llm_model: str = "deepseek-chat"
    llm_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    llm_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    # ========== 音频 ==========
    sample_rate: int = 16000
    duration_ms: int = 5000
    n_mfcc: int = 40
    n_mels: int = 128

    # ========== 模型 ==========
    model_dir: str = "./models"
    num_classes: int = 6
    class_names: tuple = ("angry", "disturbance", "fright", "hungry", "pain", "uncomfortable")
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-3

    # ========== Agent ==========
    max_rounds: int = 5


config = Config()

# 方便排查：打印加载状态
if not config.llm_api_key:
    import sys
    env_path = Path(__file__).parent.parent / ".env"
    print(f"[config] WARNING: DEEPSEEK_API_KEY not loaded (env exists: {env_path.exists()})", file=sys.stderr)
