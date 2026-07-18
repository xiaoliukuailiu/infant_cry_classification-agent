"""
Streamlit Web UI — 婴儿哭声分析
上传音频 → Agent 推理 → 自然语言解释
"""
import os
import sys
import tempfile
from pathlib import Path

# ── 确保 .env 已加载（无论从哪启动） ──────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"
if _ENV_FILE.exists():
    with open(_ENV_FILE, encoding="utf-8-sig") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                _k, _v = _k.strip(), _v.strip().strip('"').strip("'")
                if _k and _v and _k not in os.environ:
                    os.environ[_k] = _v

sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from src.config import config
from src.agent import CryAnalysisAgent
from src.features import load_audio, extract_mfcc, extract_melspectrogram

st.set_page_config(
    page_title="婴儿哭声分析",
    page_icon="👶",
    layout="wide",
)

MODEL_PATH = os.getenv("CRY_MODEL_PATH", "./models/cry_model_best.pt")


# ── 缓存：加载 Agent ──────────────────────────
@st.cache_resource(show_spinner="正在加载模型...")
def load_agent(model_path: str):
    if not os.path.exists(model_path):
        return None
    return CryAnalysisAgent(config, model_path)


# ── 标题 ──────────────────────────────────────
st.title("👶 婴儿哭声分析 Agent")
st.markdown(
    "*上传一段婴儿哭声，AI 自动分析：是饿了？疼了？还是困了？*"
)

# ── 侧边栏 ────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 系统信息")
    st.markdown(
        f"""
        | 组件 | 配置 |
        |------|------|
        | LLM | `{config.llm_model}` |
        | 模型 | 4层 CNN |
        | 特征 | MFCC ({config.n_mfcc}维) |
        | 采样率 | {config.sample_rate} Hz |
        | 输入长度 | {config.duration_ms} ms |
        """
    )
    st.divider()
    st.markdown(f"📂 模型路径: `{MODEL_PATH}`")

    st.divider()
    st.header("📖 类别说明")
    st.markdown(
        """
        | 类别 | 含义 |
        |------|------|
        | 😡 angry | 生气 |
        | 😫 disturbance | 烦躁 |
        | 😨 fright | 惊吓 |
        | 🍼 hungry | 饥饿 |
        | 😖 pain | 疼痛 |
        | 😣 uncomfortable | 不适 |
        """
    )
    st.divider()
    st.markdown("*💡 支持 wav/mp3 格式，建议 5 秒左右*")


# ── 主区域 ────────────────────────────────────
agent = load_agent(MODEL_PATH)

if agent is None:
    st.error("❌ 模型文件不存在！请先训练模型：")
    st.code("cry-agent train ./data/train/")
    st.stop()

st.success("✅ 模型已加载，请上传音频文件")

uploaded = st.file_uploader(
    "上传婴儿哭声音频",
    type=["wav", "mp3"],
    help="支持 wav 和 mp3 格式",
)

if uploaded is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.audio(uploaded, format="audio/wav")

        # 绘制特征图
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        try:
            y = load_audio(tmp_path)
            mfcc = extract_mfcc(y)

            fig, axes = plt.subplots(2, 1, figsize=(6, 4))

            axes[0].plot(np.linspace(0, len(y) / config.sample_rate, len(y)), y)
            axes[0].set_title("波形")
            axes[0].set_xlabel("时间 (s)")

            axes[1].imshow(mfcc, aspect="auto", origin="lower", cmap="viridis")
            axes[1].set_title("MFCC 特征")
            axes[1].set_xlabel("帧")
            axes[1].set_ylabel("MFCC 系数")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.error(f"特征提取失败: {e}")
            tmp_path = None

    with col2:
        st.subheader("🤖 Agent 分析结果")
        if tmp_path:
            with st.spinner("Agent 正在分析..."):
                result = agent.analyze(tmp_path)
            st.markdown(result)

            # 清理临时文件
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        else:
            st.warning("无法分析该音频文件")
