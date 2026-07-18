# 👶 婴儿哭声分类 Agent

> 上传一段婴儿哭声 → AI 自动分析：**饿了 / 疼了 / 困了**

将 **语音信号处理 + PyTorch 深度学习 + ReAct Agent** 串成一条端到端流水线。

---

## ✨ 特性

- 🎤 **音频特征提取**：MFCC、梅尔频谱自动提取
- 🧠 **CNN 分类模型**：4 层卷积网络，轻量高效
- 🤖 **ReAct Agent**：Function Calling 调用本地模型推理，并用自然语言解读结果
- 🖥️ **Streamlit Web UI**：上传音频、查看波形/特征图、Agent 对话解释
- 📊 **一目了然**：波形图 + MFCC 特征图可视化

---

## 🏗 系统架构

```
用户上传 wav/mp3
      │
      ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Streamlit   │────▶│  ReAct Agent │────▶│  CNN 模型     │
│  Web UI      │     │  (DeepSeek)  │     │  (PyTorch)    │
└──────────────┘     └──────────────┘     └──────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │  自然语言     │
                     │  分析报告     │
                     └──────────────┘
```

---

## 🚀 快速开始

### 1. 环境准备

```powershell
conda activate rag-agent
pip install torch librosa soundfile scikit-learn matplotlib streamlit
pip install -e .
```

### 2. 配置 API Key

```ini
# .env
DEEPSEEK_API_KEY=sk-xxxx
```

### 3. 准备训练数据

```
data/
  train/
    饥饿/
      cry001.wav
      cry002.wav
      ...
    疼痛/
      cry101.wav
      ...
    困倦/
      cry201.wav
      ...
```

> 💡 推荐数据集：[Baby Chillanto](http://ingenieria.uao.edu.co/baby-chillanto/) 或自录婴儿哭声

### 4. 训练模型

```powershell
cry-agent train ./data/train/
```

### 5. 命令行分析

```powershell
cry-agent analyze ./data/test/cry001.wav
```

### 6. Web 界面

```powershell
cry-agent ui
```

浏览器访问 `http://localhost:8501`

---

## 📂 项目结构

```
cry-classification-agent/
├── pyproject.toml
├── .env
├── .env.example
├── README.md
├── data/
│   └── train/                # 训练数据（按类别分子目录）
├── models/
│   └── cry_model_best.pt     # 训练好的模型
└── src/
    ├── config.py             # 配置中心
    ├── features.py           # MFCC / 梅尔频谱提取
    ├── model.py              # PyTorch CNN 模型
    ├── train.py              # 训练脚本
    ├── agent.py              # ReAct Agent 核心
    ├── prompts.py            # System Prompt
    ├── tools.py              # Function Calling 定义
    ├── main.py               # CLI 入口
    └── ui/
        └── app.py            # Streamlit Web UI
```

---

## 🔧 技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| LLM | DeepSeek (deepseek-chat) | Agent 推理与结果解读 |
| 深度学习 | PyTorch + CNN (4层) | 哭声分类模型 |
| 音频处理 | Librosa | MFCC / 梅尔频谱特征提取 |
| Agent 框架 | ReAct + Function Calling | 自主工具调用与决策 |
| Web UI | Streamlit | 音频上传 + 特征可视化 + 对话 |

---

## 📝 简历描述

> **婴儿哭声分类 Agent** | Python · PyTorch · Librosa · Streamlit
>
> - 设计 4 层 CNN 婴儿哭声分类模型，基于 MFCC 特征提取实现「饥饿/疼痛/困倦」三分类
> - 将 ReAct Agent 与训练好的 PyTorch 模型结合，通过 Function Calling 调用本地推理，实现端到端音频推理流水线
> - 使用 Streamlit 搭建 Web 界面，支持音频上传、波形/频谱可视化、Agent 自然语言解释
