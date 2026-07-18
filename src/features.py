"""
音频特征提取 — MFCC + 梅尔频谱
"""
import numpy as np
import librosa
from .config import config


def load_audio(filepath: str) -> np.ndarray:
    """加载音频，统一采样率，裁切/填充到固定长度"""
    y, sr = librosa.load(filepath, sr=config.sample_rate, mono=True)

    target_len = config.sample_rate * config.duration_ms // 1000

    if len(y) > target_len:
        y = y[:target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)))

    return y


def extract_mfcc(y: np.ndarray) -> np.ndarray:
    """提取 MFCC 特征 (n_mfcc, time_frames)"""
    mfcc = librosa.feature.mfcc(
        y=y, sr=config.sample_rate, n_mfcc=config.n_mfcc
    )
    return mfcc


def extract_melspectrogram(y: np.ndarray) -> np.ndarray:
    """提取对数梅尔频谱 (n_mels, time_frames)"""
    mel = librosa.feature.melspectrogram(
        y=y, sr=config.sample_rate, n_mels=config.n_mels
    )
    return librosa.power_to_db(mel, ref=np.max)


def preprocess(filepath: str) -> np.ndarray:
    """
    完整预处理流水线
    返回: (1, 1, n_mfcc, time_frames) — 适配 CNN 输入
    """
    y = load_audio(filepath)
    mfcc = extract_mfcc(y)          # (n_mfcc, T)
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
    return mfcc[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)


def preprocess_mel(filepath: str) -> np.ndarray:
    """
    完整预处理流水线（梅尔频谱版）
    返回: (1, 1, n_mels, time_frames)
    """
    y = load_audio(filepath)
    mel = extract_melspectrogram(y)
    mel = (mel - mel.mean()) / (mel.std() + 1e-8)
    return mel[np.newaxis, np.newaxis, :, :]
