"""
训练脚本 — 在自有数据集上训练分类模型
"""
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np

from .config import config
from .features import load_audio, extract_mfcc
from .model import CryClassifier, save_model


def build_dataset(data_dir: str) -> tuple:
    """
    从目录结构构建数据集
    期望目录结构:
      data/
        train/
          饥饿/
            xxx.wav
          ...
    返回: (X_train, X_test, y_train, y_test)
    """
    X, y = [], []

    for label_idx, class_name in enumerate(config.class_names):
        class_dir = Path(data_dir) / class_name
        if not class_dir.exists():
            print(f"[WARN] 目录不存在: {class_dir}，跳过")
            continue

        for audio_file in class_dir.glob("*.wav"):
            try:
                y_audio = load_audio(str(audio_file))
                mfcc = extract_mfcc(y_audio)
                mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
                X.append(mfcc)
                y.append(label_idx)
            except Exception as e:
                print(f"[WARN] 跳过 {audio_file.name}: {e}")

    if not X:
        raise RuntimeError(f"未找到训练数据，请将 wav 文件放入 {data_dir}/<类别>/ 子目录")

    X = np.array(X)[:, np.newaxis, :, :]  # (N, 1, H, W)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"数据集: 训练 {len(X_train)} 条, 测试 {len(X_test)} 条")
    return X_train, X_test, y_train, y_test


def train(data_dir: str = "./data/train") -> str:
    """
    训练模型，返回保存路径
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    X_train, X_test, y_train, y_test = build_dataset(data_dir)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size)

    model = CryClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    best_acc = 0.0

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                _, preds = torch.max(outputs, 1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)
        acc = correct / total

        scheduler.step(1 - acc)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{config.epochs} | "
                f"Loss: {train_loss / len(train_loader):.4f} | "
                f"Acc: {acc:.4f}"
            )

        if acc > best_acc:
            best_acc = acc
            os.makedirs(config.model_dir, exist_ok=True)
            save_path = os.path.join(config.model_dir, "cry_model_best.pt")
            save_model(model, save_path)

    print(f"\n最佳准确率: {best_acc:.4f}")
    print(f"模型保存到: {save_path}")
    return save_path


if __name__ == "__main__":
    train()
