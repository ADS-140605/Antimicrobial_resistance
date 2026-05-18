from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

import json
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader

from .models import MultimodalRamanCNN, RamanOnlyCNN, WaveletOnlyCNN
from .svm import RamanSVMClassifier


ModelName = Literal["multimodal", "raman_only", "wavelet_only", "svm"]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(model_name: ModelName, num_classes: int = 3):
    if model_name == "multimodal":
        return MultimodalRamanCNN(num_classes=num_classes)
    if model_name == "raman_only":
        return RamanOnlyCNN(num_classes=num_classes)
    if model_name == "wavelet_only":
        return WaveletOnlyCNN(num_classes=num_classes)
    if model_name == "svm":
        return RamanSVMClassifier()
    raise ValueError(f"Unknown model: {model_name}")


@dataclass
class EpochResult:
    loss: float
    accuracy: float


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device, model_name: ModelName) -> EpochResult:
    model.train()
    total_loss = 0.0
    predictions: list[int] = []
    targets: list[int] = []

    for spectral, wavelet, labels in loader:
        spectral = spectral.to(device)
        wavelet = wavelet.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(spectral, wavelet) if model_name == "multimodal" else model(spectral if model_name == "raman_only" else wavelet)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += float(loss.item()) * labels.size(0)
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())

    return EpochResult(loss=total_loss / len(loader.dataset), accuracy=accuracy_score(targets, predictions))


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, model_name: ModelName) -> dict:
    model.eval()
    total_loss = 0.0
    predictions: list[int] = []
    targets: list[int] = []

    for spectral, wavelet, labels in loader:
        spectral = spectral.to(device)
        wavelet = wavelet.to(device)
        labels = labels.to(device)

        logits = model(spectral, wavelet) if model_name == "multimodal" else model(spectral if model_name == "raman_only" else wavelet)
        loss = criterion(logits, labels)

        total_loss += float(loss.item()) * labels.size(0)
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        targets.extend(labels.cpu().tolist())

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(targets, predictions),
        "macro_precision": precision_score(targets, predictions, average="macro", zero_division=0),
        "macro_recall": recall_score(targets, predictions, average="macro", zero_division=0),
        "macro_f1": f1_score(targets, predictions, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(targets, predictions).tolist(),
        "classification_report": classification_report(targets, predictions, zero_division=0),
    }


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device, model_name: ModelName):
    model.eval()
    predictions: list[int] = []
    targets: list[int] = []

    for spectral, wavelet, labels in loader:
        spectral = spectral.to(device)
        wavelet = wavelet.to(device)
        logits = model(spectral, wavelet) if model_name == "multimodal" else model(spectral if model_name == "raman_only" else wavelet)
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        targets.extend(labels.tolist())

    return predictions, targets


def save_checkpoint(path: str | Path, model: nn.Module, metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, path)


def load_checkpoint(path: str | Path, model: nn.Module, device: torch.device) -> dict:
    checkpoint = torch.load(Path(path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint.get("metadata", {})


def write_metrics(path: str | Path, metrics: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))
