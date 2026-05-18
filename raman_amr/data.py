from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset


class RamanNpzDataset(Dataset):
    """Dataset backed by a `.npz` file with Raman tensors and labels.

    Expected arrays:
    - spectral: shape (N, 1000) or (N, 1, 1000)
    - wavelet: shape (N, 3, 224, 224) or (N, 224, 224, 3)
    - labels: shape (N,)
    """

    def __init__(
        self,
        npz_path: str | Path,
        spectral_key: str = "spectral",
        wavelet_key: str = "wavelet",
        labels_key: str = "labels",
    ) -> None:
        path = Path(npz_path)
        if not path.exists():
            raise FileNotFoundError(path)

        data = np.load(path, allow_pickle=False)
        missing_keys = [key for key in (spectral_key, wavelet_key, labels_key) if key not in data]
        if missing_keys:
            missing = ", ".join(missing_keys)
            raise KeyError(f"Missing keys in {path}: {missing}")

        self.spectral = np.asarray(data[spectral_key], dtype=np.float32)
        self.wavelet = np.asarray(data[wavelet_key], dtype=np.float32)
        self.labels = np.asarray(data[labels_key], dtype=np.int64)

        if len(self.spectral) != len(self.wavelet) or len(self.spectral) != len(self.labels):
            raise ValueError("Spectral, wavelet, and label arrays must have the same length")

        if self.wavelet.ndim == 4 and self.wavelet.shape[-1] in {1, 3}:
            self.wavelet = np.moveaxis(self.wavelet, -1, 1)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        spectral = torch.from_numpy(self.spectral[index])
        wavelet = torch.from_numpy(self.wavelet[index])
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return spectral, wavelet, label


def stratified_split(
    dataset: Dataset,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[Subset[Dataset], Subset[Dataset], Subset[Dataset]]:
    """Split a dataset into train/validation/test subsets with stratification."""

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be less than 1")

    labels = np.asarray([int(dataset[index][2]) for index in range(len(dataset))])
    indices = np.arange(len(dataset))

    train_indices, temp_indices = train_test_split(
        indices,
        train_size=train_ratio,
        random_state=seed,
        stratify=labels,
    )

    temp_labels = labels[temp_indices]
    if validation_ratio == 0:
        return Subset(dataset, train_indices.tolist()), Subset(dataset, [],), Subset(dataset, temp_indices.tolist())

    validation_size = validation_ratio / (1.0 - train_ratio)
    validation_indices, test_indices = train_test_split(
        temp_indices,
        train_size=validation_size,
        random_state=seed,
        stratify=temp_labels,
    )

    return (
        Subset(dataset, train_indices.tolist()),
        Subset(dataset, validation_indices.tolist()),
        Subset(dataset, test_indices.tolist()),
    )


def collate_multimodal(batch):
    spectral, wavelet, labels = zip(*batch)
    return torch.stack(list(spectral)), torch.stack(list(wavelet)), torch.stack(list(labels))
