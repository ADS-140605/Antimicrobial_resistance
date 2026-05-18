from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raman_amr.data import RamanNpzDataset, collate_multimodal, stratified_split
from raman_amr.models import MultimodalRamanCNN, RamanOnlyCNN, WaveletOnlyCNN
from raman_amr.svm import RamanSVMClassifier
from raman_amr.training import evaluate_model, load_checkpoint, write_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Raman classification models.")
    parser.add_argument("--data", required=True, help="Path to an .npz file with spectral, wavelet, and labels arrays.")
    parser.add_argument("--model", choices=["multimodal", "raman_only", "wavelet_only", "svm"], default="multimodal")
    parser.add_argument("--checkpoint", help="Path to a trained PyTorch checkpoint for CNN models.")
    parser.add_argument("--output", default="evaluation.json")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = RamanNpzDataset(args.data)
    train_set, validation_set, test_set = stratified_split(dataset, train_ratio=args.train_ratio, validation_ratio=args.validation_ratio, seed=args.seed)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_multimodal)

    if args.model == "svm":
        spectral = np.stack([sample[0].numpy() for sample in dataset])
        wavelet = np.stack([sample[1].numpy() for sample in dataset])
        labels = np.stack([sample[2].numpy() for sample in dataset])
        svm = RamanSVMClassifier()
        svm.fit(spectral[train_set.indices], wavelet[train_set.indices], labels[train_set.indices])
        predictions = svm.predict(spectral[test_set.indices], wavelet[test_set.indices])
        test_labels = labels[test_set.indices]
        metrics = {
            "accuracy": float((predictions == test_labels).mean()),
            "predictions": predictions.tolist(),
            "labels": test_labels.tolist(),
        }
        write_metrics(args.output, metrics)
        print(f"Test accuracy: {metrics['accuracy']:.4f}")
        return

    device = torch.device(args.device)
    if args.model == "multimodal":
        model = MultimodalRamanCNN()
    elif args.model == "raman_only":
        model = RamanOnlyCNN()
    else:
        model = WaveletOnlyCNN()

    if not args.checkpoint:
        raise SystemExit("CNN evaluation requires --checkpoint")

    load_checkpoint(args.checkpoint, model, device)
    criterion = torch.nn.CrossEntropyLoss()
    metrics = evaluate_model(model.to(device), test_loader, criterion, device, args.model)
    write_metrics(args.output, metrics)
    print(f"Test accuracy: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
