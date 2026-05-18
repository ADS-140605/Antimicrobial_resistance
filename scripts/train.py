from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raman_amr.data import RamanNpzDataset, collate_multimodal, stratified_split
from raman_amr.training import build_model, load_checkpoint, save_checkpoint, set_seed, train_one_epoch, evaluate_model, write_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Raman classification models.")
    parser.add_argument("--data", required=True, help="Path to an .npz file with spectral, wavelet, and labels arrays.")
    parser.add_argument("--model", choices=["multimodal", "raman_only", "wavelet_only", "svm"], default="multimodal")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.model == "svm":
        raise SystemExit("Use `scripts/evaluate.py` or a dedicated SVM experiment for the scikit-learn baseline.")

    dataset = RamanNpzDataset(args.data)
    train_set, validation_set, test_set = stratified_split(dataset, train_ratio=args.train_ratio, validation_ratio=args.validation_ratio, seed=args.seed)

    collate_fn = collate_multimodal
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device(args.device)
    model = build_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_validation_accuracy = -1.0
    best_checkpoint_path = Path(args.output_dir) / f"{args.model}_best.pt"

    for epoch in range(1, args.epochs + 1):
        train_result = train_one_epoch(model, train_loader, optimizer, criterion, device, args.model)
        validation_result = evaluate_model(model, validation_loader, criterion, device, args.model)
        scheduler.step(validation_result["accuracy"])

        print(
            f"Epoch {epoch:02d} | train loss {train_result.loss:.4f} acc {train_result.accuracy:.4f} | "
            f"val loss {validation_result['loss']:.4f} acc {validation_result['accuracy']:.4f}"
        )

        if validation_result["accuracy"] > best_validation_accuracy:
            best_validation_accuracy = validation_result["accuracy"]
            save_checkpoint(
                best_checkpoint_path,
                model,
                {
                    "model": args.model,
                    "epoch": epoch,
                    "best_validation_accuracy": best_validation_accuracy,
                    "seed": args.seed,
                },
            )

    print(f"Best checkpoint saved to {best_checkpoint_path}")
    load_checkpoint(best_checkpoint_path, model, device)
    test_metrics = evaluate_model(model, test_loader, criterion, device, args.model)
    write_metrics(Path(args.output_dir) / f"{args.model}_test_metrics.json", test_metrics)
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
