# Raman AMR Models

PyTorch implementations of the multimodal CNN described in the paper and the comparison models it evaluates.

## Included models

- `MultimodalRamanCNN`: raw spectrum branch + CWT branch with late concatenation fusion.
- `RamanOnlyCNN`: 1D baseline that uses only raw spectra.
- `WaveletOnlyCNN`: 2D baseline that uses only the wavelet image.
- `RamanSVMClassifier`: scikit-learn RBF SVM baseline using concatenated spectral and wavelet features.

## Usage

```bash
pip install -r requirements.txt
python scripts/inspect_models.py
```

## Input shapes

- Spectral input: `(batch, 1000)` or `(batch, 1, 1000)`
- Wavelet input: `(batch, 3, 224, 224)`

The code focuses on the model definitions. You can plug these into your own training loop or dataset loader.

## Training and evaluation

The repository now includes simple CLI scripts for `.npz` datasets with `spectral`, `wavelet`, and `labels` arrays.

Train a CNN:

```bash
python scripts/train.py --data path/to/dataset.npz --model multimodal --epochs 40
```

Evaluate a trained CNN:

```bash
python scripts/evaluate.py --data path/to/dataset.npz --model multimodal --checkpoint outputs/multimodal_best.pt
```

Evaluate the SVM baseline:

```bash
python scripts/evaluate.py --data path/to/dataset.npz --model svm
```

