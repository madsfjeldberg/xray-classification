# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Binary chest X-ray image classification (NORMAL vs PNEUMONIA) using multiple ML models: MLP (PyTorch), Decision Tree, and Random Forest (scikit-learn). Academic exam project.

## Setup & Commands

```bash
# Install dependencies
uv sync

# Launch Jupyter
jupyter notebook

# Run individual model scripts directly
python models/mlp.py
python models/decisionTree.py
python models/randomForest.py
```

No test framework is configured. No linting or build steps beyond `uv sync`.

## Architecture

### Data Flow

1. **Load** — `util.py:load_dataset()` reads images from two possible directory layouts:
   - Flat: `path/NORMAL/` and `path/PNEUMONIA/`
   - Split: `path/train|test|val/NORMAL|PNEUMONIA/`
2. **Preprocess** — `util.py:normalize_image()` resizes to 224×224 grayscale, normalizes pixels to `[0, 1]` (float32). `normalize_and_save_images()` caches results to `normalized_images/`.
3. **Train & Evaluate** — each model exposes `run()` / `run_validate()` functions called from notebooks.
4. **Visualize** — `graphs.py:plot_loss_flattening()` detects training plateau (gradient < 10% of average) and highlights convergence.

### Models (`models/`)

| File | Framework | Entry point |
|---|---|---|
| `mlp.py` | PyTorch | `train_epoch()`, `evaluate()` |
| `decisionTree.py` | scikit-learn | `run()`, `run_validate()` |
| `randomForest.py` | scikit-learn | `run_validate()` |

MLP architecture: 150 528 → 512 (ReLU, Dropout 0.4) → 128 (ReLU, Dropout 0.3) → 1 (Sigmoid).

### Notebooks

- `main.ipynb` — primary training/evaluation workflow comparing all models
- `mlp.ipynb` — dedicated MLP experimentation

### Key Dependencies

- `torch` / `torchvision` — MLP training
- `scikit-learn` — Decision Tree, Random Forest
- `kagglehub` — dataset download
- `mlx` — Apple ML framework (available but usage is secondary)
- `uv` — package manager (use instead of pip)
