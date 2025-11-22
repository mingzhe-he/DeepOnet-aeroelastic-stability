# Tier 1: Scalar Surrogate Model - Standalone Package

This package contains the implementation of the **Tier 1 Scalar Surrogate Model** for predicting Strouhal number and aerodynamic coefficients of triangular shapes. It uses an Ensemble of Multi-Layer Perceptrons (MLPs) and includes a robust preprocessing pipeline.

## 📦 Package Contents

- `src/`: Source code for data processing, modeling, training, and evaluation.
- `experiments/tier1_baseline/`: Configuration files and results directory.
- `data/splits/`: JSON files defining the training/validation/test splits for reproducibility.
- `data/processed/`: Contains `summary.parquet` (pre-processed data) for quick start.
- `notebooks/`: Jupyter notebooks for data preprocessing.
- `pyproject.toml`: Dependency definitions.

## 🚀 Quick Start

### 1. Prerequisites

- **Python 3.11**
- **uv** (recommended for dependency management) or `pip`.

### 2. Installation

We recommend using `uv` for a fast and isolated environment setup.

```bash
# 1. Create a virtual environment
uv venv

# 2. Activate the environment
source .venv/bin/activate

# 3. Install dependencies
uv pip install -e .
```

If using standard `pip`:
```bash
pip install -e .
```

### 3. Running Training (Immediate)

The package comes with pre-processed data (`data/processed/summary.parquet`). You can run the baseline training script immediately using `uv`:

```bash
uv run python src/training/train_tier1.py --config experiments/tier1_baseline/config.yaml
```

This will:
1.  Train an **Ensemble of 5 MLPs**.
2.  Train a **Ridge Regression** baseline.
3.  Evaluate on the test set (AoA Interpolation split).
4.  Save results, metrics, and plots to `experiments/tier1_baseline/results/`.

### 4. Reproducing Preprocessing from Raw DES Data

If you have access to the raw OpenFOAM data (folders such as `baseline/`, `baseline_lowU/`, `baseline_mediumU/`, `shorter/`, `higher/` under your DES root), you can rebuild `summary.parquet` in a single command:

```bash
uv run python -m src.data.preprocessing \
  --raw-root /Volumes/MHSSD/Projects/aeroelasticity/data \
  --output data/processed/summary.parquet
```

This will:

- Discover all available cases.
- Correct the force coefficients for **simulation velocity** and **shape depth**.
- Compute time-averaged (`mean_Cd`, `mean_Cl`, `mean_Cm`) and fluctuation statistics.
- Extract spectral descriptors (`St_peak`, `A_peak`, `Q`, etc.) from the 4000 Hz `C_L(t)` signals.

## 📊 Baseline Tier 1 Model Details

- **Input Features**: $D, H, H/D, Re, \sin(\alpha), \cos(\alpha)$
- **Targets**: $St_{peak}, A_{peak}, Q, \\overline{C_d}, \\overline{C_l}, \\overline{C_m}$
- **Architecture**: Ensemble of 5 MLPs (64-64-32 hidden units, ReLU, BatchNorm, Dropout).
- **Strategy**: AoA Interpolation (Training on subset of angles, testing on unseen angles).

## 📂 Output Structure

After training, check `experiments/tier1_baseline/results/` for:
- `metrics.json`: Detailed performance metrics.
- `predictions_vs_actual_mlp.png`: Parity plots.
- `mean_coeffs_vs_aoa.png`: Aerodynamic curves comparison.
- `ensemble_metadata.pt`: Saved metadata for inference.
- `mlp_model_*.pt`: Saved model checkpoints.

## 🔬 Additional Surrogate Approaches

The repository now implements four distinct approaches:

1. **Improved Tier 1 Ensemble MLP (Approach 1)**
   - Physics-informed features including AoA shift around 90°.
   - Target-wise weighted MSE loss to balance Strouhal, Q and mean coefficients.
   - Run:
     ```bash
     uv run python src/training/train_tier1_improved.py \
       --config experiments/tier1_improved/config.yaml
     ```

2. **Multi-task MLP with Derivative Regularisation (4B, Approach 2)**
   - Implements Recommendation 0, Section 4B.
   - Inputs: $(H/D, Re, \\alpha_0, \\sin\\alpha_0, \\cos\\alpha_0)$ with $\\alpha_0 = \\alpha - 90^\\circ$.
   - Outputs: $[St_{peak}, \\overline{C_d}, \\overline{C_l}, \\overline{C_m}]$.
   - Uses autograd to match $\\partial \\langle C_L \\rangle / \\partial \\alpha$ to finite-difference estimates and plots a Den Hartog diagram based on the surrogate.
   - Run:
     ```bash
     uv run python src/training/train_mlp_derivative.py \
       --config experiments/tier1_4b_mlp/config.yaml
     ```

3. **Physics-informed ROM (Approach 3)**
   - GP-based ROM for $\\langle C_L \\rangle$, $\\langle C_D \\rangle$ vs $(H/D, \\alpha)$ and Den Hartog coefficient.
   - Regime-aware Strouhal surrogate combining clustering, RF classification and per-regime GP/Ridge.
   - Generates ROM-based Den Hartog diagrams and Strouhal vs AoA/Re plots.
   - Run:
     ```bash
     uv run python src/training/train_rom.py \
       --config experiments/tier1_rom/config.yaml
     ```

4. **LightGBM Surrogate (Approach 4)**
   - Gradient-boosted trees on engineered features
     $[H/D, Re, U_{ref}, \\alpha, \\alpha^2, \\sin\\alpha, \\cos\\alpha, \\sin 2\\alpha, \\text{shape\_type}]$.
   - Targets: $[St_{peak}, \\overline{C_l}, \\overline{C_d}, \\overline{C_m}, \\text{std}(C_l)]$.
   - Uses K-fold CV with out-of-fold parity plots for robust performance estimates.
   - Run:
     ```bash
     uv run python src/training/train_lightgbm.py \
       --config experiments/tier1_lightgbm/config.yaml
     ```

## 🛠️ Troubleshooting

- **Missing Dependencies**: Ensure you installed with `-e .` to install the project in editable mode.
- **Path Errors**: Run scripts from the root of this package directory.
