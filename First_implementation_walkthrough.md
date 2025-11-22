# Tier 1 Implementation Walkthrough

I have implemented the four suggested approaches for the Tier 1 package. Here is a summary of the work done and the initial results.

## Implemented Approaches

### 1. Improved MLP Ensemble
- **Code**: [src/training/train_tier1_improved.py](file:///Users/mingz/Projects/tier1_package/src/training/train_tier1_improved.py)
- **Config**: [experiments/tier1_improved/config.yaml](file:///Users/mingz/Projects/tier1_package/experiments/tier1_improved/config.yaml)
- **Description**: Ensemble of MLPs with improved features (H/D, Re_star, AoA symmetry) and weighted loss.
- **Results**:
  - `mean_Cd`: R² = 0.96
  - `mean_Cm`: R² = 0.92
  - `mean_Cl`: R² = 0.88
  - `A_peak`: R² = 0.74
  - `St_peak`: R² = 0.68

### 2. Multi-task MLP with Derivative Regularization
- **Code**: [src/training/train_mlp_derivative.py](file:///Users/mingz/Projects/tier1_package/src/training/train_mlp_derivative.py)
- **Config**: [experiments/tier1_4b_mlp/config.yaml](file:///Users/mingz/Projects/tier1_package/experiments/tier1_4b_mlp/config.yaml)
- **Description**: MLP trained with explicit regularization on the derivative `d<C_L>/dα` using autograd.
- **Results**:
  - Initial results are poor (negative R² for some targets). This approach requires significant hyperparameter tuning (learning rate, lambda_deriv, batch size) and potentially more data or different normalization scaling for the derivative loss.

### 3. Physics-informed ROM
- **Code**: [src/rom/galloping_gp.py](file:///Users/mingz/Projects/tier1_package/src/rom/galloping_gp.py), [src/rom/shedding_model.py](file:///Users/mingz/Projects/tier1_package/src/rom/shedding_model.py), [src/rom/landau.py](file:///Users/mingz/Projects/tier1_package/src/rom/landau.py), [src/training/train_rom.py](file:///Users/mingz/Projects/tier1_package/src/training/train_rom.py)
- **Config**: [experiments/tier1_rom/config.yaml](file:///Users/mingz/Projects/tier1_package/experiments/tier1_rom/config.yaml)
- **Description**: Gaussian Process ROMs for mean coefficients and a regime-aware shedding model. Includes Landau-Stuart reconstruction.
- **Results**:
  - `St_peak`: R² = 0.25
  - Mean coefficients R² near 0. This suggests the GPs need kernel tuning (length scales) or the data is too sparse/noisy for the current GP configuration.

### 4. LightGBM Surrogate
- **Code**: [src/training/train_lightgbm.py](file:///Users/mingz/Projects/tier1_package/src/training/train_lightgbm.py)
- **Config**: [experiments/tier1_lightgbm/config.yaml](file:///Users/mingz/Projects/tier1_package/experiments/tier1_lightgbm/config.yaml)
- **Description**: Gradient Boosting Decision Trees with engineered features.
- **Results**:
  - `std_Cl`: R² = 0.72
  - `mean_Cl`: R² ~ 0.61
  - `mean_Cd`: R² = 0.44
  - `St_peak`: R² = 0.13

## How to Run

You can re-run any approach using `uv run`:

```bash
# Approach 1
uv run python src/training/train_tier1_improved.py --config experiments/tier1_improved/config.yaml

# Approach 2
uv run python src/training/train_mlp_derivative.py --config experiments/tier1_4b_mlp/config.yaml

# Approach 3
uv run python src/training/train_rom.py --config experiments/tier1_rom/config.yaml

# Approach 4
uv run python src/training/train_lightgbm.py --config experiments/tier1_lightgbm/config.yaml
```

## Recommendations

1.  **Approach 1 (Improved MLP)** is currently the best performing model. It should be the baseline for further work.
2.  **Approach 2 & 3** require tuning. The poor performance is likely due to unoptimized hyperparameters (e.g., regularization strength, kernel parameters) rather than fundamental flaws.
3.  **Preprocessing**: I used the existing [data/processed/summary.parquet](file:///Users/mingz/Projects/tier1_package/data/processed/summary.parquet). If you locate the raw data, you can regenerate it using:
    ```bash
    uv run python -m src.data.preprocessing --raw-root /path/to/raw/data --output data/processed/summary.parquet
    ```
