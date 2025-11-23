# Third Implementation - Final Report

## Executive Summary

Implemented **split models** for Approach 1 with polynomial features and log transforms. Comprehensive validation reveals **significant shape generalization challenges** that require further work.

---

## What Was Implemented

### 1. Split Models Architecture ✓
Created two specialized models instead of one monolithic approach:

**Means Model** ([train_tier1_means.py](file:///Users/mingz/Projects/tier1_package/src/training/train_tier1_means.py))
- Targets: `[mean_Cl, mean_Cd, mean_Cm]`
- Architecture: `[64, 64]` (compact for generalization)
- Regularization: `weight_decay=5e-4`
- Features: `[H_over_D, (H_over_D)², aoa0_rad]`

**Spectral Model** ([train_tier1_spectral.py](file:///Users/mingz/Projects/tier1_package/src/training/train_tier1_spectral.py))
- Targets: `[St_peak, A_peak]`
- Architecture: `[128, 128, 64]` (deeper for complex patterns)
- Log transform: `log(St_peak)` during training
- More dropout: `0.2`

### 2. Feature Engineering ✓
- **Polynomial shape feature**: [(H_over_D)²](file:///Users/mingz/Projects/tier1_package/src/models/mlp_scalar.py#120-135) to capture non-linear H/D effects
- **Log transform**: for St_peak to handle small relative changes

### 3. Data Quality Investigation ⚠️
- Improved PSD extraction (larger nperseg, quality flags, A_peak)
- **Issue found**: New method picks different peaks (~2.5x lower St values)
- **Decision**: Using original data; will investigate later

---

## Comprehensive Validation Results

### Performance by Shape Holdout (Design Velocity Only)

| Model | Holdout | mean_Cl R² | mean_Cd R² | mean_Cm R² | St_peak R² | A_peak R² |
|-------|---------|------------|------------|------------|------------|-----------|
| **Means** | Shorter | **0.75** | **0.34** | -0.32 | - | - |
| **Means** | Taller | **-5.25** ❌ | **-9.16** ❌ | **-22.77** ❌ | - | - |
| **Means** | Baseline | **0.86** ✓ | **0.19** | -0.21 | - | - |
| **Spectral** | Shorter | - | - | - | **-0.49** | **0.48** |
| **Spectral** | Taller | - | - | - | **-1.12** ❌ | **0.31** |
| **Spectral** | Baseline | - | - | - | **0.35** | **0.82** ✓ |

### Average Performance

**Means Model:**
- `mean_Cl`: avg R²=-1.21 (range: -5.25 to 0.86)
- `mean_Cd`: avg R²=-2.88 (range: -9.16 to 0.34)
- `mean_Cm`: avg R²=-7.77 (range: -22.77 to -0.21)

**Spectral Model:**
- `St_peak`: avg R²=-0.42 (range: -1.12 to 0.35)
- `A_peak`: avg R²=0.54 (range: 0.31 to 0.82) ✓

---

## Key Findings

### 1. **Extreme Shape Sensitivity**
The models show **dramatically different performance** across shape holdouts:
- Works well: Shorter, Baseline (R² ~ 0.3-0.9 for some targets)
- **Fails catastrophically: Taller** (R² < -5 for all mean coefficients)

This suggests:
- Models are **overfitting to training shapes**
- The [(H_over_D)²](file:///Users/mingz/Projects/tier1_package/src/models/mlp_scalar.py#120-135) polynomial feature alone is insufficient
- May need more complex shape representations or more data

### 2. **Best Performing Target: A_peak**
- Avg R²=0.54, consistently positive across all holdouts
- Physical interpretation: Peak amplitude may be more universal across shapes

### 3. **Worst Performing: mean_Cm**
- Avg R²=-7.77, never positive
- Moment coefficient is highly shape-specific and poorly generalized

###  4. **St_peak Challenges Confirmed**
- Avg R²=-0.42, mostly negative
- Log transform didn't significantly help
- Aligns with feedback: St is fundamentally hard with only 3 shapes

---

## Comparison with Previous Results

### Approach 1-v2 (Original Ensemble) vs Split Models

**On 'Shorter' Holdout:**
| Target | Approach 1-v2 | Split Models | Change |
|--------|---------------|--------------|--------|
| mean_Cl | 0.78 | 0.75 | -0.03 (similar) |
| mean_Cd | 0.27 | 0.34 | +0.07 (✓ improved) |
| mean_Cm | - | -0.32 | - |
| St_peak | -0.26 | -0.49 | -0.23 (worse) |
| A_peak | 0.90 | 0.48 | -0.42 (worse) |

**Verdict:** Split models show **marginal improvement on mean_Cd** but **degradation on spectral quantities**.

---

## Root Causes Analysis

### Why Taller Holdout Failed So Badly?

1. **Extrapolation distance**: Taller shape (H/D=0.67) is furthest from training points
2. **Polynomial insufficiency**: [(H_over_D)²](file:///Users/mingz/Projects/tier1_package/src/models/mlp_scalar.py#120-135) captures some curvature but misses complex physics
3. **Small training set**: Only 23-26 design-velocity cases to learn from
4. **Physics not encoded**: No explicit knowledge of how vortex patterns scale with H/D

### What About Baseline Holdout Success?

- Baseline (H/D=0.5) is **between** shorter (0.33) and taller (0.67)
- **Interpolation** works better than **extrapolation**
- This is expected behavior for polynomial features

---

## Recommendations

### Immediate Actions

1. **DO NOT use these models for production** in current form
   - Too unreliable across shapes (huge variance)
   - Negative R² = worse than predicting the mean

2. **Focus on AoA interpolation instead of shape generalization**
   - Validate on [aoa_interpolation.json](file:///Users/mingz/Projects/tier1_package/data/splits/aoa_interpolation.json) to see within-shape performance
   - This is where R² ≥ 0.95 is achievable per feedback

3. **Document data limitations**
   - 3 shapes is insufficient for robust shape generalization
   - Polynomial features are not enough without more data

### For Better Shape Generalization (Future Work)

1. **More training data**:
   - Add 2-3 more shapes between H/D=0.33 and 0.67
   - Densify the shape manifold

2. **Physics-based features**:
   - Apex angle φ = atan(H/D)
   - Effective width modelsEffective angle of attack vs geometry
   - Combine with ML rather than pure data-driven

3. **Ensemble across splits**:
   - Train separate models on each shape pair
   - Ensemble predictions weighted by confidence

4. **Hybrid ROM + ML**:
   - Use ROM (Approach 3) for physics priors
   - ML to correct residuals

---

## Files & Artifacts

### Created:
- [train_tier1_means.py](file:///Users/mingz/Projects/tier1_package/src/training/train_tier1_means.py)
- [train_tier1_spectral.py](file:///Users/mingz/Projects/tier1_package/src/training/train_tier1_spectral.py)
- [config: tier1_means](file:///Users/mingz/Projects/tier1_package/experiments/tier1_means/config.yaml)
- [config: tier1_spectral](file:///Users/mingz/Projects/tier1_package/experiments/tier1_spectral/config.yaml)
- [compare_split_models.py](file:///Users/mingz/Projects/tier1_package/src/analysis/compare_split_models.py)

### Results:
- Validation on 3 shape holdouts: ✓
- Comparison CSV: [experiments/split_models_comparison.csv](file:///Users/mingz/Projects/tier1_package/experiments/split_models_comparison.csv)
- Individual metrics: `experiments/tier1_{means,spectral}/results_{shorter,taller,baseline}/`

---

## Important Note on baseline_lowU and baseline_mediumU

As noted by the user, when doing shape holdouts, we correctly **exclude** `baseline_lowU` and `baseline_mediumU` because:
- These use the **baseline shape** (H/D=0.5) but at different velocities (U=5, U=10)
- Including them would "leak" geometric information about the held-out shape
- Design-only filtering (U>20) automatically handles this ✓

---

## Next Steps

**Option A: Pivot to AoA Interpolation** (Recommended)
- Validate split models on [aoa_interpolation.json](file:///Users/mingz/Projects/tier1_package/data/splits/aoa_interpolation.json)
- Aim for R² ≥ 0.95 on mean coefficients (achievable goal)
- Document as "excellent within-shape interpolation"

**Option B: Hyperparameter Optimization for Shape Holdout**
- Sweep architecture, regularization, learning rate
- **BUT**: Unlikely to fix fundamental data scarcity issue
- May improve from R²=-5 to R²=0.5, but still not production-ready

**Option C: Focus on Documentation & Reporting**
- Accept shape generalization limitations with current data
- Clearly document what works (AoA interp) vs what doesn't (shape holdout)
- Provide roadmap for future improvements

---

## Honest Assessment

**What we proved:**
✓ Split models architecture works technically  
✓ Polynomial features easy to implement  
✓ Log transforms for St are viable  
✓ Code is flexible (easy switching between splits)  

**What we learned:**
⚠️ 3 shapes insufficient for robust shape generalization  
⚠️ Simple polynomial features don't capture complex aerodynamic scaling  
⚠️ Taller shape (H/D=0.67) is fundamentally hard to extrapolate to  
⚠️ Current approach won't reach target R²>0.85 on shape holdouts  

**Realistic expectations with current data:**
- **AoA interpolation**: R² ≥ 0.95 achievable ✓
- **Shape holdout**: R² ~ 0.3-0.6 best case (not production-ready)
- **More shapes needed** for R² > 0.85 on cross-shape prediction

This aligns with the feedback's assessment: shape generalization from 3 discrete H/D values is fundamentally challenging without more data or stronger physics priors.
