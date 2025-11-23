I’ll go approach by approach, but first I’ll address the core issues that cut across everything: feature choices, normalisation, data splits, and why St/A_peak/mean_Cl are hardest.

1. Cross-cutting diagnosis
1.1 What your current results say
From your walkthrough:

Improved MLP (AoA-interp split)

mean_Cd: R² ≈ 0.96
mean_Cm: R² ≈ 0.92
mean_Cl: R² ≈ 0.88
A_peak: R² ≈ 0.74
St_peak: R² ≈ 0.68
4B MLP with derivative

Some targets negative R².
ROM (GP + regime-aware St)

St_peak: R² ≈ 0.25
mean_*: R² ≈ 0.
LightGBM (K-fold CV)

std_Cl: R² ≈ 0.72
mean_Cl: R² ≈ 0.61
mean_Cd: R² ≈ 0.44
St_peak: R² ≈ 0.13
Interpretation:

On AoA interpolation within known shapes, your improved MLP is already fairly strong, especially for mean coefficients.
St and A_peak are harder (expected); ROM and LightGBM are underperforming (implementation/theory issues, not just “data too small”).
You haven’t yet assessed shape generalisation explicitly (using the shape_holdout_* splits) — this is critical given your new requirement.
1.2 Physics + data realities
Geometry: All shapes are isosceles triangles with fixed base D = 3 m and varying depth H. The relevant π-group is H/D (or apex angle φ). With only 3 shapes, the “shape manifold” is extremely sparse.
Reynolds number / U_ref:
Cases: 5, 10, 21.5 m/s → Re ≈ 1.0×10⁶, 2.0×10⁶, 4.3×10⁶ for D=3 m.
At these Re, coefficients and St are only weakly Re-dependent for sharp-edged triangles; most variability is from H/D and AoA.
Targets:
mean_Cd, mean_Cm are relatively smooth and monotonic in α for each shape.
mean_Cl is sign-changing and more nonlinear; derivative matters for galloping.
St_peak and A_peak involve:
Spectral peak detection.
Regime changes (different shedding modes).
So they are piecewise-smooth with jumps in α and H/D.
Implication: St and A_peak require either a regime-aware or piecewise model, and the labels themselves may be noisy (spectrum estimation). Mean coefficients can be handled by smooth regressors (MLP, GP, GBM) if the features are aligned with the physics.

1.3 Features: what should be in, what should be out
Given D is fixed and U_ref is not your primary concern anymore:

For learning:
Use H_over_D as the sole geometric scalar.
Do not include raw D or H; they add no information beyond H/D.
Include Re only if you want to explicitly model Re-trends; otherwise, you can drop the speed variation and focus on design U=21.5 m/s.
For AoA:
Use both a scalar and trigonometric encoding:
aoa_rad = α *π/180.
aoa0_rad = (α − 90°)* π/180 (centered).
sin_aoa, cos_aoa.
For derivative-focused models: aoa0_rad is the key.
For Re dependence:
If you keep it, use Re_star = log10(Re / 1e6) to keep scale reasonable.
In many of your design-use cases, it’s defensible to train on only U=21.5 m/s and drop Re entirely for maximal accuracy.
Your improved MLP script is consistent with this philosophy (H_over_D, Re_star, aoa0_rad, sin/cos α). The ROM and 4B MLP also follow it in spirit.

2. Specific implementation checks and issues
2.1 Improved MLP ensemble (train_tier1_improved.py)
Implementation looks sound:

Uses:
H_over_D, Re_star, aoa0_rad, sin_aoa, cos_aoa.
Standardisation of X and y via FeatureNormalizer.
Inverse-variance weights per target to balance loss.
Ensemble of MLPs with AdamW and ReduceLROnPlateau.
Metrics are computed in physical units (after de-normalisation) with scikit’s R², RMSE, MAE.
No obvious coding “bug” here; the performance you see is representative.

2.2 4B MLP with derivative (train_mlp_derivative.py)
Implementation is mostly correct, but there are a few subtleties:

Data:
target_derivative is computed by np.gradient(mean_Cl, wrt_col) grouped by shape_variant.
With 5° AoA spacing and only 25 angles per shape_variant, derivatives will be noisy, especially near St/CL regime changes.
Inputs and scaling:
Inputs are normalised; outputs (mean_Cl, etc.) and derivative remain in physical units.
Derivative scaling is handled correctly: physical derivative is obtained via scale_factor = 1/std_wrt applied to grad w.r.t. normalised AoA feature.
Training loop:
Correct use of torch.autograd.grad for the target output; central gradient is implicitly per-sample (summing outputs is OK).
Validation loss includes both regression and derivative term.
The main reason for poor performance is not a bug, but loss balance and label noise:

λ_deriv = 0.2 may be too strong given noisy finite-difference derivatives; the network can sacrifice accuracy on mean_Cl to fit noisy derivative labels.
The derivative MSE is unweighted; its scale relative to regression loss can easily dominate.
2.3 ROM (train_rom.py, galloping_gp.py, shedding_model.py)
Key points:

Galloping GP ROM:
Uses:
Features from config (rom_a or rom_b): e.g. [H_over_D, aoa_rad] or plus Re_star.
Kernel: C(1.0) * RBF(length_scale=1.0) + WhiteKernel(1e-5).
Fits separate GPs for mean_Cl and mean_Cd.
Shedding ROM:
KMeans clustering in [standardised X, scaled St].
RandomForestClassifier to map X → regime.
Per-regime GP or Ridge regressor for St.
Implementations are coherent, but:

The GP kernel is isotropic and uses a single length scale for all features; with very few samples and anisotropic behaviour in AoA vs H/D, optimisation can land in suboptimal parts of hyperparameter space → essentially near-constant predictions (R² ~ 0).
The regime model uses clustering in a very small dataset (75 points); labels will be unstable and model capacity is high for the available data.
In short: the ROM is correctly coded but over-ambitious relative to data and too sensitive to kernel/cluster choices.

2.4 LightGBM (train_lightgbm.py)
Uses:
H_over_D, Re_star, aoa_rad, sin_aoa, cos_aoa, aoa_deg2, sin_2aoa, optional shape_type.
5-fold random CV (ignores AoA/shape structure).
Implementation is straightforward; two caveats:
The split_path in config is ignored; you’re not evaluating on AoA-interpolation or shape-holdout splits, so comparisons to the MLP/ROM are not apples-to-apples.
LightGBM is trained separately per target with identical hyperparameters; St and mean_Cd might need different regularisation (depth, num_leaves) from mean_Cm.
No glaring bug; performance is constrained by data and feature/param tuning.

2.5 Preprocessing (src/data/preprocessing.py)
rescale_force_coefficients:
Uses the physically correct formula:
[
C_\text{true} = C_\text{file} \frac{U_\text{design}^2 A_\text{file}}{U_\text{sim}^2 A_\text{true}}
]
With A_true = H² (consistent with your note about front area 1.5² for baseline).
process_case:
Detects baseline, shorter, taller/higher.
Uses correct U_sim based on folder (21.5 vs 10 vs 5 m/s).
Computes Re from U_sim and D = 3 m.
Computes St_peak from PSD and freq_peak * D / U_sim.
This script is not used in your current runs (you used the existing summary.parquet), but it is conceptually consistent.
3. Scientific critique and improvement suggestions per approach
3.1 Approach 1 – Improved MLP Ensemble
You already have:

Good AoA interpolation performance, especially for mean coefficients.
Moderate performance for St and A_peak.
Limitations:

Noise and regime changes in St/A_peak: Single smooth MLP is trying to fit discontinuities associated with different shedding modes.
Mixed tasks: St/A_peak and mean coefficients have very different noise/complexity; a single multi-task model may compromise St to better fit means.
Shape generalisation: Current split is AoA-interpolation only; shape generalisation is untested, and with only 3 shapes, it’s very easy to overfit those shape points (especially if Re-variation muddles the picture).
Improvements:

a) Separate heads or models for spectra vs mean coefficients
Option 1: One MLP for [mean_Cd, mean_Cl, mean_Cm], another for [St_peak, A_peak].
Option 2: Multi-task MLP with two output “blocks” and different loss weights (e.g. higher priority to St and mean_Cl, lower to A_peak).
b) Remove low-U cases for the “design” surrogate
Train a high-accuracy “design” MLP using only U=21.5 m/s cases.
This simplifies the input to [H_over_D, aoa0_rad, sin_aoa, cos_aoa] and removes Re noise.
c) Regularise for shape generalisation
For shape-holdout experiments, add stronger L2 weight decay or reduce network width to prevent memorising the three shapes.
Consider adding a very small penalty on the second derivative w.r.t H_over_D (computed via autograd): enforce smooth dependence of outputs on H_over_D.
d) Use shape-holdout splits to tune
Run:
Split AoA-interp (current).
shape_holdout_shorter, shape_holdout_taller, shape_holdout_baseline.
Report R² for each target under each split; choose hyperparameters that maintain good AoA interpolation while not collapsing on shape-holdout.
3.2 Approach 2 – 4B MLP with derivative regularisation
This is the right idea for galloping but you’re in a fragile regime:

Finite-difference derivatives are noisy (25 AoAs per shape, 5° spacing).
λ_deriv = 0.2 can easily dominate the loss if derivative residuals are large.
The same network is asked to fit both mean_* and derivatives; with small N, this can destabilise training.
Improvements:

a) Normalise derivative target and re-weight properly
Compute variance of the derivative target on the training set.
Use a target-wise weight akin to Approach 1:
Normalise derivative residuals by their std, or
Set λ_deriv = λ₀ / Var(derivative) so that its contribution is comparable to the regression loss.
b) Two-stage training
Stage 1: Train purely on regression (no derivative term) until convergence.
Stage 2: Fine-tune with a small λ_deriv (e.g. 0.02–0.05), keeping learning rate lower, so the network uses derivative term to “polish” the shape of ⟨C_L⟩(α) without destroying its fit.
c) Restrict outputs and/or tasks
For this approach, focus on [mean_Cl, mean_Cd] (and optionally St) rather than all four targets; reducing the task dimension helps the network focus on the relevant derivatives.
d) Filter and smooth derivative labels
Consider smoothing mean_Cl(α) per shape with a low-order polynomial or spline before taking numerical derivatives.
This gives smoother d⟨C_L⟩/dα labels and reduces the pressure on the MLP to fit raw finite-difference noise.
Goal: you won’t beat the raw MLP on pure pointwise metrics, but you should get smoother, more physically plausible 〈C_L〉(α) curves and derivative fields, which is what you need for galloping assessment.

3.3 Approach 3 – Physics-informed ROM
Current problems:

GP ROM gives R² ≈ 0 for mean_*: the kernel is too generic and may be mis-tuned; with only 75 points, GP hyperparameter optimisation is delicate.
Shedding model R² ≈ 0.25 for St: regime structure is real, but your clustering/classification layers are too brittle given N.
Improvements:

a) Use ARD kernel and standardised inputs
Before feeding into GP, standardise [H_over_D, aoa_rad, Re_star] to zero mean, unit variance.
Use:
kernel = C(1.0, (1e-2, 1e2)) *RBF(length_scale=[1.0, 1.0, 1.0],
                                  length_scale_bounds=(1e-2, 1e2)) \
         + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1))
This lets GP learn different length scales in H/D vs AoA vs Re_*, which is essential.
b) Simplify: Re-independent ROM for design velocity
For your main use case, restrict ROM training to U=21.5 m/s and use inputs [H_over_D, aoa_rad].
Show that adding Re_star does not improve performance significantly; this supports the asymptotic Re-independence argument.
c) Simplify shedding model first
First, fit a single GP (or Ridge) for St vs [H_over_D, aoa_rad] without regimes; get a baseline St R².
Only then, if needed, introduce a simple regime split:
e.g. manually split AoA into 2–3 bands (e.g. attached vs separated regimes) and fit separate regressors per band.
KMeans in (X,y) with 75 points is too fragile; small shifts in cluster centres can radically change per-regime training data.
d) Use this ROM mainly for interpretative / uncertainty plots
Given the stronger performance of the MLP, the ROM may be best positioned as:
A smooth, uncertainty-aware surrogate for mean Cl/Cd curves (with credible bands).
A tool for Den Hartog diagram exploration rather than the primary pointwise predictor.
3.4 Approach 4 – LightGBM
LightGBM is conceptually appropriate for small, tabular data, but:

It is currently evaluated with random K-fold CV, which:
Mixes AoAs and shapes across folds.
Makes it hard to compare with AoA-interpolation and shape-holdout experiments.
The same hyperparameters are used for all targets, even though St and mean_Cd behave very differently.
Improvements:

a) Align splits with other approaches
Allow train_lightgbm.py to optionally use split_path:
If provided, train and evaluate on that split (AoA interpolation or shape_holdout).
Otherwise, fall back to KFold.
This lets you compare LightGBM directly to the MLP and ROM on identical splits.
b) Target-specific hyperparameters or at least per-group tuning
At minimum, tune a set of hyperparameters for:
Group 1: [mean_Cl, mean_Cd, mean_Cm].
Group 2: [St_peak, std_Cl].
Often St requires deeper trees or more estimators than mean coefficients.
c) Consider adding apex angle φ-based features
Beyond H_over_D, you can encode shape via:
φ = arctan(2H/D), and/or
alpha_rel = aoa / φ (normalised AoA).
Trees love such engineered ratios; this may sharpen St vs α transitions.
4. Shape generalisation and design-velocity-focused modelling
You now want:

Accurate interpolation in AoA and accurate prediction for new shapes, primarily at U=21.5 m/s.
Given the small number of shapes (3), you should:

Explicitly test shape generalisation
Use shape_holdout_shorter.json, shape_holdout_taller.json, shape_holdout_baseline.json for all four approaches.
Measure R² for [St_peak, mean_Cl, mean_Cd, mean_Cm] on each held-out shape.
Expect a drop relative to AoA interpolation; this gives a realistic sense of shape extrapolation ability.
Train “design-only” surrogates
Filter the dataset to U_ref = 21.5 m/s and build dedicated surrogates:
Improved MLP (Approach 1) with inputs [H_over_D, aoa0_rad, sin_aoa, cos_aoa].
4B MLP (Approach 2) with [H_over_D, aoa0_rad].
ROM (Approach 3) with [H_over_D, aoa_rad].
LightGBM (Approach 4) with [H_over_D, aoa_rad, sin_aoa, cos_aoa, ...].
This removes confounding from varying U and focuses all capacity on shape + AoA.
Use AoA-interpolation as an internal test, shape-holdout as the main scientific test
In your paper:
AoA-interp results show what’s achievable for “densely sampled shapes”.
Shape-holdout results show what’s achievable for “new shapes at design velocity”, which is your real engineering use case.
5. Concrete next-step plan (scientifically robust)
Data preparation

Keep existing summary.parquet for now.
Create a filtered design-velocity view (in code) where U_ref == 21.5 for all training scripts.
Ensure you have derived features in all training scripts:
H_over_D
aoa_rad, aoa0_rad, sin_aoa, cos_aoa
(optionally) Re_star if you keep low/medium U cases.
Re-run Approach 1 under two regimes

AoA-interp (current split) with full data (all U).
Shape-holdout (three splits) with U=21.5 only.
Record R²/RMSE for [St_peak, mean_Cl, mean_Cd, mean_Cm] and note differences.
Try:
Removing A_peak from the target list to see if St and mean coefficients improve.
Training separate MLP just for [St_peak, mean_Cl, mean_Cd, mean_Cm].
Re-tune Approach 2 (4B)

Start with design-only data, inputs [H_over_D, aoa0_rad], outputs [mean_Cl, mean_Cd] plus derivative.
Run a small λ_deriv sweep: e.g. 0.01, 0.02, 0.05, 0.1.
Optionally, pretrain without derivative loss then fine-tune.
Evaluate:
R² for mean_Cl/mean_Cd.
R² for derivative.
Sign of S_DH vs AoA, per shape.
Refine ROM hyperparameters

Standardise inputs and add ARD RBF kernel for the GPs.
Compare:
ROM-A: inputs [H_over_D, aoa_rad].
ROM-B: inputs [H_over_D, aoa_rad, Re_star].
Train and evaluate only at U=21.5 first; if Re adds no value, you can state that in the paper.
For St_ROM, start with a single GP before re-introducing any regime machinery.
Align LightGBM with the other approaches

Add an option in train_lightgbm.py to use split_path:
If present, train on train+val, evaluate on test for that split.
Run both AoA-interp and shape-holdout splits.
Tune basic hyperparameters (depth, num_leaves, learning_rate) to match the scale of your data.
Build a comparison notebook / script

Read all metrics_*.json from:
tier1_improved, tier1_4b_mlp, tier1_rom, tier1_lightgbm.
Aggregate:
R²/RMSE for St, mean_Cl, mean_Cd, mean_Cm on each split.
Derivative metrics and galloping classification where applicable.
Produce:
Tables for hyperparameter sweeps.
Bar plots / boxplots comparing approaches per target.
Representative Den Hartog diagrams per approach.
This will give you a scientifically solid basis to:

Justify feature choices (H/D, AoA, Re_star).
Show why the improved MLP is currently the best “workhorse” surrogate.
Position the 4B MLP and ROM as physically refined tools (derivatives, stability diagrams) rather than pure pointwise approximators.
Demonstrate the limits of shape generalisation given only three shapes, while still extracting maximal information from the data you have.
If you like, I can next propose the exact config changes and a small metrics-aggregation script to standardise the comparison across all approaches and splits.
