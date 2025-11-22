 High-level summary

  - I’ve kept your baseline Tier 1 scalar surrogate intact and added three new modelling pipelines around it, all driven by the physics (Strouhal, VIV vs galloping, Den Hartog) and by your data regime (~75 DES runs).
  - I also added a robust preprocessing CLI that corrects the force coefficients for both simulation velocity and triangle depth, and regenerates summary.parquet directly from the raw OpenFOAM folders.

  Below I’ll walk through: data/preprocessing, the four approaches, feature engineering choices, tuning, visualisation, and codebase review.

  ———

  1. Data, raw-folder structure, and preprocessing

  - Assumed raw DES root (from your description):
    .../aeroelasticity/data/{baseline,baseline_lowU,baseline_mediumU,shorter,higher}/<AoA>/postProcessing/cylinder/0/forceCoeffs.dat.
  - Axis convention is respected: x = along-wind, z = across-wind, y unused; shape “depth” is in z, width (chord) is 3 m in x.
  - The important physical complication: all forceCoeffs were non-dimensionalised with fixed reference values
    U_design = 21.5 m/s, lRef = 1.5 m, Aref = 1.5^2 m², regardless of actual U and depth H.

  New preprocessing utilities in src/data/preprocessing.py:

  - Header parsing:
      - parse_force_coeffs_header now extracts lRef, Aref, CofR and U_design (from magUInf/UInf) if present.
  - Velocity + depth rescaling of coefficients:
      - New function rescale_force_coefficients(df, metadata, shape_type, U_sim, design_U_default=21.5) implements
        [
        C^* = C_{\text{raw}} \frac{U_{\text{design}}^2 A_{\text{file}}}{U_{\text{sim}}^2 A_{\text{true}}}, \quad A_{\text{true}} = H^2
        ]
        with H taken from SHAPE_PARAMS (baseline 1.5 m, shorter 1.0 m, taller/higher 2.0 m).
      - This rescales C_L and C_D for correct simulation velocity and shape depth; C_M is additionally scaled by lRef_file/H to account for the moment reference length.
      - Baseline cases at 21.5 m/s and H=1.5 m remain unchanged; low/medium U and different depths are corrected.
  - Shape handling:
      - SHAPE_PARAMS supports both "taller" and "higher" as aliases for the H=2.0 m shape.
  - Case processing:
      - process_case now:
          - Loads forceCoeffs.dat.
          - Detects U_sim from folder name (baseline_lowU → 5 m/s, baseline_mediumU → 10 m/s, else 21.5).
          - Calls rescale_force_coefficients to fix CL, CD, CM.
          - Computes mean/std coefficients after a settling time (default 80 s).
          - Computes Re = U_sim·D/ν with D=3.0 m.
          - Extracts spectral features from the 4000 Hz C_L(t) via Welch PSD: St_peak, A_peak, Q, bandwidth, freq_peak.
          - Builds non-dimensional features: H_over_D, aoa_rad, sin_aoa, cos_aoa, U_ref, Re.
  - Case discovery and summary building:
      - discover_cases(base_path) scans your raw DES folder for all {shape_variant}/{AoA}/postProcessing/cylinder/0/forceCoeffs.dat.
      - build_summary_dataframe(base_path, settling_time, nu) loops over all cases, calls process_case, and returns a row per case with:
          - Geometry/flow: shape_type, shape_variant, aoa, aoa_rad, D, H, H_over_D, U_ref, Re.
          - Means/stds: mean_Cd, mean_Cl, mean_Cm, std_Cd, std_Cl, std_Cm.
          - Spectral: St_peak, A_peak, Q, bandwidth, freq_peak.
          - AoA features: sin_aoa, cos_aoa.
      - build_summary_parquet(base_path, output_path, ...) writes this to Parquet.

  CLI entrypoint (preprocessing):

  uv run python -m src.data.preprocessing \
    --raw-root /Volumes/MHSSD/Projects/aeroelasticity/data \
    --output data/processed/summary.parquet

  This is the fastest way to regenerate a physically consistent dataset from your raw OpenFOAM folders.

  ———

  2. Theoretical background and modelling choices

  - Strouhal number: For vortex shedding,
    [
    St = \frac{f_{\text{shed}} D}{U}
    ]
    For your triangular prisms at high Re, St primarily depends on geometry (H/D) and AoA, with only weak Re-dependence once separation is corner-fixed.
  - Galloping vs VIV:
      - VIV: narrow-band resonance near the lock-in region; St is the main descriptor.
      - Galloping: quasi-steady instability governed by the slope of the mean lift curve; Den Hartog criterion:
        [
        S_{DH} = \frac{\partial \langle C_L \rangle}{\partial \alpha} + \langle C_D \rangle < 0 \Rightarrow \text{galloping}
        ]
  - Small-data regime:
      - With ~75 DES cases, deep architectures (DeepONets, Neural Operators) are strongly overparameterised. The recommended strategy is:
          - Reduce inputs via Buckingham π (use H/D and α as primary variables).
          - Use smooth, data-efficient surrogates (MLPs, GPs, tree ensembles).
          - Exploit differentiability (MLPs, GPs) or analytic structure (trees, ROM) for derivatives instead of learning time histories directly.

  The four approaches below reflect this:

  1. Improved MLP ensemble (same philosophy as your baseline, but better features and loss).
  2. 4B multi-task MLP with derivative regularisation (explicitly targeting galloping derivatives).
  3. Physics-informed ROM (Gaussian-process galloping model + regime-aware St surrogate + Landau reconstructor).
  4. LightGBM small-data engineering surrogate (tree-based, robust and interpretable).

  ———

  3. Approach 1 – Improved Tier 1 ensemble MLP

  File: src/training/train_tier1_improved.py
  Config: experiments/tier1_improved/config.yaml
  Run:

  uv run python src/training/train_tier1_improved.py \
    --config experiments/tier1_improved/config.yaml

  Core ideas:

  - Features (experiments/tier1_improved/config.yaml):
      - D, H, H_over_D, Re: geometry + Re.
      - aoa0_rad, sin_aoa0, cos_aoa0: AoA shifted around 90° (α0 = α−90°, radians) with periodic encoding to respect symmetry.
      - add_aoa_shift_features inside the script constructs these from aoa.
  - Targets: St_peak, A_peak, Q, mean_Cd, mean_Cl, mean_Cm (same as baseline).
  - Model: Ensemble of 5 MLPs (MLPScalarSurrogate) with slightly larger network:
      - Hidden dims [128, 128, 64], ReLU, BatchNorm, dropout.
      - Device automatically chosen via get_device() (cuda on NVIDIA, mps on Apple Silicon).
  - Weighted multi-task loss:
      - Target-wise weights computed from training-set variances:
          - compute_target_weights(y_train, method="inverse_var") → 1 / Var(y_j) scaled to mean 1.
      - Training loss:
        [
        \mathcal{L} = \frac{1}{N} \sum_i \sum_j w_j (y_{ij}^{\text{pred}} - y_{ij})^2
        ]
        so low-variance targets like St_peak and Q aren’t drowned by larger-magnitude mean coefficients.
      - Uses AdamW with configurable weight_decay (default 1e-4) and ReduceLROnPlateau LR scheduling.
  - Outputs & visualisation (under experiments/tier1_improved/results/):
      - metrics_improved.json: metrics for train/val/test and Ridge baseline.
      - ensemble_improved_metadata.pt: normalisers and model metadata.
      - Parity plots: predictions_vs_actual_mlp_improved.png (MLP), predictions_vs_actual_ridge_improved.png.
      - Residuals: residuals_mlp_improved.png.
      - Aerodynamic curves: mean_coeffs_vs_aoa_improved.png, strouhal_vs_params_improved.png.
      - Den Hartog (from CFD data): den_hartog_diagram_improved.png.

  This approach is a direct, more physics-aware upgrade of your existing MLP ensemble.

  ———

  4. Approach 2 – Section 4B multi-task MLP with derivative regularisation

  File: src/training/train_mlp_derivative.py
  Config: experiments/tier1_4b_mlp/config.yaml
  Run:

  uv run python src/training/train_mlp_derivative.py \
    --config experiments/tier1_4b_mlp/config.yaml

  Physics-guided design (exactly your Recommendation 0, Section 4B):

  - Features (config feature_cols):
      - H_over_D: aspect ratio (primary geometric π-group).
      - Re: optional, here kept for completeness (you can omit if you prefer strict Re-independence at high Re).
      - aoa0_rad: α0 = α−90° (radians), centering AoA around a symmetric configuration.
      - sin_aoa0, cos_aoa0: periodic encoding of α0.
  - Outputs:
      - St_peak, mean_Cd, mean_Cl, mean_Cm (multi-task regression).
  - Derivative target:
      - add_cl_derivative_column groups by shape_variant and uses np.gradient w.r.t. aoa_rad to build a finite-difference estimate dmean_Cl_daoa for each case:
        [
        \frac{\partial \langle C_L \rangle}{\partial \alpha} \approx \frac{\Delta \langle C_L \rangle}{\Delta \alpha}
        ]
  - Training:
      - Features are normalised (FeatureNormalizer); targets and derivative remain in physical units.
      - For each batch:
          - Compute standard MSE loss on targets.
          - Use autograd on the MLP to obtain ∂⟨C_L⟩/∂α0 w.r.t. the normalised AoA feature; convert to physical units using chain rule:
            [
            \frac{\partial \langle C_L \rangle}{\partial \alpha} =
            \frac{1}{\sigma_{\alpha_0}} \frac{\partial \langle C_L \rangle}{\partial \alpha_{0,\text{norm}}}.
            ]
          - Add derivative loss:
            [
            \mathcal{L}{\text{deriv}} = \lambda{\text{deriv}} , | (\partial \langle C_L \rangle/\partial \alpha)_{\text{pred}} -
            (\partial \langle C_L \rangle/\partial \alpha){\text{FD}} |_2^2
            ]
            (lambda_deriv configurable; default 0.2).
          - Optional curvature penalty on second derivative is supported via lambda_curvature (default 0).
  - Metrics:
      - Standard regression metrics for outputs on train/val/test.
      - Additional derivative metrics on test:
          - r2_dmean_Cl_daoa, rmse_dmean_Cl_daoa, mae_dmean_Cl_daoa.
  - Visualisation (experiments/tier1_4b_mlp/results/):
      - metrics_4b_mlp.json with per-target and derivative metrics.
      - Parity: predictions_vs_actual_4b_mlp.png.
      - Mean coefficient curves: mean_coeffs_vs_aoa_4b_mlp.png (true vs MLP).
      - Strouhal: strouhal_vs_params_4b_mlp.png.
      - Den Hartog (from surrogate): den_hartog_diagram_4b_mlp.png computed on the predicted mean_Cl/mean_Cd curves.

  This gives you a smooth, differentiable surrogate for ⟨C_L⟩(α) and ⟨C_D⟩(α), with direct control of the derivative used in the Den Hartog criterion.

  ———

  5. Approach 3 – Physics-informed ROM

  Core modules:

  - src/rom/galloping_gp.py – GP ROM for mean coefficients and Den Hartog.
  - src/rom/shedding_model.py – regime-aware Strouhal surrogate.
  - src/rom/landau.py – Landau-style time-history reconstruction.

  Training script: src/training/train_rom.py
  Config: experiments/tier1_rom/config.yaml
  Run:

  uv run python src/training/train_rom.py \
    --config experiments/tier1_rom/config.yaml

  Galloping GP ROM (Phase 2)

  - Input: X = [H_over_D, aoa_rad].
  - Outputs: two independent GPs
      - mean_Cl(X), mean_Cd(X) using GaussianProcessRegressor with smooth RBF-based kernel and small noise.
  - Class GallopingGPROM:
      - .from_dataframe(train_df, feature_cols=["H_over_D","aoa_rad"], cl_col="mean_Cl", cd_col="mean_Cd")
      - .predict_mean(X) → (mean_Cl, mean_Cd)
      - .estimate_derivative_and_den_hartog(X, delta_deg):
          - Uses finite differences on the GP surrogate:
            [
            \frac{\partial \langle C_L \rangle}{\partial \alpha} \approx \frac{\langle C_L \rangle(\alpha + \delta) - \langle C_L \rangle(\alpha - \delta)}{2\delta}
            ]
          - Computes predicted S_DH = dCl/dα + Cd.

  Shedding ROM (Phase 3)

  - SheddingRegimeModel.from_dataframe(train_df, feature_cols=["H_over_D","aoa_rad"], target_col="St_peak", n_regimes=2):
      - K-means in (H/D, α, St) to label regimes.
      - RandomForestClassifier: (H/D, α) → regime.
      - Per-regime regressor:
          - GP if enough points,
          - Ridge if the regime is sparse.
  - .predict(X): regime classification + regime-specific regression to predict St_peak.

  Landau reconstruction (Phase 4)

  - reconstruct_cl_time_series(t, mean_cl, A_rms, St, U, D, phase=None):
    [
    C_L(t) = \bar{C}L + A{\text{rms}} \cos\left(2\pi St \frac{U}{D} t + \phi\right)
    ]
  - For A_rms you can use std_Cl from the summary or a GP surrogate if desired.
  - Helper make_time_array(duration, fs) to generate time grids.

  ROM training and outputs

  - train_rom.py:
      - Uses AoA-interpolation split for training/testing.
      - Fits GallopingGPROM and SheddingRegimeModel on train subset.
      - Evaluates on test subset:
          - Metrics for [mean_Cl, mean_Cd].
          - Metrics for St_peak (ROM vs CFD).
      - Saves models:
          - galloping_gprom.pkl, shedding_model.pkl.
      - Saves metrics: metrics_rom.json.
      - Visualisation (experiments/tier1_rom/results/):
          - mean_coeffs_vs_aoa_rom.png: GP-predicted vs true curves.
          - strouhal_vs_params_rom.png: ROM Strouhal vs parameters.
          - den_hartog_diagram_rom.png: S_DH from GP ROM across shapes and AoA.

  This approach directly matches the “physics-informed ROM” proposal and gives you uncertainty-aware, smooth surrogates for both mean coefficients and Strouhal regimes.

  ———

  6. Approach 4 – LightGBM surrogate

  File: src/training/train_lightgbm.py
  Config: experiments/tier1_lightgbm/config.yaml
  Run:

  uv run python src/training/train_lightgbm.py \
    --config experiments/tier1_lightgbm/config.yaml

  Design:

  - Features (feature_cols):
      - H_over_D, Re, U_ref – geometry & flow.
      - aoa_rad, sin_aoa, cos_aoa, aoa_deg2 (α²), sin_2aoa (sin 2α) – AoA and harmonics.
      - shape_type – categorical (shorter/baseline/taller or higher).
  - Targets:
      - St_peak, mean_Cl, mean_Cd, mean_Cm, std_Cl (std_Cl ≈ RMS of lift fluctuations).
  - Preprocessing:
      - add_engineered_features adds AoA trigonometric/polynomial features.
      - shape_type is cast to category and passed to LightGBM as a categorical feature.
  - Training:
      - K-fold CV (n_folds default 5) via KFold with shuffling.
      - For each fold:
          - Train one LGBMRegressor per target.
          - Compute predictions on the fold’s test subset.
          - Accumulate out-of-fold predictions into y_pred_oof.
      - Overall out-of-fold metrics: compute_metrics(y, y_pred_oof, target_names) summarised as “OOF (all folds)”.
      - Final models: retrained on all cases for each target and saved in lightgbm_models.pkl.
  - Outputs (experiments/tier1_lightgbm/results/):
      - metrics_lightgbm.json: per-fold and overall CV metrics.
      - lightgbm_models.pkl: final models and per-fold models per target.
      - predictions_vs_actual_lightgbm.png: parity plot using OOF predictions.

  This is the most “production-engineering” tool: robust in small data, fast at prediction, and amenable to SHAP analysis if you decide to add it later.

  ———

  7. Feature engineering and galloping-specific considerations

  Across all approaches:

  - Geometry: we work primarily with H_over_D (and occasionally D, H separately) as the dominant shape parameter; this follows Buckingham π and your recommendations.
  - AoA encoding:
      - Baseline/Improved MLP: sin(α), cos(α) plus α0 = α−90° and sin(α0), cos(α0) for symmetry-aware inputs.
      - 4B MLP: uses α0 explicitly as the differentiation variable.
      - ROM/LightGBM: AoA in radians, plus trigonometric and polynomial terms for tree models.
  - Reynolds number:
      - Included as Re for completeness; at your high-Re DES conditions its influence is weak, but it keeps the surrogate general.
  - Spectral features:
      - St_peak, A_peak, Q, bandwidth, freq_peak extracted via Welch PSD from the corrected 4000 Hz C_L(t); used directly as targets (Tier 1) and in regime classification (ROM).
  - Galloping stability:
      - 4B MLP explicitly learns the derivative of ⟨C_L⟩ w.r.t AoA via autograd.
      - ROM uses GP-based surrogates with finite-difference derivatives and builds a full Den Hartog diagram.
      - Improved MLP and LightGBM provide smooth ⟨C_L⟩(α) curves that can be post-processed using src.eval.metrics.estimate_cl_derivative if needed.

  ———

  8. Fine-tuning and optimisation suggestions

  Per approach:

  - Improved MLP ensemble:
      - Tune hidden_dims, dropout, weight_decay, and loss_weighting (switch between "inverse_var" and "none") using validation metrics.
      - For very small data, you may reduce ensemble size n_models to ~3 to limit variance.
  - 4B MLP:
      - Key knobs: lambda_deriv (controls how strongly you enforce the derivative match) and lambda_curvature.
      - If derivative metrics show over-smoothing (underfitting), reduce lambda_deriv; if oscillatory, increase it or add a small curvature penalty.
  - ROM:
      - GP kernels: adjust length scales and noise levels in _default_kernel() and _st_kernel() if you see over/under-smoothing.
      - Number of regimes: n_regimes in experiments/tier1_rom/config.yaml (2 or 3) to reflect observed shedding regimes.
  - LightGBM:
      - Grid search over num_leaves, max_depth, learning_rate, n_estimators using the OOF metrics as objective.
      - You can enable GPU-acceleration in LightGBM if you have a CUDA GPU available by adding the appropriate device parameters; on Mac M4Max CPU-based training is typically fast enough for this dataset.

  ———

  9. Visualisation coverage

  For each approach you get:

  - Improved MLP: Parity, residuals, mean coefficients vs AoA, St vs Re/AoA, Den Hartog (true).
  - 4B MLP: Parity, mean coefficients vs AoA (true vs pred), St vs parameters (true vs pred), Den Hartog diagram computed from surrogate curves, plus explicit derivative metrics.
  - ROM: Mean coefficients vs AoA (GP vs CFD), St vs parameters (ROM vs CFD), ROM-based Den Hartog diagram.
  - LightGBM: Parity plot based on out-of-fold predictions.

  These give you complementary views of interpolation quality, regime capture, and galloping stability trends.

  ———

  10. Codebase review and suggested improvements

  Already implemented improvements:

  - Corrected force coefficient scaling for velocity and depth in src/data/preprocessing.py, aligning with your note that all cases were originally normalised with fixed (21.5 m/s, 1.5 m, 1.5² m²).
  - Added a scriptable preprocessing path (src.data.preprocessing main) so you no longer need to rely solely on notebooks for summary generation.
  - Refactored training into separate scripts per approach with YAML-configurable hyperparameters and clear uv run commands.
  - Ensured all PyTorch training uses get_device() with MPS/CUDA support for Apple Silicon and NVIDIA GPUs.

  Further possible improvements (optional, not implemented):

  - Factor out shared AoA feature utilities (add_aoa_shift_features, add_engineered_features) into a small helper module (e.g. src/data/angle_features.py) to reduce duplication.
  - Extend ROM to include uncertainty bands on S_DH (GP predictive variance + finite-difference propagation).
  - Add SHAP analysis for LightGBM to quantify the influence of H/D vs AoA vs Re on each target (requires adding shap dependency).
  - If you ever expand the dataset significantly (> 150–200 cases), revisit a modest PINN/Neural Operator approach for spectrum prediction; for now, the small-data regime is better served by the current surrogates.

  ———

  How to proceed

  1. Regenerate summary.parquet from your raw DES data using the new preprocessing CLI (important for corrected CL/CD/CM).
  2. Run each approach via the provided uv run commands and inspect metrics/plots in the corresponding experiments/*/results folders.
  3. For galloping-focused work, rely primarily on:
      - 4B MLP for fast differentiable evaluation and parameter sweeps.
      - ROM for uncertainty-aware Den Hartog diagrams and regime-aware Strouhal.
  4. If you’d like, I can next:
      - Add a small utility notebook demonstrating how to query each surrogate for a new (H/D, U, AoA) triplet and reconstruct C_L(t) via the Landau ROM.
      - Help design a comparison figure (or table) across the four approaches for a paper or thesis chapter.

## Follow up questions
 A few points:
  1. in your proposed 1. Data, raw-folder structure, and preprocessing section: Considering Re is based on a fixed length D, U_ref is proportional to Re in this case, is it still needed to have both U_ref and Re both in the
  inputs?
  2. Considering D is fixed at the moment and H_over_D is also in the input, do you still need D, H and H_over_D all as the input?
  3. For approach 3, I think the Re should also be an important input? Please for all approaches, from the scientific point of view, use ultrathink to think carefully and critically the most appropriate inputs that should
  be included and also use your extensive experience in this field to think about the feature engineering for each of the approaches
  4. Please make sure you consider appropriate data normalisation or standardisation for better learning outcome
  5. Please make sure you have metrics/visualisation to compare the performance between different approaches. I will need to show in the paper the fine tuning strategy and experiments for each approach and in the end the
  performance between different approaches for my final reommendation and conclusions

  Based on the above and as a senior professor with all the necessary skills and knowledge in the relevant fields, please use your full reasoning capability to provide the most scientifically robust implementation plan

## Answer to the above questions
• I’ll answer your five points explicitly and then lay out a unified, scientifically grounded implementation plan for all four approaches.

  ———

  ### 1–3. What should be in the inputs?

  I’ll be strict here and separate “physics that truly matters” from “numerical conveniences”.

  #### 1. U_ref vs Re when D is fixed

  With your current dataset:

  - D is fixed (3 m).
  - ν is fixed (air at ~20°C).
  - Re = U_ref·D/ν is therefore exactly proportional to U_ref.

  From a similarity point of view:

  - The physically relevant dimensionless parameter is Re (for transitions, separation, etc.).
  - U_ref is only needed to convert back to dimensional frequencies (f = St·U_ref/D) or to reconstruct C_L(t).

  So for learning:

  - Neural models / GPs / trees should use Re (or log Re) as the input, not U_ref.
  - U_ref should be treated as metadata for reconstruction and post-processing, not as a separate feature.

  Implementation consequence:

  - For all four approaches, drop U_ref as a feature and keep Re only.
  - Keep U_ref in the summary table for time-series reconstruction and Strouhal frequency conversion, not for learning.

  #### 2. D, H, H_over_D when D is fixed

  For your 2D isosceles triangles:

  - Base length D is fixed (3 m).
  - Depth H varies; the shape is uniquely characterised by H/D (or equivalently apex angle φ).

  Thus:

  - D adds no information for the current dataset.
  - H and H/D are monotonically related; including both is redundant.

  From a fluid-dynamics standpoint, the relevant geometric π-group is:

  - Aspect ratio: ( AR = H/D ), or equivalently the apex angle ( \phi = \arctan(2H/D) ).

  So for your current dataset:

  - Use H_over_D as the sole geometric scalar feature.
  - Optionally, for future-proofing, you can keep D in the table but not in the ML/R0M feature vector unless you start varying base chord.

  Implementation consequence:

  - For all learning approaches, geometry input = H_over_D.
  - shape_type can be used as a categorical if you want to allow for minor differences beyond H/D (e.g. if future meshes or corner radii differ), but physically it is redundant given your idealised isosceles geometry.

  #### 3. Should Re be in the ROM (and other approaches)?

  Physics:

  - Your Re range (based on U = 5, 10, 21.5 m/s, D = 3 m, ν ≈ 1.5×10⁻⁵ m²/s) is about:
      - Re ≈ 1.0×10⁶, 2.0×10⁶, 4.3×10⁶.
  - For sharp-edged bluff bodies at such high Re, both mean coefficients and St are weakly Re-dependent: separation points are corner-fixed, and coefficients/St are approximately asymptotic.
  - However, you intentionally varied Re, and small trends with Re can still be present in DES (e.g. turbulence model sensitivity, minor changes in wake structure).

  So:

  - Scientifically: the mapping is ( (H/D, \alpha, Re) \mapsto {St, \langle C_L\rangle, \langle C_D\rangle, \langle C_M\rangle} ), but with very long correlation length in Re.
  - For current data, including Re should not dramatically change predictions. It should mostly act as a check that the model can learn that dependence is weak above Re ~10⁶.

  For the ROM and other approaches:

  - Yes, include Re, but:
      - Scale it appropriately (e.g. use log10(Re/Re₀) so it lives in O(1)).
      - Use strong smoothness in the Re direction (e.g. GP kernels with large length scale in Re).
  - Also run an ablation:
      - Train models with and without Re.
      - If performance is unchanged, you have quantitative evidence that Re can be safely dropped in the high-Re asymptotic regime—good for a paper.

  Implementation consequence:

  - Final input set for all approaches (core):
      - ( x = [H/D,\ \alpha,\ Re^*] ) in some encoding.
  - For each approach, we specialise this (below).

  ———

  ### 4. Final feature sets and normalisation per approach

  I’ll define a core feature encoding, then tailor each approach on top of this.

  #### Core continuous features (for all approaches)

  - Geometry:
      - AR = H_over_D (dimensionless, O(1)).
  - Reynolds:
      - Re_star = log10(Re / 1e6) (shifts Re around 0, magnitude ~0–0.6).
  - Angle of attack:
      - Use both a scalar and trigonometric encoding:
          - aoa_rad = α (deg) * π/180.
          - aoa0_rad = (α − 90°) * π/180 (centered about flat-base orientation).
          - sin_aoa, cos_aoa.
          - (for some approaches) sin_aoa0, cos_aoa0.

  Why this is robust:

  - Trig encoding handles periodicity and avoids discontinuity at ±180°.
  - Centering at 90° (α0) aligns with geometry symmetry and simplifies derivative interpretation for galloping.
  - Re_star log-scaling keeps the Re dimension numerically moderate and emphasises relative variation.

  #### Normalisation strategy

  - MLPs (Approaches 1 & 2):
      - Standardise all continuous input features with mean 0, std 1 (fit on train only).
      - Standardise outputs for pure regression (Approach 1); keep outputs in physical units when they appear in derivative losses (Approach 2 already does that).
  - ROM (GPs, Approach 3):
      - Inputs:
          - Use AR, aoa_rad, Re_star (all O(1)), no further scaling strictly necessary.
      - Outputs:
          - Use normalize_y=True for GPs; this subtracts mean and scales automatically.
  - LightGBM (Approach 4):
      - Tree-based models do not need scaling; but using AR, aoa_rad, Re_star already ensures sensible ranges.
      - Categorical shape_type if used is handled by LightGBM as a category.

  ———

  ### 5. Approach-specific feature sets and rationale

  #### Approach 1 – Improved MLP ensemble (scalar surrogate)

  Inputs:

  - Continuous:
      - H_over_D
      - Re_star
      - aoa0_rad (α−90° in radians)
      - sin_aoa, cos_aoa
  - No D, H, or U_ref as features.

  Targets:

  - St_peak, A_peak, Q, mean_Cd, mean_Cl, mean_Cm.

  Why:

  - This matches the reduced physics: geometry via AR, AoA via α0 and trigs, Re as “small correction” factor.
  - Redundancy between aoa0_rad and sin_aoa/cos_aoa is acceptable here (the network will learn a stable representation), and it helps derivative-related interpretability later.

  Normalisation:

  - Standardise inputs and outputs (already done via FeatureNormalizer).

  Feature ablations to run (for the paper):

  - F1: [H_over_D, sin_aoa, cos_aoa] (no Re, no α0).
  - F2: [H_over_D, Re_star, sin_aoa, cos_aoa].
  - F3: [H_over_D, Re_star, aoa0_rad, sin_aoa, cos_aoa] (final).
  - Report test R² and RMSE per target for F1–F3 to demonstrate the marginal effect of Re and α0.

  #### Approach 2 – 4B MLP with derivative regularisation

  Inputs (tighter, derivative-focused):

  - H_over_D
  - Re_star
  - aoa0_rad
  - Optionally sin_aoa0, cos_aoa0 (for symmetry), but the derivative is taken w.r.t aoa0_rad.

  Targets:

  - Main regression: St_peak, mean_Cd, mean_Cl, mean_Cm.
  - Derivative target: d⟨C_L⟩/dα (computed per shape_variant via finite differences on CFD means).

  Why:

  - The 4B goal is a clean map ( (H/D, Re, \alpha_0) \mapsto [St, \langle C_L\rangle, \langle C_D\rangle, \langle C_M\rangle] ) plus a smooth derivative of ⟨C_L⟩ with respect to α.
  - Using α0 directly as a scalar makes the autograd derivative well-defined and interpretable.
  - Including Re_star allows you to explicitly verify that ∂⟨C_L⟩/∂α is approximately Re-invariant — an interesting result to report.

  Normalisation:

  - Standardise inputs.
  - Targets remain in physical units so the derivative loss has meaningful scale.
  - Derivative loss λ_deriv is tuned so its contribution is comparable to the main regression loss.

  Feature ablations to run:

  - G1: [H_over_D, aoa0_rad] (no Re).
  - G2: [H_over_D, Re_star, aoa0_rad] (final).
  - G3 (optional): add sin_aoa0, cos_aoa0 and see if derivative metrics improve.

  Metrics specific to this approach:

  - Test R²/RMSE for mean_Cl and mean_Cd as usual.
  - Test R²/RMSE for derivative d⟨C_L⟩/dα.
  - Den Hartog classification:
      - Compare sign(S_DH) from surrogate vs CFD:
          - Accuracy, precision, recall for galloping-unstable points (S_DH < 0).

  #### Approach 3 – Physics-informed ROM

  Galloping GP ROM

  Inputs:

  - H_over_D
  - aoa_rad
  - Re_star (optional but recommended to test).

  Two variants:

  - ROM-A: GP inputs [H_over_D, aoa_rad] (assume Re-independent high-Re regime).
  - ROM-B: GP inputs [H_over_D, aoa_rad, Re_star].

  Outputs:

  - mean_Cl, mean_Cd.

  Derivatives & Den Hartog:

  - For a dense grid over α (and optionally Re), evaluate GP, compute d⟨C_L⟩/dα with finite differences, then S_DH.

  Shedding ROM

  Inputs:

  - Same core features as GP: [H_over_D, aoa_rad, Re_star].

  Outputs:

  - St_peak.

  Regime-awareness:

  - K-means clustering in [H_over_D, aoa_rad, St_peak] (and possibly Re_star).
  - RF classifier: (H_over_D, aoa_rad, Re_star) → regime.
  - Per-regime GP or Ridge for St.

  Landau reconstruction:

  - Uses:
      - mean_Cl from GP.
      - A_rms ≈ std_Cl from CFD (or another simple surrogate).
      - St from shedding ROM.
      - U_ref, D to reconstruct time scales.

  Normalisation:

  - Inputs: already dimensionless; optionally normalise Re_star to unit variance if needed.
  - Outputs: normalize_y=True in scikit GPs.

  Feature ablations and metrics:

  - Compare ROM-A vs ROM-B on:
      - Test R²/RMSE for mean_Cl, mean_Cd.
      - Test classification of galloping via S_DH sign vs CFD.
  - For St_ROM:
      - Test R²/RMSE of St on AoA-interpolation test set, and potentially on a “shape-holdout” split.

  #### Approach 4 – LightGBM surrogate

  Inputs:

  - Mandatory:
      - H_over_D, Re_star, aoa_rad, sin_aoa, cos_aoa, aoa_deg2, sin_2aoa.
  - Optional:
      - shape_type as categorical (to allow for non-H/D-based quirks, but physically redundant given perfect isosceles geometry).

  Targets:

  - St_peak, mean_Cl, mean_Cd, mean_Cm, std_Cl.

  Justification:

  - Trees handle redundant features well; here, the richer feature set helps piecewise-constant functions approximate smooth curves.
  - Including std_Cl gives you an amplitude surrogate that you can plug into the Landau reconstructor.

  Normalisation:

  - None required beyond the physical scaling we already have; LightGBM copes with mixed scales well.

  Ablations:

  - T1: features without Re_star.
  - T2: add Re_star.
  - T3: add shape_type.
  - Compare CV metrics for each.

  ———

  ### 6. Normalisation and data-standardisation summary

  - Neural nets (Approaches 1 & 2):
      - Standardise inputs: subtract mean, divide by std (fit on train only).
      - For Approach 1, standardise targets as well.
      - For derivative-based Approach 2, keep targets and derivative in physical units; interpret the derivative in physical α via chain rule.
  - GP ROM:
      - Use dimensionless inputs (H_over_D, aoa_rad, Re_star) with moderate ranges.
      - Rely on normalize_y in GP to scale outputs.
  - LightGBM:
      - Use physically scaled features only (no z-score required).
      - Treat shape_type as categorical.

  ———

  ### 7. Cross-approach metrics, visualisation, and fine-tuning experiments

  To support your paper, you want:

  1. Within-approach fine-tuning experiments (hyperparameter and feature ablations).
  2. Cross-approach comparison on the same test set and targets.

  #### Common split and targets

  - Use the existing AoA-interpolation split (data/splits/aoa_interpolation.json) for:
      - Improved MLP (Approach 1).
      - 4B MLP (Approach 2).
      - ROM (Approach 3).
  - For LightGBM:
      - Keep K-fold CV for robustness.
      - Additionally, run a variant on the AoA-interpolation split to allow direct test-set comparison.

  Common comparison targets:

  - St_peak, mean_Cl, mean_Cd, mean_Cm.
  - Derived:
      - d⟨C_L⟩/dα on a fine AoA grid.
      - S_DH and galloping classification (stable vs unstable).

  #### Numerical metrics (for each approach)

  For regression targets (St_peak, mean_Cl, mean_Cd, mean_Cm):

  - Test set:
      - R² per target.
      - RMSE and MAE per target.
      - Relative error (MAPE-style) where magnitudes permit.
  - Summarise across targets:
      - R²_overall, RMSE_overall.

  For derivatives and galloping:

  - R² and RMSE for d⟨C_L⟩/dα compared to CFD-derived derivatives (per shape).
  - Den Hartog classification:
      - Label points on AoA grid as unstable if S_DH < 0 (CFD vs model).
      - Compute:
          - Accuracy, precision, recall, F1 for unstable class.
          - If desired, an AUC using S_DH as a score.

  #### Visualisation (for figures)

  For each approach:

  - Parity plots (already in place):
      - True vs predicted for each target with 1:1 line and R² annotated.
  - Mean coefficient curves:
      - C_D(α), C_L(α), C_M(α) for each shape, true vs predicted.
  - Strouhal plots:
      - St vs Re (log axis) and St vs AoA for each shape, true vs predicted.
  - Den Hartog diagrams:
      - S_DH vs AoA for each shape; overlay CFD (reference) with curves from each approach.

  Cross-approach summary figure:

  - A panel of bar charts:
      - For each target (St, C_L, C_D, C_M), show test-set R² for the four approaches side-by-side.
      - A separate panel for derivative R² and galloping classification F1.

  Fine-tuning documentation:

  - For each approach, plan 3–4 variant experiments (feature sets, λ_deriv, network width, etc.) and record:
      - Config (in YAML).
      - Resulting metrics.
      - A short table summarising performance vs hyperparameter choice.

  This becomes a “model selection” section in your paper.

  ———

  ### 8. Final, concise implementation plan

  Putting everything together, here’s the concrete plan:

  1. Preprocessing
      - Use src.data.preprocessing to rebuild summary.parquet from the raw OpenFOAM tree, ensuring:
          - CL, CD, CM are renormalised to true U and H.
          - Spectral features from corrected 4000 Hz C_L(t) are included.
          - Core features H_over_D, aoa, Re are present.
      - Extend summary with Re_star, aoa_rad, aoa0_rad, sin_aoa, cos_aoa, etc. as per the feature sets above.
  2. Approach 1 (Improved MLP)
      - Inputs: [H_over_D, Re_star, aoa0_rad, sin_aoa, cos_aoa].
      - Targets: [St_peak, A_peak, Q, mean_Cd, mean_Cl, mean_Cm].
      - Train ensembles with weighted MSE; run feature ablations F1–F3.
  3. Approach 2 (4B MLP with derivative)
      - Inputs: [H_over_D, Re_star, aoa0_rad] (+ optional sin_aoa0, cos_aoa0).
      - Targets: [St_peak, mean_Cd, mean_Cl, mean_Cm] + derivative d⟨C_L⟩/dα.
      - Train with λ_deriv tuned via validation; compute derivative metrics and Den Hartog diagrams.
  4. Approach 3 (ROM)
      - Galloping GP ROM:
          - Inputs: ROM-A [H_over_D, aoa_rad], ROM-B [H_over_D, aoa_rad, Re_star].
          - Outputs: [mean_Cl, mean_Cd].
          - Evaluate derivatives and S_DH; compare ROM-A vs ROM-B.
      - Shedding ROM:
          - Inputs: [H_over_D, aoa_rad, Re_star].
          - Outputs: St_peak.
          - Regime-aware prediction via clustering + RF + per-regime GP/Ridge.
      - Use Landau reconstructor for C_L(t) where needed.
  5. Approach 4 (LightGBM)
      - Inputs: [H_over_D, Re_star, aoa_rad, sin_aoa, cos_aoa, aoa_deg2, sin_2aoa] (+ optional shape_type).
      - Targets: [St_peak, mean_Cl, mean_Cd, mean_Cm, std_Cl].
      - Run K-fold CV; additionally train on AoA-interpolation split to compare directly to neural/ROM approaches.
  6. Evaluation and comparison
      - Use the same AoA-interpolation split for Approaches 1–3 (and LightGBM AoA variant).
      - Collect metrics in a unified format (JSON per experiment).
      - Implement a small post-processing script/notebook to:
          - Aggregate metrics across methods.
          - Generate bar charts and tables for R²/RMSE per target and per approach.
          - Plot cross-approach Den Hartog diagrams for each shape.

  If you’d like, I can next:

  - Refine the existing training scripts and configs to match exactly these feature sets (drop D, H, U_ref; add Re_star and AoA encodings).
  - Add a single “comparison” notebook that pulls results from all experiments and produces publication-ready tables and figures.