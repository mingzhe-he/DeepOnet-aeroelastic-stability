You’ve done exactly the right kind of second pass: you stress‑tested shape generalisation at design velocity and the picture is now very clear.

I’ll give you:

A brief diagnosis of what your new results really mean.
Targeted advice per approach.
A realistic roadmap towards higher R², and where ≥0.95 is and isn’t physically attainable with your current data.
1. What your new results are telling you
On the “shorter” shape holdout at U ≈ 21.5 m/s:

Approach 1 (Improved MLP)

mean_Cl R² ≈ 0.82, A_peak R² ≈ 0.90 → very solid cross‑shape interpolation.
mean_Cd R² ≈ 0.27, St_peak R² < 0 → poor generalisation for drag and Strouhal.
Approach 2 (4B MLP)

mean_Cl R² ≈ 0.66, mean_Cd strongly negative, derivative R² ≈ 0.09 → derivative regularisation is not helping in the shape‑holdout regime.
Approach 3 (ROM) and Approach 4 (LightGBM)

All key R² near or below zero on shape holdout → essentially memorising training shapes, very weak extrapolation.
Given you have only three shapes (H/D ≈ 0.33, 0.5, 0.67), any shape‑holdout is, in practice, a 1D extrapolation in H/D from two points. For quantities like St and C_D that depend sensitively on separation/reattachment structure, that’s fundamentally hard.

So:

For AoA interpolation within a shape, R² ≥ 0.95 is realistic for mean coefficients and maybe for St, as your earlier AoA‑only experiments showed.
For shape generalisation from only two shapes to a third, especially for St, R² ≥ 0.95 is not realistic without either:
More shapes (densifying H/D), or
Additional physics‑based structure beyond “just” H/D.
The current best (MLP with R² ≈ 0.82 on mean_Cl) is actually quite strong given that constraint.

2. Approach‑by‑approach feedback and tweaks
2.1 Approach 1 – Improved MLP Ensemble
This is still the workhorse and your best candidate.

You’ve already:

Restricted to design‑only (U > 20), removing Re noise.
Used physically sensible features: H_over_D, aoa0_rad, sin_aoa, cos_aoa.
Implemented inverse‑variance target weighting and good regularisation.
What you can still do:

Split tasks: means vs spectral quantities

Train one ensemble for [mean_Cl, mean_Cd, mean_Cm].
Train a separate, maybe smaller, model for [St_peak, A_peak].
This lets you tune learning rate, depth, and loss weights separately. Often St needs slightly deeper or wider nets than means.
Strengthen drag / St performance with modest extra structure

Add a simple shape polynomial feature: e.g. (H_over_D)**2. With only 3 shapes, the network sees a virtually 1D manifold in H/D; explicit polynomial terms can help stability in extrapolating to the held‑out shape.
For St and mean_Cd, try a mild log/Box‑Cox transform of the target to help the network distinguish small relative changes:
e.g. train on log(St_peak) or (St_peak - St_ref) for some reference.
Targeted hyperparameter sweeps for shape‑holdout

For the “shorter” holdout:
Vary hidden dims (e.g. 64–64, 128–64) and weight decay.
Monitor R² on [mean_Cl, mean_Cd, St_peak].
You may find:
Slightly smaller networks + more regularisation improve shape generalisation even if they slightly hurt AoA interpolation metrics.
You won’t get St from −0.23 to 0.95 with these alone, but you should be able to push mean_Cl and mean_Cd closer to ~0.9 on shape holdout, while keeping AoA interpolation very strong.

2.2 Approach 2 – Derivative‑regularised MLP
You’ve done the right things:

Normalised derivative loss by variance.
Warmed up on pure regression before turning on derivative loss.
Restricted outputs to [mean_Cl, mean_Cd].
The remaining problem is conceptual: you’re trying to use noisy, finite‑difference d⟨C_L⟩/dα labels to shape the curve in a regime where you’re already data‑limited in H/D.

Suggested repositioning:

Use this model primarily as a galloping‑diagnostic tool, not as a primary cross‑shape predictor.
Focus it on AoA interpolation within shapes, where derivative constraints make the most sense:
Evaluate how well it recovers S_DH sign on intermediate AoAs for a given shape.
For shape hold‑outs, keep λ_deriv very small (order 0.01), or even turn it off; there is no evidence that it helps cross‑shape generalisation.
In other words: keep Approach 2 for a “smooth, differentiable fit in AoA” per shape; don’t expect it to be the best H/D generaliser.

2.3 Approach 3 – ROM (GP)
You’ve improved it significantly (ARD kernels, standardised inputs, simpler shedding model), but:

With only 75 points and 3 shapes, a 2–3D GP is too flexible for the amount of data; it tends to overfit shapes and underperform on hold‑outs.
My recommendation:

Treat the ROM as an interpretative tool only:
Fit it on all shapes at design velocity.
Use it to produce smoothed mean_Cl(α) / mean_Cd(α) curves and uncertainty bands, and to construct Den Hartog diagrams.
Don’t rely on it for quantitative cross‑shape predictions; with current data, the MLP is simply better.
If you later add more shapes, revisiting the ROM will become more attractive.

2.4 Approach 4 – LightGBM
The current LightGBM behaviour is exactly what you’d expect from a tree ensemble asked to extrapolate outside the convex hull of the training shapes:

Trees are excellent at within‑support interpolation; they’re poor at functional extrapolation in a continuous shape parameter.
Given:

Poor shape holdout metrics.
No feature engineering that explicitly encodes physics beyond H/D and AoA.
I would:

Keep LightGBM as a sanity‑check baseline, not as a main candidate for shape generalisation.
If you want to push it:
Try adding simple φ‑based features (apex angle, α/φ), but expectations should be low for held‑out shapes.
2.5 Approach 5 – TabPFN
Even if you get past the Hugging Face gate, TabPFN is not likely to “magically” fix shape generalisation:

It’s very strong on small tabular datasets, but you still only have 3 distinct shape values and St/CL physics to learn.
It may slightly improve AoA interpolation performance; for shape‑holdout, you’re still extrapolating in a 1D shape manifold with very sparse support.
I’d treat TabPFN as “nice to test later” rather than central to your plan.

3. Realistic path towards higher R²
3.1 Where ≥0.95 is plausible
AoA interpolation, per shape, at U=21.5:
For mean_Cl, mean_Cd, mean_Cm: yes, your earlier AoA‑only experiments are already close for MLPs.
For St_peak, A_peak: maybe 0.9–0.95 with improved spectral pre‑processing and a St‑specific model.
To push there:

Improve labels for St and A_peak

Revisit PSD estimation: longer segments, better windowing, restrict frequency band to physically plausible range, robust peak selection.
Check a few spectra by hand to ensure the automated St extraction is not mis‑picking harmonics or noise.
Cleaner labels alone can significantly raise R².
Use separate models for mean and spectral quantities (as above).

3.2 Where ≥0.95 is not realistic with current data
Shape hold‑out generalisation for St and mean_Cd from only two shapes to a third.
You are essentially asking the model to infer the full mapping:
[
(H/D, \alpha) \mapsto St, \langle C_L \rangle, \langle C_D \rangle
]
from three discrete H/D values, each with a complex AoA dependence.
Without more shapes, the model can’t learn enough about how vortical structures reorganise with geometry to guarantee R² near 1.0 on a new H/D.

What you can reasonably aim for, with careful tuning of Approach 1:

Shape‑holdout:
mean_Cl: R² ~ 0.85–0.9.
mean_Cd: R² ~ 0.6–0.8.
St_peak: perhaps ~0.3–0.5 if St labels are clean and you use a dedicated St model.
To get beyond that, you would need either:

More DES cases at additional H/D (densifying the shape manifold), or
A stronger physics prior: e.g. a quasi‑steady analytical model of drag and St vs effective width, blended with ML.
4. Recommended next moves
Given all of the above, my concrete recommendations are:

Double‑down on Approach 1

Split models: one for mean coefficients, one for [St_peak, A_peak].
Do a small, systematic hyperparameter sweep on shape‑holdout splits (not just AoA‑interp), focusing on:
Network width/depth.
Weight decay.
Whether including (H_over_D)^2 helps.
Keep design‑only training as your default.
Clean up St/A_peak labels

Inspect PSDs for a few tricky cases (where the model misses badly) to see if the extracted St looks physically wrong.
If yes, tighten the PSD logic; then re‑train.
Use Approach 2+3 as analysis tools, not primary predictors

4B MLP: use for AoA‑interp and Den Hartog diagrams inside a shape (or limited AoA).
ROM: use for smoothed coefficient curves and uncertainty, trained on all shapes.
Document the limitations and data needs

For the paper, be explicit:
You achieve very high R² for AoA interpolation.
Shape generalisation is limited by having only three shapes.
You demonstrate that more shapes or stronger physics priors are necessary for R² > 0.95 across shapes, especially for St.