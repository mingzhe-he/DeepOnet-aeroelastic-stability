"""
Evaluation metrics for scalar surrogate models.
"""

import numpy as np
from typing import Dict, List
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_names: List[str]) -> Dict:
    """
    Compute regression metrics for each target.
    
    Args:
        y_true: True values (n_samples, n_targets)
        y_pred: Predicted values (n_samples, n_targets)
        target_names: List of target names
        
    Returns:
        Dictionary with metrics for each target
    """
    n_targets = y_true.shape[1]
    
    metrics = {}
    
    for i, name in enumerate(target_names):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]
        
        metrics[f'r2_{name}'] = r2_score(y_t, y_p)
        metrics[f'mse_{name}'] = mean_squared_error(y_t, y_p)
        metrics[f'rmse_{name}'] = np.sqrt(mean_squared_error(y_t, y_p))
        metrics[f'mae_{name}'] = mean_absolute_error(y_t, y_p)
        
        # Relative error (MAPE-like)
        mask = np.abs(y_t) > 1e-8  # Avoid division by very small values
        if mask.sum() > 0:
            rel_err = np.abs((y_t[mask] - y_p[mask]) / y_t[mask])
            metrics[f'rel_err_{name}'] = rel_err.mean()
        else:
            metrics[f'rel_err_{name}'] = np.nan
    
    # Overall metrics
    metrics['r2_overall'] = r2_score(y_true.ravel(), y_pred.ravel())
    metrics['mse_overall'] = mean_squared_error(y_true.ravel(), y_pred.ravel())
    metrics['mae_overall'] = mean_absolute_error(y_true.ravel(), y_pred.ravel())
    
    return metrics


def compute_den_hartog_coefficient(Cd: np.ndarray, dCl_dalpha: np.ndarray) -> np.ndarray:
    """
    Compute Den Hartog stability coefficient.
    
    S_DH = dCL/dα + CD
    
    Negative S_DH indicates galloping instability.
    
    Args:
        Cd: Drag coefficient
        dCl_dalpha: Derivative of lift coefficient w.r.t. angle of attack
        
    Returns:
        Den Hartog coefficient
    """
    return dCl_dalpha + Cd


def estimate_cl_derivative(
    aoa: np.ndarray,
    cl: np.ndarray,
    method: str = 'finite_diff'
) -> np.ndarray:
    """
    Estimate dCL/dα from discrete samples.
    
    Args:
        aoa: Angle of attack (radians)
        cl: Lift coefficient
        method: 'finite_diff' for finite differences, 'polyfit' for polynomial fit
        
    Returns:
        Estimated derivative dCL/dα
    """
    if method == 'finite_diff':
        # Central differences where possible
        dCl_dalpha = np.gradient(cl, aoa)
        return dCl_dalpha
    
    elif method == 'polyfit':
        # Fit polynomial and take derivative
        poly = np.polyfit(aoa, cl, deg=2)
        # Derivative of polynomial
        poly_deriv = np.polyder(poly)
        dCl_dalpha = np.polyval(poly_deriv, aoa)
        return dCl_dalpha
    
    else:
        raise ValueError(f"Unknown method: {method}")


def print_metrics_summary(metrics: Dict, prefix: str = "") -> None:
    """
    Print formatted metrics summary.
    
    Args:
        metrics: Metrics dictionary
        prefix: Prefix for output (e.g., "Train", "Test")
    """
    print(f"\n{'='*60}")
    print(f"{prefix} Metrics")
    print(f"{'='*60}")
    
    # Overall metrics
    if 'r2_overall' in metrics:
        print(f"\nOverall:")
        print(f"  R² = {metrics['r2_overall']:.4f}")
        print(f"  MSE = {metrics['mse_overall']:.6f}")
        print(f"  MAE = {metrics['mae_overall']:.6f}")
    
    # Per-target metrics
    target_names = []
    for key in metrics:
        if key.startswith('r2_') and not key.endswith('_overall'):
            target_name = key[3:]  # Remove 'r2_' prefix
            target_names.append(target_name)
    
    if target_names:
        print(f"\nPer-target:")
        for name in target_names:
            print(f"\n  {name}:")
            print(f"    R² = {metrics[f'r2_{name}']:.4f}")
            print(f"    RMSE = {metrics[f'rmse_{name}']:.6f}")
            print(f"    MAE = {metrics[f'mae_{name}']:.6f}")
            if f'rel_err_{name}' in metrics and not np.isnan(metrics[f'rel_err_{name}']):
                print(f"    Rel Err = {metrics[f'rel_err_{name}']:.2%}")
