"""
Plotting utilities for visualization of results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str],
    split_name: str = "Test",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create scatter plots of predictions vs actual values for each target.
    
    Args:
        y_true: True values (n_samples, n_targets)
        y_pred: Predicted values (n_samples, n_targets)
        target_names: Names of targets
        split_name: Name of split (for title)
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure
    """
    n_targets = len(target_names)
    n_cols = 3
    n_rows = int(np.ceil(n_targets / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
    axes = axes.flatten() if n_targets > 1 else [axes]
    
    for i, name in enumerate(target_names):
        ax = axes[i]
        y_t = y_true[:, i]
        y_p = y_pred[:, i]
        
        # Scatter plot
        ax.scatter(y_t, y_p, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(y_t.min(), y_p.min())
        max_val = max(y_t.max(), y_p.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
        
        # Compute R²
        from sklearn.metrics import r2_score
        r2 = r2_score(y_t, y_p)
        
        ax.set_xlabel(f'True {name}')
        ax.set_ylabel(f'Predicted {name}')
        ax.set_title(f'{name} (R² = {r2:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_targets, len(axes)):
        axes[i].axis('off')
    
    fig.suptitle(f'{split_name} Set: Predictions vs Actual', fontsize=14, y=1.00)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str],
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot residuals for each target.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        target_names: Target names
        save_path: Optional save path
        
    Returns:
        matplotlib Figure
    """
    n_targets = len(target_names)
    n_cols = 3
    n_rows = int(np.ceil(n_targets / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
    axes = axes.flatten() if n_targets > 1 else [axes]
    
    for i, name in enumerate(target_names):
        ax = axes[i]
        y_t = y_true[:, i]
        y_p = y_pred[:, i]
        residuals = y_t - y_p
        
        # Residual plot
        ax.scatter(y_p, residuals, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
        ax.axhline(0, color='r', linestyle='--', lw=2)
        
        ax.set_xlabel(f'Predicted {name}')
        ax.set_ylabel('Residual')
        ax.set_title(f'{name} Residuals')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_targets, len(axes)):
        axes[i].axis('off')
    
    fig.suptitle('Residual Plots', fontsize=14)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    
    return fig


def plot_mean_coeffs_vs_aoa(
    df: pd.DataFrame,
    pred_df: Optional[pd.DataFrame] = None,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot mean coefficients (Cd, Cl, Cm) vs angle of attack.
    
    Args:
        df: DataFrame with columns: aoa, mean_Cd, mean_Cl, mean_Cm, shape_type
        pred_df: Optional DataFrame with predictions
        save_path: Optional save path
        
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    coeffs = ['mean_Cd', 'mean_Cl', 'mean_Cm']
    labels = ['C_D', 'C_L', 'C_M']
    
    shapes = df['shape_type'].unique()
    colors = sns.color_palette('Set2', len(shapes))
    
    for i, (coeff, label) in enumerate(zip(coeffs, labels)):
        ax = axes[i]
        
        for shape, color in zip(shapes, colors):
            shape_df = df[df['shape_type'] == shape].sort_values('aoa')
            ax.plot(shape_df['aoa'], shape_df[coeff], 'o-', label=f'{shape} (true)',
                   color=color, markersize=8, linewidth=2)
            
            if pred_df is not None:
                shape_pred = pred_df[pred_df['shape_type'] == shape].sort_values('aoa')
                ax.plot(shape_pred['aoa'], shape_pred[coeff], 's--', label=f'{shape} (pred)',
                       color=color, markersize=6, linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('Angle of Attack (deg)')
        ax.set_ylabel(f'$\\overline{{{label}}}$')
        ax.set_title(f'Mean {label} vs AoA')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    
    return fig


def plot_strouhal_vs_params(
    df: pd.DataFrame,
    pred_df: Optional[pd.DataFrame] = None,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot Strouhal number vs various parameters.
    
    Args:
        df: DataFrame with St_peak, Re, aoa, shape_type
        pred_df: Optional predictions
        save_path: Optional save path
        
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    shapes = df['shape_type'].unique()
    colors = sns.color_palette('Set2', len(shapes))
    
    # St vs Re
    ax = axes[0]
    for shape, color in zip(shapes, colors):
        shape_df = df[df['shape_type'] == shape]
        ax.scatter(shape_df['Re'], shape_df['St_peak'], label=f'{shape} (true)',
                  color=color, s=80, alpha=0.7, edgecolors='k')
        
        if pred_df is not None:
            shape_pred = pred_df[pred_df['shape_type'] == shape]
            ax.scatter(shape_pred['Re'], shape_pred['St_peak'], label=f'{shape} (pred)',
                      color=color, s=60, alpha=0.5, marker='s')
    
    ax.set_xlabel('Reynolds Number')
    ax.set_ylabel('Strouhal Number')
    ax.set_title('Strouhal vs Reynolds Number')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # St vs AoA
    ax = axes[1]
    for shape, color in zip(shapes, colors):
        shape_df = df[df['shape_type'] == shape].sort_values('aoa')
        ax.plot(shape_df['aoa'], shape_df['St_peak'], 'o-', label=f'{shape} (true)',
               color=color, markersize=8, linewidth=2)
        
        if pred_df is not None:
            shape_pred = pred_df[pred_df['shape_type'] == shape].sort_values('aoa')
            ax.plot(shape_pred['aoa'], shape_pred['St_peak'], 's--', label=f'{shape} (pred)',
                   color=color, markersize=6, linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Angle of Attack (deg)')
    ax.set_ylabel('Strouhal Number')
    ax.set_title('Strouhal vs AoA')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    
    return fig


def plot_den_hartog_diagram(
    df: pd.DataFrame,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot Den Hartog stability diagram.
    
    Shows S_DH = dCL/dα + CD vs AoA.
    Negative values indicate galloping instability.
    
    Args:
        df: DataFrame with aoa, mean_Cd, mean_Cl, shape_type
        save_path: Optional save path
        
    Returns:
        matplotlib Figure
    """
    from src.eval.metrics import estimate_cl_derivative
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    shapes = df['shape_type'].unique()
    colors = sns.color_palette('Set2', len(shapes))
    
    for shape, color in zip(shapes, colors):
        shape_df = df[df['shape_type'] == shape].sort_values('aoa')
        
        aoa_rad = np.deg2rad(shape_df['aoa'].values)
        cl = shape_df['mean_Cl'].values
        cd = shape_df['mean_Cd'].values
        
        # Estimate dCL/dα
        dCl_dalpha = estimate_cl_derivative(aoa_rad, cl)
        
        # Compute S_DH
        s_dh = dCl_dalpha + cd
        
        ax.plot(shape_df['aoa'], s_dh, 'o-', label=shape, color=color,
               markersize=8, linewidth=2.5)
    
    # Zero line (stability boundary)
    ax.axhline(0, color='red', linestyle='--', linewidth=2, label='Stability boundary')
    
    # Shade unstable region
    ax.axhspan(-10, 0, alpha=0.1, color='red', label='Unstable (galloping)')
    
    ax.set_xlabel('Angle of Attack (deg)', fontsize=12)
    ax.set_ylabel(r'$S_{DH} = \frac{dC_L}{d\alpha} + C_D$', fontsize=12)
    ax.set_title('Den Hartog Stability Diagram', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    
    return fig
