"""
Training script for Tier 1: Lightweight scalar surrogate.

This script:
1. Loads preprocessed data from summary.parquet
2. Applies case-level splits (no leakage)
3. Trains MLP and baseline models
4. Evaluates on train/val/test sets
5. Saves results and plots
"""

import argparse
import json
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# Local imports
from src.models.mlp_scalar import (
    MLPScalarSurrogate,
    FeatureNormalizer,
    get_device,
    create_mlp_model
)
from src.models.baseline import RidgeBaseline
from src.data.splits import load_split
from src.eval.metrics import compute_metrics, print_metrics_summary
from src.eval.plots import (
    plot_predictions_vs_actual,
    plot_residuals,
    plot_mean_coeffs_vs_aoa,
    plot_strouhal_vs_params,
    plot_den_hartog_diagram
)


def load_data(data_path: Path) -> pd.DataFrame:
    """Load summary data from parquet file."""
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} cases from {data_path}")
    return df


def prepare_features_targets(
    df: pd.DataFrame,
    feature_cols: list,
    target_cols: list
) -> tuple:
    """
    Extract features and targets from DataFrame.
    
    Returns:
        X (np.ndarray), y (np.ndarray), feature_names, target_names
    """
    X = df[feature_cols].values
    y = df[target_cols].values
    
    return X, y, feature_cols, target_cols


def train_mlp(
    model: MLPScalarSurrogate,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 200,
    lr: float = 1e-3,
    device: torch.device = None,
    patience: int = 20
) -> dict:
    """
    Train MLP model.
    
    Args:
        model: MLP model
        train_loader: Training data loader
        val_loader: Validation data loader
        n_epochs: Maximum number of epochs
        lr: Learning rate
        device: Device to train on
        patience: Early stopping patience
        
    Returns:
        Training history
    """
    if device is None:
        device = get_device()
    
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    history = {
        'train_loss': [],
        'val_loss': [],
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                
                val_loss += loss.item() * X_batch.size(0)
        
        val_loss /= len(val_loader.dataset)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {
                'model': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
            }
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}: "
                  f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state['model'])
        print(f"Restored best model from epoch {best_state['epoch']+1} "
              f"(val_loss={best_state['val_loss']:.6f})")
    
    return history


def evaluate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    target_names: list,
    is_torch: bool = True,
    device: torch.device = None
) -> tuple:
    """
    Evaluate model and return predictions + metrics.
    
    Returns:
        y_pred (np.ndarray), metrics (dict)
    """
    if is_torch:
        if device is None:
            device = get_device()
        model.eval()
        with torch.no_grad():
            X_torch = torch.FloatTensor(X).to(device)
            y_pred_torch = model(X_torch)
            y_pred = y_pred_torch.cpu().numpy()
    else:
        y_pred = model.predict(X)
    
    metrics = compute_metrics(y, y_pred, target_names)
    
    return y_pred, metrics


def main(config_path: Path):
    """Main training function with Ensembling."""
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    print(f"\nLoaded config from {config_path}")
    print(json.dumps(config, indent=2))
    
    # Set base random seed
    base_seed = config.get('random_seed', 42)
    
    # Paths
    data_path = Path(config['data_path'])
    split_path = Path(config['split_path'])
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_data(data_path)
    
    # Load split
    split = load_split(split_path)
    print(f"\nUsing split: {split['metadata']['split_type']}")
    
    # Prepare features and targets
    feature_cols = config['feature_cols']
    target_cols = config['target_cols']
    
    X, y, feature_names, target_names = prepare_features_targets(
        df, feature_cols, target_cols
    )
    
    # Split data
    X_train = X[split['train']]
    y_train = y[split['train']]
    X_val = X[split['val']]
    y_val = y[split['val']]
    X_test = X[split['test']]
    y_test = y[split['test']]
    
    print(f"\nData shapes:")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Val:   X={X_val.shape}, y={y_val.shape}")
    print(f"  Test:  X={X_test.shape}, y={y_test.shape}")
    
    # Normalize features (fit on train only)
    feature_normalizer = FeatureNormalizer()
    target_normalizer = FeatureNormalizer()
    
    X_train_norm = feature_normalizer.fit_transform(torch.FloatTensor(X_train))
    y_train_norm = target_normalizer.fit_transform(torch.FloatTensor(y_train))
    X_val_norm = feature_normalizer.transform(torch.FloatTensor(X_val))
    y_val_norm = target_normalizer.transform(torch.FloatTensor(y_val))
    X_test_norm = feature_normalizer.transform(torch.FloatTensor(X_test))
    
    # Create data loaders
    batch_size = config.get('batch_size', 32)
    train_dataset = TensorDataset(X_train_norm, y_train_norm)
    val_dataset = TensorDataset(X_val_norm, y_val_norm)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = get_device()
    print(f"\nUsing device: {device}")
    
    # --- Ensemble Training ---
    n_models = 5
    models = []
    histories = []
    
    print("\n" + "="*60)
    print(f"Training Ensemble of {n_models} MLP Models")
    print("="*60)
    
    for i in range(n_models):
        print(f"\nTraining Model {i+1}/{n_models}...")
        
        # Set seed for this model
        current_seed = base_seed + i
        np.random.seed(current_seed)
        torch.manual_seed(current_seed)
        
        model = create_mlp_model(
            n_features=len(feature_cols),
            n_targets=len(target_cols),
            hidden_dims=config.get('hidden_dims', [64, 64, 32]),
            dropout=config.get('dropout', 0.1),
            batch_norm=config.get('batch_norm', True)
        )
        
        history = train_mlp(
            model,
            train_loader,
            val_loader,
            n_epochs=config.get('n_epochs', 200),
            lr=config.get('learning_rate', 1e-3),
            device=device,
            patience=config.get('patience', 20)
        )
        
        models.append(model)
        histories.append(history)
        
        # Save individual model
        torch.save(model.state_dict(), output_dir / f'mlp_model_{i}.pt')

    # --- Ensemble Evaluation ---
    print("\nEvaluating Ensemble...")
    
    def predict_ensemble(models, X_norm):
        predictions = []
        with torch.no_grad():
            X_norm = X_norm.to(device)
            for model in models:
                model.eval()
                pred = model(X_norm).cpu()
                predictions.append(pred)
        # Average predictions
        avg_pred = torch.stack(predictions).mean(dim=0)
        return avg_pred

    # Get ensemble predictions (normalized)
    y_train_pred_norm = predict_ensemble(models, X_train_norm)
    y_val_pred_norm = predict_ensemble(models, X_val_norm)
    y_test_pred_norm = predict_ensemble(models, X_test_norm)
    
    # Denormalize
    y_train_pred = target_normalizer.inverse_transform(y_train_pred_norm).numpy()
    y_val_pred = target_normalizer.inverse_transform(y_val_pred_norm).numpy()
    y_test_pred = target_normalizer.inverse_transform(y_test_pred_norm).numpy()
    
    # Compute metrics
    metrics_train = compute_metrics(y_train, y_train_pred, target_names)
    metrics_val = compute_metrics(y_val, y_val_pred, target_names)
    metrics_test = compute_metrics(y_test, y_test_pred, target_names)
    
    print_metrics_summary(metrics_train, "Ensemble MLP Train")
    print_metrics_summary(metrics_val, "Ensemble MLP Val")
    print_metrics_summary(metrics_test, "Ensemble MLP Test")
    
    # Train baseline (Ridge)
    print("\n" + "="*60)
    print("Training Ridge Baseline")
    print("="*60)
    
    ridge_model = RidgeBaseline(alpha=config.get('ridge_alpha', 1.0))
    ridge_model.fit(X_train, y_train, target_names)
    
    y_train_pred_ridge = ridge_model.predict(X_train)
    y_test_pred_ridge = ridge_model.predict(X_test)
    
    metrics_train_ridge = compute_metrics(y_train, y_train_pred_ridge, target_names)
    metrics_test_ridge = compute_metrics(y_test, y_test_pred_ridge, target_names)
    
    print_metrics_summary(metrics_train_ridge, "Ridge Train")
    print_metrics_summary(metrics_test_ridge, "Ridge Test")
    
    # Save results
    results = {
        'config': config,
        'split_metadata': split['metadata'],
        'mlp_ensemble': {
            'train': metrics_train,
            'val': metrics_val,
            'test': metrics_test,
            'histories': histories,
        },
        'ridge': {
            'train': metrics_train_ridge,
            'test': metrics_test_ridge,
        }
    }
    
    results_path = output_dir / 'metrics.json'
    with open(results_path, 'w') as f:
        results_json = json.loads(json.dumps(results, default=float))
        json.dump(results_json, f, indent=2)
    print(f"\nSaved metrics to {results_path}")
    
    # Save metadata for inference
    torch.save({
        'feature_normalizer': feature_normalizer.state_dict(),
        'target_normalizer': target_normalizer.state_dict(),
        'config': config,
        'feature_names': feature_names,
        'target_names': target_names,
        'n_models': n_models
    }, output_dir / 'ensemble_metadata.pt')
    print(f"Saved ensemble metadata to {output_dir / 'ensemble_metadata.pt'}")
    
    # Generate plots
    print("\n" + "="*60)
    print("Generating Plots")
    print("="*60)
    
    fig = plot_predictions_vs_actual(
        y_test, y_test_pred, target_names, split_name="Test (Ensemble MLP)",
        save_path=output_dir / 'predictions_vs_actual_mlp.png'
    )
    plt.close(fig)
    
    fig = plot_predictions_vs_actual(
        y_test, y_test_pred_ridge, target_names, split_name="Test (Ridge)",
        save_path=output_dir / 'predictions_vs_actual_ridge.png'
    )
    plt.close(fig)
    
    fig = plot_residuals(
        y_test, y_test_pred, target_names,
        save_path=output_dir / 'residuals_mlp.png'
    )
    plt.close(fig)
    
    # Create DataFrames with predictions for plotting
    test_df = df.iloc[split['test']].copy()
    test_df_pred = test_df.copy()
    for i, col in enumerate(target_cols):
        test_df_pred[col] = y_test_pred[:, i]
    
    fig = plot_mean_coeffs_vs_aoa(
        test_df, test_df_pred,
        save_path=output_dir / 'mean_coeffs_vs_aoa.png'
    )
    plt.close(fig)
    
    if 'St_peak' in target_cols:
        fig = plot_strouhal_vs_params(
            test_df, test_df_pred,
            save_path=output_dir / 'strouhal_vs_params.png'
        )
        plt.close(fig)
    
    # Den Hartog diagram
    if 'mean_Cd' in target_cols and 'mean_Cl' in target_cols:
        fig = plot_den_hartog_diagram(
            test_df,
            save_path=output_dir / 'den_hartog_diagram.png'
        )
        plt.close(fig)
    
    print("\nTraining complete!")
    print(f"Results saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Tier 1 models')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML file')
    
    args = parser.parse_args()
    
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    
    main(Path(args.config))
