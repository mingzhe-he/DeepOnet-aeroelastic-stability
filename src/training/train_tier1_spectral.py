import argparse
import yaml
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.models.mlp_scalar import MLPScalarSurrogate, get_device, FeatureNormalizer
from src.data.splits import load_split

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def compute_target_weights(y_train_norm, method="inverse_variance"):
    """Compute per-target loss weights."""
    if method == "inverse_variance":
        variances = torch.var(y_train_norm, dim=0)
        weights = 1.0 / (variances + 1e-8)
        weights = weights / weights.sum() * len(weights)
    elif method == "uniform":
        weights = torch.ones(y_train_norm.shape[1])
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    return weights

def train_one_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=1000, patience=50, weights=None):
    """Train a single model with early stopping."""
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            
            if weights is not None:
                loss = ((y_pred - y_batch)**2 * weights.to(device)).mean()
            else:
                loss = criterion(y_pred, y_batch)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = model(X_batch)
                
                if weights is not None:
                    loss = ((y_pred - y_batch)**2 * weights.to(device)).mean()
                else:
                    loss = criterion(y_pred, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
            
    return model, best_val_loss

def main():
    parser = argparse.ArgumentParser(description="Train Tier 1 Spectral Model")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--design-only", action="store_true", help="Train only on design velocity (U=21.5)")
    parser.add_argument("--weight-decay", type=float, default=None, help="Override weight decay")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--split-path", type=str, default=None, help="Override split path")
    parser.add_argument("--no-log-transform", action="store_true", help="Disable log transform for St")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Setup
    device = get_device() if config['device'] == 'auto' else torch.device(config['device'])
    print(f"Using device: {device}")
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = pd.read_parquet(config['data_path'])
    if args.split_path:
        split = load_split(Path(args.split_path))
    else:
        split = load_split(Path(config['split_path']))
    
    # Feature Engineering
    df['Re_star'] = np.log10(df['Re'] / 1e6)
    
    design_indices = None
    if args.design_only:
        print("Filtering for design velocity (U ~ 21.5 m/s)...")
        design_indices = df[df['U_ref'] > 20.0].index.tolist()
        print(f"Design velocity cases: {len(design_indices)}")
    
    # AoA features
    if 'aoa_rad' not in df.columns:
        df['aoa_rad'] = np.radians(df['aoa'])
        
    df['aoa0_rad'] = np.radians(df['aoa'] - 90.0)
    df['sin_aoa'] = np.sin(df['aoa_rad'])
    df['cos_aoa'] = np.cos(df['aoa_rad'])
    
    # Polynomial shape features
    df['H_over_D_sq'] = df['H_over_D'] ** 2
    
    # Target transform for St_peak
    use_log_transform = not args.no_log_transform and config.get('use_log_transform_st', True)
    if use_log_transform:
        print("Using log transform for St_peak")
        df['log_St_peak'] = np.log(df['St_peak'])
    
    # Select features and targets
    feature_cols = []
    feature_cols.extend(config['features']['geometry'])
    # Add polynomial feature
    if 'H_over_D_sq' in df.columns and config['features'].get('use_polynomial', True):
        feature_cols.append('H_over_D_sq')
    
    if not args.design_only:
        feature_cols.append('Re_star')
    feature_cols.append('aoa0_rad')
    
    # Targets: St_peak and A_peak
    target_cols = config['targets']  # Should be ['St_peak', 'A_peak']
    
    # If using log transform, replace St_peak with log_St_peak
    target_cols_transformed = []
    st_idx = None
    for i, t in enumerate(target_cols):
        if t == 'St_peak' and use_log_transform:
            target_cols_transformed.append('log_St_peak')
            st_idx = i
        else:
            target_cols_transformed.append(t)
    
    print(f"Features: {feature_cols}")
    print(f"Targets: {target_cols_transformed}")
    
    # Prepare tensors
    X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    y = torch.tensor(df[target_cols_transformed].values, dtype=torch.float32)
    
    # Split
    train_idx = split['train']
    val_idx = split['val']
    test_idx = split['test']
    
    if args.design_only:
        design_set = set(design_indices)
        train_idx = [i for i in train_idx if i in design_set]
        val_idx = [i for i in val_idx if i in design_set]
        test_idx = [i for i in test_idx if i in design_set]
        print(f"Filtered split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    # Normalize
    x_normalizer = FeatureNormalizer()
    y_normalizer = FeatureNormalizer()
    
    X_train_norm = x_normalizer.fit_transform(X_train)
    y_train_norm = y_normalizer.fit_transform(y_train)
    
    X_val_norm = x_normalizer.transform(X_val)
    y_val_norm = y_normalizer.transform(y_val)
    
    X_test_norm = x_normalizer.transform(X_test)
    y_test_norm = y_normalizer.transform(y_test)
    
    # Dataloaders
    batch_size = config['training']['batch_size']
    train_ds = torch.utils.data.TensorDataset(X_train_norm, y_train_norm)
    val_ds = torch.utils.data.TensorDataset(X_val_norm, y_val_norm)
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # Weights
    weights = compute_target_weights(y_train_norm, method=config['training']['loss_weighting'])
    print(f"Target weights: {weights}")
    
    # Train Ensemble
    n_models = config['model']['n_models']
    models = []
    
    for i in range(n_models):
        print(f"Training model {i+1}/{n_models}...")
        set_seed(42 + i)
        
        model = MLPScalarSurrogate(
            input_dim=len(feature_cols),
            output_dim=len(target_cols_transformed),
            hidden_dims=config['model']['hidden_dims'],
            activation=config['model']['activation'],
            dropout=config['model']['dropout'],
            batch_norm=config['model']['use_batch_norm']
        ).to(device)
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=args.weight_decay if args.weight_decay is not None else config['training']['weight_decay']
        )
        
        criterion = nn.MSELoss()
        
        model, best_loss = train_one_model(
            model, train_loader, val_loader, criterion, optimizer, device,
            epochs=config['training']['epochs'],
            patience=config['training']['patience'],
            weights=weights
        )
        
        models.append(model)
        
    # Evaluation
    print("Evaluating ensemble...")
    ensemble_preds_norm = []
    
    for model in models:
        model.eval()
        with torch.no_grad():
            pred = model(X_test_norm.to(device)).cpu()
            ensemble_preds_norm.append(pred)
            
    # Average predictions (in normalized space)
    avg_pred_norm = torch.stack(ensemble_preds_norm).mean(dim=0)
    
    # Denormalize
    y_pred_transformed = y_normalizer.inverse_transform(avg_pred_norm).numpy()
    y_true_transformed = y_test.numpy()
    
    # Inverse transform St if needed
    if use_log_transform and st_idx is not None:
        y_pred = y_pred_transformed.copy()
        y_true = y_true_transformed.copy()
        # Convert log_St back to St
        y_pred[:, st_idx] = np.exp(y_pred_transformed[:, st_idx])
        y_true[:, st_idx] = np.exp(y_true_transformed[:, st_idx])
    else:
        y_pred = y_pred_transformed
        y_true = y_true_transformed
    
    # Metrics
    metrics = {}
    for i, target in enumerate(target_cols):
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        
        metrics[target] = {
            "R2": float(r2),
            "RMSE": float(rmse),
            "MAE": float(mae)
        }
        print(f"{target}: R2={r2:.4f}, RMSE={rmse:.4f}")
        
    # Also report metrics in transformed space for St if using log
    if use_log_transform and st_idx is not None:
        r2_log = r2_score(y_true_transformed[:, st_idx], y_pred_transformed[:, st_idx])
        print(f"log_St_peak R2 (transformed space): {r2_log:.4f}")
        metrics['log_St_peak'] = {"R2": float(r2_log)}
    
    # Save results
    results = {
        "metrics": metrics,
        "config": config,
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "use_log_transform": use_log_transform
    }
    
    with open(output_dir / "metrics_spectral.json", 'w') as f:
        json.dump(results, f, indent=2, cls=json.JSONEncoder)
        
    # Save models and normalizers
    torch.save({
        "models": [m.state_dict() for m in models],
        "x_normalizer": x_normalizer.state_dict(),
        "y_normalizer": y_normalizer.state_dict(),
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "target_cols_transformed": target_cols_transformed,
        "use_log_transform": use_log_transform,
        "st_idx": st_idx
    }, output_dir / "ensemble_spectral.pt")
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()
