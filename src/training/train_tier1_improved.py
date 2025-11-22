import argparse
import yaml
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.models.mlp_scalar import MLPScalarSurrogate, get_device, FeatureNormalizer
# from src.data.features import FeatureNormalizer
from src.data.splits import load_split

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def compute_target_weights(y_train, method="inverse_var"):
    """
    Compute weights for each target to balance the loss.
    """
    if method == "none":
        return torch.ones(y_train.shape[1])
    elif method == "inverse_var":
        # Inverse variance weighting
        vars = torch.var(y_train, dim=0)
        # Avoid division by zero
        vars = torch.where(vars > 1e-8, vars, torch.ones_like(vars))
        weights = 1.0 / vars
        # Normalize to mean 1
        weights = weights / weights.mean()
        return weights
    elif method == "manual":
        # Placeholder for manual weights if needed
        return torch.ones(y_train.shape[1])
    else:
        raise ValueError(f"Unknown weighting method: {method}")

def train_one_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, patience, weights=None):
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience//2)
    
    if weights is not None:
        weights = weights.to(device)

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            
            if weights is not None:
                loss = torch.mean(weights * (y_pred - y_batch)**2)
            else:
                loss = criterion(y_pred, y_batch)
                
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                
                if weights is not None:
                    loss = torch.mean(weights * (y_pred - y_batch)**2)
                else:
                    loss = criterion(y_pred, y_batch)
                
                val_loss += loss.item() * X_batch.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
            
    model.load_state_dict(best_state)
    return model, best_val_loss

def main():
    parser = argparse.ArgumentParser(description="Train Tier 1 Improved MLP Ensemble")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Setup
    device = get_device() if config['device'] == 'auto' else torch.device(config['device'])
    print(f"Using device: {device}")
    
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = pd.read_parquet(config['data_path'])
    split = load_split(Path(config['split_path']))
    
    # Feature Engineering
    # H_over_D is already in df
    # Re -> Re_star
    df['Re_star'] = np.log10(df['Re'] / 1e6)
    
    # AoA features
    # Ensure aoa_rad is present
    if 'aoa_rad' not in df.columns:
        df['aoa_rad'] = np.radians(df['aoa'])
        
    df['aoa0_rad'] = np.radians(df['aoa'] - 90.0)
    df['sin_aoa'] = np.sin(df['aoa_rad'])
    df['cos_aoa'] = np.cos(df['aoa_rad'])
    
    # Select features and targets
    feature_cols = []
    feature_cols.extend(config['features']['geometry'])
    feature_cols.append('Re_star') # Hardcoded as per plan
    feature_cols.append('aoa0_rad')
    feature_cols.append('sin_aoa')
    feature_cols.append('cos_aoa')
    
    target_cols = config['targets']
    
    print(f"Features: {feature_cols}")
    print(f"Targets: {target_cols}")
    
    # Prepare tensors
    X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    y = torch.tensor(df[target_cols].values, dtype=torch.float32)
    
    # Split
    train_idx = split['train']
    val_idx = split['val']
    test_idx = split['test']
    
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
    y_test_norm = y_normalizer.transform(y_test) # Just for consistency, though we evaluate on physical units usually
    
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
            output_dim=len(target_cols),
            hidden_dims=config['model']['hidden_dims'],
            activation=config['model']['activation'],
            dropout=config['model']['dropout'],
            batch_norm=config['model']['use_batch_norm']
        ).to(device)
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
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
    y_pred = y_normalizer.inverse_transform(avg_pred_norm).numpy()
    y_true = y_test.numpy()
    
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
        
    # Save results
    results = {
        "metrics": metrics,
        "config": config,
        "feature_cols": feature_cols,
        "target_cols": target_cols
    }
    
    with open(output_dir / "metrics_improved.json", 'w') as f:
        json.dump(results, f, indent=2, cls=json.JSONEncoder)
        
    # Save models and normalizers
    torch.save({
        "models": [m.state_dict() for m in models],
        "x_normalizer": x_normalizer.state_dict(),
        "y_normalizer": y_normalizer.state_dict(),
        "feature_cols": feature_cols,
        "target_cols": target_cols
    }, output_dir / "ensemble_improved_metadata.pt")
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()
