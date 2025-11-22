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
from src.data.splits import load_split

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def compute_finite_difference_derivatives(df, target_col, wrt_col, group_col='shape_variant'):
    """
    Compute finite difference derivatives for a target column with respect to a feature column,
    grouped by shape.
    """
    df = df.copy()
    df['derivative'] = np.nan
    
    for name, group in df.groupby(group_col):
        # Sort by wrt_col (e.g. aoa)
        group = group.sort_values(wrt_col)
        
        y = group[target_col].values
        x = group[wrt_col].values
        
        # Central difference
        dy_dx = np.gradient(y, x)
        
        df.loc[group.index, 'derivative'] = dy_dx
        
    return df['derivative']

def train_model_with_derivative(
    model, train_loader, val_loader, optimizer, device, epochs, patience,
    lambda_deriv, lambda_curvature, x_normalizer, wrt_idx, target_idx
):
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience//2)
    criterion = nn.MSELoss()
    
    # Scale factor for derivative: dY/dX = (dY_norm/dX_norm) * (std_Y / std_X)
    # But here we are predicting Y_physical directly (no y_normalizer for model output, or we handle it)
    # Wait, the plan says "Targets and derivative remain in physical units".
    # So the model output is physical.
    # Input X is normalized.
    # So dY_phys / dX_phys = (dY_phys / dX_norm) * (dX_norm / dX_phys)
    # dX_norm / dX_phys = 1 / std_X
    
    # We need std_X for the wrt feature
    std_wrt = x_normalizer.std[0, wrt_idx].item()
    scale_factor = 1.0 / std_wrt
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_reg_loss = 0.0
        train_deriv_loss = 0.0
        
        for X_batch, y_batch, deriv_batch in train_loader:
            X_batch = X_batch.to(device).requires_grad_(True)
            y_batch = y_batch.to(device)
            deriv_batch = deriv_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            y_pred = model(X_batch)
            
            # Main loss
            reg_loss = criterion(y_pred, y_batch)
            
            # Derivative loss
            # We want d(y_pred[:, target_idx]) / d(X_batch[:, wrt_idx])
            
            # Compute gradients
            # We need to compute gradients of the specific target output w.r.t input
            target_output = y_pred[:, target_idx].sum()
            
            grads = torch.autograd.grad(target_output, X_batch, create_graph=True)[0]
            
            # Extract gradient w.r.t the specific feature
            dY_dX_norm = grads[:, wrt_idx]
            
            # Convert to physical derivative
            dY_dX_phys = dY_dX_norm * scale_factor
            
            deriv_loss = criterion(dY_dX_phys, deriv_batch)
            
            # Total loss
            loss = reg_loss + lambda_deriv * deriv_loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            train_reg_loss += reg_loss.item() * X_batch.size(0)
            train_deriv_loss += deriv_loss.item() * X_batch.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Val
        model.eval()
        val_loss = 0.0
        
        # For validation, we can just check regression loss or both
        # Let's check total loss to be consistent with early stopping
        with torch.no_grad(): # We can't use no_grad if we want derivatives in val? 
            # Actually we can use torch.enable_grad() context if needed, but usually we just track regression loss for validation
            # or we can track both. Let's track regression loss for simplicity and robustness.
            # But if we want to stop based on derivative performance, we should include it.
            # Let's include it but we need enable_grad for the derivative part.
            pass
            
        # Re-implement val loop with grad enabled for derivative computation
        val_loss = 0.0
        with torch.set_grad_enabled(True): # Enable grad for derivative
            for X_batch, y_batch, deriv_batch in val_loader:
                X_batch = X_batch.to(device).requires_grad_(True)
                y_batch = y_batch.to(device)
                deriv_batch = deriv_batch.to(device)
                
                y_pred = model(X_batch)
                reg_loss = criterion(y_pred, y_batch)
                
                target_output = y_pred[:, target_idx].sum()
                grads = torch.autograd.grad(target_output, X_batch, create_graph=True)[0]
                dY_dX_norm = grads[:, wrt_idx]
                dY_dX_phys = dY_dX_norm * scale_factor
                
                deriv_loss = criterion(dY_dX_phys, deriv_batch)
                
                loss = reg_loss + lambda_deriv * deriv_loss
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
    parser = argparse.ArgumentParser(description="Train Tier 1 MLP with Derivative")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    device = get_device() if config['device'] == 'auto' else torch.device(config['device'])
    print(f"Using device: {device}")
    
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = pd.read_parquet(config['data_path'])
    split = load_split(Path(config['split_path']))
    
    # Feature Engineering
    df['Re_star'] = np.log10(df['Re'] / 1e6)
    if 'aoa_rad' not in df.columns:
        df['aoa_rad'] = np.radians(df['aoa'])
    df['aoa0_rad'] = np.radians(df['aoa'] - 90.0)
    
    # Compute Derivatives
    target_deriv = config['derivative']['target'] # e.g. mean_Cl
    wrt_deriv = config['derivative']['wrt'] # e.g. aoa0_rad
    
    # We need to compute derivative w.r.t aoa0_rad.
    # Note: d/d(aoa0_rad) is same as d/d(aoa_rad) since they differ by constant.
    # We'll use the column specified in config.
    
    print(f"Computing derivatives of {target_deriv} w.r.t {wrt_deriv}...")
    df['target_derivative'] = compute_finite_difference_derivatives(
        df, target_deriv, wrt_deriv, group_col='shape_variant'
    )
    
    # Drop NaNs (endpoints of finite diff might be less accurate, but np.gradient handles edges)
    # np.gradient preserves shape.
    
    # Features and Targets
    feature_cols = config['features']['geometry'] + ['Re_star'] + config['features']['angle']
    # Note: config['features']['angle'] is ["aoa_rad"] in config, but we want "aoa0_rad" based on plan?
    # Config says: angle: ["aoa_rad"] # Will be used to generate aoa0_rad
    # But in the code I should use what I generated.
    # Let's stick to the plan: Inputs: [H_over_D, Re_star, aoa0_rad]
    
    feature_cols = ['H_over_D', 'Re_star', 'aoa0_rad']
    target_cols = config['targets']
    
    print(f"Features: {feature_cols}")
    print(f"Targets: {target_cols}")
    
    # Prepare tensors
    X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    y = torch.tensor(df[target_cols].values, dtype=torch.float32)
    derivs = torch.tensor(df['target_derivative'].values, dtype=torch.float32)
    
    # Split
    train_idx = split['train']
    val_idx = split['val']
    test_idx = split['test']
    
    # Normalize Inputs ONLY
    x_normalizer = FeatureNormalizer()
    X_train = X[train_idx]
    x_normalizer.fit(X_train)
    
    X_train_norm = x_normalizer.transform(X[train_idx])
    X_val_norm = x_normalizer.transform(X[val_idx])
    X_test_norm = x_normalizer.transform(X[test_idx])
    
    # Targets remain physical
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]
    
    deriv_train = derivs[train_idx]
    deriv_val = derivs[val_idx]
    deriv_test = derivs[test_idx]
    
    # Dataloaders
    batch_size = config['training']['batch_size']
    train_ds = torch.utils.data.TensorDataset(X_train_norm, y_train, deriv_train)
    val_ds = torch.utils.data.TensorDataset(X_val_norm, y_val, deriv_val)
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # Model
    set_seed(42)
    model = MLPScalarSurrogate(
        input_dim=len(feature_cols),
        output_dim=len(target_cols),
        hidden_dims=config['model']['hidden_dims'],
        activation=config['model']['activation'],
        dropout=config['model']['dropout'],
        batch_norm=config['model']['batch_norm']
    ).to(device)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Indices for derivative
    try:
        wrt_idx = feature_cols.index(wrt_deriv)
        target_idx = target_cols.index(target_deriv)
    except ValueError as e:
        print(f"Error finding indices: {e}")
        print(f"Features: {feature_cols}")
        print(f"Targets: {target_cols}")
        return

    print(f"Training with derivative reg on target idx {target_idx} w.r.t feature idx {wrt_idx}")
    
    model, best_loss = train_model_with_derivative(
        model, train_loader, val_loader, optimizer, device,
        epochs=config['training']['epochs'],
        patience=config['training']['patience'],
        lambda_deriv=config['derivative']['lambda_deriv'],
        lambda_curvature=config['derivative']['lambda_curvature'],
        x_normalizer=x_normalizer,
        wrt_idx=wrt_idx,
        target_idx=target_idx
    )
    
    # Evaluation
    model.eval()
    
    # Predict on test
    with torch.no_grad():
        y_pred = model(X_test_norm.to(device)).cpu().numpy()
        
    # Predict derivatives on test
    # Need grad
    X_test_tensor = X_test_norm.to(device).requires_grad_(True)
    y_pred_tensor = model(X_test_tensor)
    target_output = y_pred_tensor[:, target_idx].sum()
    grads = torch.autograd.grad(target_output, X_test_tensor)[0]
    
    std_wrt = x_normalizer.std[0, wrt_idx].item()
    scale_factor = 1.0 / std_wrt
    
    deriv_pred = (grads[:, wrt_idx] * scale_factor).cpu().detach().numpy()
    
    # Metrics
    metrics = {}
    y_true = y_test.numpy()
    deriv_true = deriv_test.numpy()
    
    for i, target in enumerate(target_cols):
        metrics[target] = {
            "R2": float(r2_score(y_true[:, i], y_pred[:, i])),
            "RMSE": float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])))
        }
        print(f"{target}: R2={metrics[target]['R2']:.4f}")
        
    # Derivative metrics
    metrics['derivative'] = {
        "R2": float(r2_score(deriv_true, deriv_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(deriv_true, deriv_pred)))
    }
    print(f"Derivative ({target_deriv} wrt {wrt_deriv}): R2={metrics['derivative']['R2']:.4f}")
    
    # Save
    results = {
        "metrics": metrics,
        "config": config
    }
    
    with open(output_dir / "metrics_4b_mlp.json", 'w') as f:
        json.dump(results, f, indent=2, cls=json.JSONEncoder)
        
    torch.save({
        "model": model.state_dict(),
        "x_normalizer": x_normalizer.state_dict(),
        "feature_cols": feature_cols,
        "target_cols": target_cols
    }, output_dir / "model_4b_mlp.pt")
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()
