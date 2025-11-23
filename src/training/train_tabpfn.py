import argparse
import yaml
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tabpfn import TabPFNRegressor

from src.data.splits import load_split

def main():
    parser = argparse.ArgumentParser(description="Train Tier 1 TabPFN")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--design-only", action="store_true", help="Train only on design velocity")
    parser.add_argument("--split-path", type=str, default=None, help="Override split path")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = config.get('device', 'auto')
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # TabPFN might prefer 'cpu' on MPS? Or maybe it supports MPS.
        # Let's stick to 'cpu' or 'cuda' as TabPFN relies on PyTorch.
        # On Mac, 'mps' is available but TabPFN might not explicitly support it in its internal checks.
        # Safe bet: use 'cpu' if no cuda, unless we want to try 'mps'.
        if torch.backends.mps.is_available():
             device = 'mps'
             
    print(f"Using device: {device}")
    
    # Load data
    df = pd.read_parquet(config['data_path'])
    
    design_indices = None
    if args.design_only:
        print("Filtering for design velocity (U ~ 21.5 m/s)...")
        design_indices = df[df['U_ref'] > 20.0].index.tolist()
        print(f"Design velocity cases: {len(design_indices)}")
        
    if args.split_path:
        split = load_split(Path(args.split_path))
    else:
        split = load_split(Path(config['split_path']))
        
    # Feature Engineering
    df['Re_star'] = np.log10(df['Re'] / 1e6)
    if 'aoa_rad' not in df.columns:
        df['aoa_rad'] = np.radians(df['aoa'])
    df['aoa0_rad'] = np.radians(df['aoa'] - 90.0)
    
    # Select features
    feature_cols = config['features']['mandatory']
    if args.design_only and "Re_star" in feature_cols:
        # Remove Re_star if present and design-only
        # But config might not have it.
        pass
        
    target_cols = config['targets']
    
    print(f"Features: {feature_cols}")
    print(f"Targets: {target_cols}")
    
    # Split
    train_idx = split['train']
    val_idx = split['val'] # TabPFN doesn't really need val set for early stopping, but we can use it for evaluation
    test_idx = split['test']
    
    if args.design_only:
        design_set = set(design_indices)
        train_idx = [i for i in train_idx if i in design_set]
        val_idx = [i for i in val_idx if i in design_set]
        test_idx = [i for i in test_idx if i in design_set]
        print(f"Filtered split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
        
    # Combine train and val for TabPFN as it's a prior-data method?
    # Or just train on 'train' and evaluate on 'test'.
    # Let's stick to 'train' only to be comparable with other methods (which use val for early stopping).
    # Or we can merge train+val since TabPFN doesn't overfit in the same way?
    # Let's just use train_idx.
    
    X = df[feature_cols]
    y = df[target_cols]
    
    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]
    X_test = X.loc[test_idx]
    y_test = y.loc[test_idx]
    
    metrics = {}
    
    for target in target_cols:
        print(f"\nTraining TabPFN for {target}...")
        
        y_train_target = y_train[target].values
        y_test_target = y_test[target].values
        
        # Initialize TabPFN
        # N_ensemble_configurations controls speed/accuracy trade-off
        regressor = TabPFNRegressor(device=device)
        
        # Fit
        # TabPFN expects numpy arrays or pandas dfs
        regressor.fit(X_train, y_train_target)
        
        # Predict
        # Check if predict accepts N_ensemble_configurations
        try:
            y_pred = regressor.predict(X_test, N_ensemble_configurations=config['training']['N_ensemble_configurations'])
        except TypeError:
             print("Warning: predict does not accept N_ensemble_configurations, using default.")
             y_pred = regressor.predict(X_test)
        
        # Metrics
        r2 = r2_score(y_test_target, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_target, y_pred))
        
        metrics[target] = {
            "R2": float(r2),
            "RMSE": float(rmse)
        }
        print(f"  R2: {r2:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        
    # Save results
    results = {
        "metrics": metrics,
        "config": config
    }
    
    with open(output_dir / "metrics_tabpfn.json", 'w') as f:
        json.dump(results, f, indent=2, cls=json.JSONEncoder)
        
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()
