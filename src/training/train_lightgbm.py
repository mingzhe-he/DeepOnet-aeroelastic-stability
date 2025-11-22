import argparse
import yaml
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def main():
    parser = argparse.ArgumentParser(description="Train Tier 1 LightGBM")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = pd.read_parquet(config['data_path'])
    
    # Feature Engineering
    df['Re_star'] = np.log10(df['Re'] / 1e6)
    if 'aoa_rad' not in df.columns:
        df['aoa_rad'] = np.radians(df['aoa'])
        
    df['sin_aoa'] = np.sin(df['aoa_rad'])
    df['cos_aoa'] = np.cos(df['aoa_rad'])
    df['aoa_deg2'] = df['aoa']**2
    df['sin_2aoa'] = np.sin(2 * df['aoa_rad'])
    
    if 'shape_type' in config['features']['optional']:
        df['shape_type'] = df['shape_type'].astype('category')
        
    # Select features
    feature_cols = config['features']['mandatory']
    if 'shape_type' in config['features']['optional']:
        feature_cols.append('shape_type')
        
    target_cols = config['targets']
    
    print(f"Features: {feature_cols}")
    print(f"Targets: {target_cols}")
    
    # K-Fold CV
    n_folds = config['training']['n_folds']
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    metrics = {target: {"R2": [], "RMSE": []} for target in target_cols}
    models = {target: [] for target in target_cols}
    
    # We train one model per target
    for target in target_cols:
        print(f"\nTraining for target: {target}")
        
        y = df[target].values
        X = df[feature_cols]
        
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # LightGBM Dataset
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            params = config['training']['params'].copy()
            
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=params['early_stopping_rounds']), lgb.log_evaluation(0)]
            )
            
            # Predict
            y_pred = model.predict(X_val, num_iteration=model.best_iteration)
            
            # Metrics
            r2 = r2_score(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            
            metrics[target]["R2"].append(float(r2))
            metrics[target]["RMSE"].append(float(rmse))
            
            # Save model (optional, maybe just save final one)
            # models[target].append(model)
            
        # Average metrics
        avg_r2 = np.mean(metrics[target]["R2"])
        avg_rmse = np.mean(metrics[target]["RMSE"])
        print(f"  Avg R2: {avg_r2:.4f}")
        print(f"  Avg RMSE: {avg_rmse:.4f}")
        
        # Train final model on all data
        full_data = lgb.Dataset(X, label=y)
        final_params = config['training']['params'].copy()
        if 'early_stopping_rounds' in final_params:
            del final_params['early_stopping_rounds']
            
        final_model = lgb.train(
            final_params,
            full_data,
            num_boost_round=model.best_iteration
        )
        models[target] = final_model
        
    # Save results
    results = {
        "metrics": metrics,
        "config": config
    }
    
    with open(output_dir / "metrics_lightgbm.json", 'w') as f:
        json.dump(results, f, indent=2, cls=json.JSONEncoder)
        
    joblib.dump(models, output_dir / "lightgbm_models.pkl")
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()
