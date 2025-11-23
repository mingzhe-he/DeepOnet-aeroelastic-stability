import argparse
import yaml
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score

from src.rom.galloping_gp import GallopingGPROM
from src.rom.shedding_model import SheddingRegimeModel
from src.data.splits import load_split

def main():
    parser = argparse.ArgumentParser(description="Train Tier 1 ROM")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--design-only", action="store_true", help="Train only on design velocity")
    parser.add_argument("--split-path", type=str, default=None, help="Override split path")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
        
    # Select features
    variant = config['training']['variant']
    feature_cols = config['features'][variant]
    if args.design_only and "Re_star" in feature_cols:
        feature_cols.remove("Re_star")
        
    print(f"Using features: {feature_cols}")
    
    # Split
    train_idx = split['train']
    test_idx = split['test']
    
    if args.design_only:
        design_set = set(design_indices)
        train_idx = [i for i in train_idx if i in design_set]
        test_idx = [i for i in test_idx if i in design_set]
        print(f"Filtered split sizes: Train={len(train_idx)}, Test={len(test_idx)}")
    
    df_train = df.loc[train_idx]
    df_test = df.loc[test_idx]
    
    # 1. Train Galloping GP ROM
    print("Training Galloping GP ROM...")
    gp_rom = GallopingGPROM(
        n_restarts_optimizer=config['galloping_gp']['n_restarts']
    )
    gp_rom.from_dataframe(df_train, feature_cols)
    
    # 2. Train Shedding Model
    print("Training Shedding Model...")
    shedding_model = SheddingRegimeModel(
        n_regimes=config['shedding_model']['n_regimes'],
        regressor_type=config['shedding_model']['regressor']
    )
    shedding_model.from_dataframe(df_train, feature_cols, target_col="St_peak")
    
    # Evaluation
    print("Evaluating...")
    
    # Mean Coefficients
    cl_pred, cd_pred = gp_rom.predict(df_test[feature_cols].values)
    cl_true = df_test['mean_Cl'].values
    cd_true = df_test['mean_Cd'].values
    
    metrics = {
        "mean_Cl": {
            "R2": float(r2_score(cl_true, cl_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(cl_true, cl_pred)))
        },
        "mean_Cd": {
            "R2": float(r2_score(cd_true, cd_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(cd_true, cd_pred)))
        }
    }
    
    # Strouhal
    st_pred = shedding_model.predict(df_test[feature_cols].values)
    st_true = df_test['St_peak'].values
    
    # Handle NaNs in prediction (if any)
    mask = ~np.isnan(st_pred)
    if np.sum(mask) < len(st_pred):
        print(f"Warning: {len(st_pred) - np.sum(mask)} NaN predictions in Strouhal model")
        
    metrics["St_peak"] = {
        "R2": float(r2_score(st_true[mask], st_pred[mask])),
        "RMSE": float(np.sqrt(mean_squared_error(st_true[mask], st_pred[mask])))
    }
    
    print(f"mean_Cl R2: {metrics['mean_Cl']['R2']:.4f}")
    print(f"mean_Cd R2: {metrics['mean_Cd']['R2']:.4f}")
    print(f"St_peak R2: {metrics['St_peak']['R2']:.4f}")
    
    # Galloping Stability (Den Hartog)
    # We need to estimate derivatives for test set
    # This is tricky because we don't have ground truth derivatives easily unless we computed them
    # But we can compare the Sign of S_DH if we had ground truth S_DH.
    # For now, let's just save the models and metrics.
    
    # Save
    results = {
        "metrics": metrics,
        "config": config
    }
    
    with open(output_dir / "metrics_rom.json", 'w') as f:
        json.dump(results, f, indent=2, cls=json.JSONEncoder)
        
    joblib.dump(gp_rom, output_dir / "galloping_gprom.pkl")
    joblib.dump(shedding_model, output_dir / "shedding_model.pkl")
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()
