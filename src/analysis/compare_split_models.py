import json
import pandas as pd
from pathlib import Path

def load_metrics(path):
    """Load metrics from JSON file."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, 'r') as f:
        return json.load(f)

def main():
    # Define experiments for both models
    experiments = {
        "Means - Shorter Holdout": "experiments/tier1_means/results_shorter/metrics_means.json",
        "Means - Taller Holdout": "experiments/tier1_means/results_taller/metrics_means.json",
        "Means - Baseline Holdout": "experiments/tier1_means/results_baseline/metrics_means.json",
        "Spectral - Shorter Holdout": "experiments/tier1_spectral/results_shorter/metrics_spectral.json",
        "Spectral - Taller Holdout": "experiments/tier1_spectral/results_taller/metrics_spectral.json",
        "Spectral - Baseline Holdout": "experiments/tier1_spectral/results_baseline/metrics_spectral.json",
    }
    
    results = []
    
    for name, path in experiments.items():
        data = load_metrics(path)
        if data is None:
            print(f"Warning: {path} not found")
            continue
            
        metrics = data['metrics']
        
        row = {"Model": name}
        
        # Get all targets from this experiment
        for target, target_metrics in metrics.items():
            if target == 'log_St_peak':  # Skip transformed space metric
                continue
            row[f"{target}_R2"] = target_metrics.get('R2', None)
            row[f"{target}_RMSE"] = target_metrics.get('RMSE', None)
                
        results.append(row)
        
    df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("COMPREHENSIVE SHAPE HOLDOUT RESULTS (Design Velocity Only)")
    print("="*80)
    print(df.to_string(index=False))
    
    # Calculate average R2 across holdouts for each model
    print("\n" + "="*80)
    print("AVERAGE R² BY TARGET (Across All Shape Holdouts)")
    print("="*80)
    
    # Means model averages
    means_rows = df[df['Model'].str.contains('Means')]
    spectral_rows = df[df['Model'].str.contains('Spectral')]
    
    print("\nMeans Model:")
    for target in ['mean_Cl', 'mean_Cd', 'mean_Cm']:
        col = f"{target}_R2"
        if col in means_rows.columns:
            avg = means_rows[col].mean()
            min_val = means_rows[col].min()
            max_val = means_rows[col].max()
            print(f"  {target:12s}: avg={avg:6.3f}, range=[{min_val:6.3f}, {max_val:6.3f}]")
    
    print("\nSpectral Model:")
    for target in ['St_peak', 'A_peak']:
        col = f"{target}_R2"
        if col in spectral_rows.columns:
            avg = spectral_rows[col].mean()
            min_val = spectral_rows[col].min()
            max_val = spectral_rows[col].max()
            print(f"  {target:12s}: avg={avg:6.3f}, range=[{min_val:6.3f}, {max_val:6.3f}]")
    
    # Save
    df.to_csv("experiments/split_models_comparison.csv", index=False)
    print(f"\nSaved to experiments/split_models_comparison.csv")
    
if __name__ == "__main__":
    main()
