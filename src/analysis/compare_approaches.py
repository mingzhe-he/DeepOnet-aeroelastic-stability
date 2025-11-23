import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_metrics(path):
    path = Path(path)
    if not path.exists():
        print(f"Warning: {path} not found")
        return None
    with open(path, 'r') as f:
        return json.load(f)

def main():
    # Define experiments
    experiments = {
        "Approach 1 (MLP Ensemble)": "experiments/tier1_improved/results_shorter/metrics_improved.json",
        "Approach 2 (Derivative MLP)": "experiments/tier1_4b_mlp/results/metrics_4b_mlp.json",
        "Approach 3 (ROM)": "experiments/tier1_rom/results/metrics_rom.json",
        "Approach 4 (LightGBM)": "experiments/tier1_lightgbm/results/metrics_lightgbm.json",
        "Approach 5 (TabPFN)": "experiments/tier1_tabpfn/results/metrics_tabpfn.json"
    }
    
    results = []
    
    for name, path in experiments.items():
        data = load_metrics(path)
        if data is None:
            continue
            
        metrics = data['metrics']
        
        # Extract R2 for key targets
        row = {"Approach": name}
        
        targets = ["mean_Cl", "mean_Cd", "St_peak"]
        for target in targets:
            if target in metrics:
                row[f"{target} R2"] = metrics[target]['R2']
                row[f"{target} RMSE"] = metrics[target]['RMSE']
            else:
                row[f"{target} R2"] = None
                row[f"{target} RMSE"] = None
                
        results.append(row)
        
    df = pd.DataFrame(results)
    print("\nComparison on 'Shorter' Shape Holdout (Design Velocity):")
    print(df.to_string(index=False))
    
    # Save
    df.to_csv("experiments/comparison_shorter.csv", index=False)
    print("\nSaved to experiments/comparison_shorter.csv")
    
if __name__ == "__main__":
    main()
