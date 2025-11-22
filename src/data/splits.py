"""
Data splitting utilities for case-level train/val/test splits.

This module ensures no data leakage by splitting at the case level
(geometry + AoA combination), not at the frequency or time sample level.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def create_shape_holdout_split(
    cases_df: pd.DataFrame,
    holdout_shape: str,
    val_fraction: float = 0.15,
    random_seed: int = 42
) -> Dict[str, List[int]]:
    """
    Create train/val/test split by holding out one shape entirely.
    
    Args:
        cases_df: DataFrame with all cases (must have 'shape_type' column)
        holdout_shape: Shape to hold out for testing
        val_fraction: Fraction of training data to use for validation
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with 'train', 'val', 'test' indices
    """
    np.random.seed(random_seed)
    
    # Test set: all cases from holdout shape
    test_mask = cases_df['shape_type'] == holdout_shape
    test_indices = cases_df[test_mask].index.tolist()
    
    # Train + val: remaining shapes
    train_val_indices = cases_df[~test_mask].index.tolist()
    
    # Split train/val randomly
    n_val = int(len(train_val_indices) * val_fraction)
    shuffled = np.random.permutation(train_val_indices)
    
    val_indices = shuffled[:n_val].tolist()
    train_indices = shuffled[n_val:].tolist()
    
    return {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices,
        'metadata': {
            'split_type': 'shape_holdout',
            'holdout_shape': holdout_shape,
            'val_fraction': val_fraction,
            'random_seed': random_seed,
        }
    }


def create_aoa_interpolation_split(
    cases_df: pd.DataFrame,
    test_aoas: List[int],
    val_fraction: float = 0.15,
    random_seed: int = 42
) -> Dict[str, List[int]]:
    """
    Create train/val/test split by holding out specific AoA values.
    
    This tests interpolation capability: train on some angles,
    test on intermediate angles.
    
    Args:
        cases_df: DataFrame with all cases (must have 'aoa' column)
        test_aoas: List of AoA values to hold out for testing
        val_fraction: Fraction of training data for validation
        random_seed: Random seed
        
    Returns:
        Dictionary with 'train', 'val', 'test' indices
    """
    np.random.seed(random_seed)
    
    # Test set: specified AoA values across all shapes
    test_mask = cases_df['aoa'].isin(test_aoas)
    test_indices = cases_df[test_mask].index.tolist()
    
    # Train + val: other AoA values
    train_val_indices = cases_df[~test_mask].index.tolist()
    
    # Split train/val
    n_val = int(len(train_val_indices) * val_fraction)
    shuffled = np.random.permutation(train_val_indices)
    
    val_indices = shuffled[:n_val].tolist()
    train_indices = shuffled[n_val:].tolist()
    
    return {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices,
        'metadata': {
            'split_type': 'aoa_interpolation',
            'test_aoas': test_aoas,
            'val_fraction': val_fraction,
            'random_seed': random_seed,
        }
    }


def create_kfold_splits(
    cases_df: pd.DataFrame,
    n_folds: int = 5,
    random_seed: int = 42
) -> List[Dict[str, List[int]]]:
    """
    Create K-fold cross-validation splits at case level.
    
    Args:
        cases_df: DataFrame with all cases
        n_folds: Number of folds
        random_seed: Random seed
        
    Returns:
        List of split dictionaries, one per fold
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    
    indices = cases_df.index.to_numpy()
    splits = []
    
    for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(indices)):
        train_val_indices = indices[train_val_idx].tolist()
        test_indices = indices[test_idx].tolist()
        
        # Further split train_val into train and val (80/20)
        n_val = len(train_val_indices) // 5
        val_indices = train_val_indices[:n_val]
        train_indices = train_val_indices[n_val:]
        
        splits.append({
            'train': train_indices,
            'val': val_indices,
            'test': test_indices,
            'metadata': {
                'split_type': 'kfold',
                'fold': fold_idx,
                'n_folds': n_folds,
                'random_seed': random_seed,
            }
        })
    
    return splits


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def save_split(split: Dict, output_path: Path) -> None:
    """
    Save split dictionary to JSON file.
    
    Args:
        split: Split dictionary
        output_path: Path to save JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(split, f, indent=2, cls=NumpyEncoder)
    
    print(f"Saved split to {output_path}")


def load_split(split_path: Path) -> Dict:
    """
    Load split from JSON file.
    
    Args:
        split_path: Path to split JSON file
        
    Returns:
        Split dictionary
    """
    with open(split_path, 'r') as f:
        split = json.load(f)
    
    return split


def validate_split(split: Dict, n_total_cases: int) -> bool:
    """
    Validate that split has no leakage and covers all cases exactly once.
    
    Args:
        split: Split dictionary
        n_total_cases: Total number of cases
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    train = set(split['train'])
    val = set(split['val'])
    test = set(split['test'])
    
    # Check no overlap
    if train & val:
        raise ValueError(f"Train/val overlap: {train & val}")
    if train & test:
        raise ValueError(f"Train/test overlap: {train & test}")
    if val & test:
        raise ValueError(f"Val/test overlap: {val & test}")
    
    # Check all cases covered
    all_indices = train | val | test
    if len(all_indices) != n_total_cases:
        raise ValueError(
            f"Split covers {len(all_indices)} cases, but {n_total_cases} expected"
        )
    
    print(f"✓ Split validation passed: {len(train)} train, {len(val)} val, {len(test)} test")
    return True


def print_split_summary(split: Dict, cases_df: pd.DataFrame) -> None:
    """
    Print human-readable summary of a split.
    
    Args:
        split: Split dictionary
        cases_df: DataFrame with all cases
    """
    print("\n" + "="*60)
    print(f"Split type: {split['metadata']['split_type']}")
    print("="*60)
    
    for subset in ['train', 'val', 'test']:
        indices = split[subset]
        subset_df = cases_df.loc[indices]
        
        print(f"\n{subset.upper()} ({len(indices)} cases):")
        print(f"  Shapes: {subset_df['shape_type'].value_counts().to_dict()}")
        print(f"  AoA range: {subset_df['aoa'].min()}° to {subset_df['aoa'].max()}°")
        if 'Re' in subset_df.columns:
            print(f"  Re range: {subset_df['Re'].min():.1e} to {subset_df['Re'].max():.1e}")
