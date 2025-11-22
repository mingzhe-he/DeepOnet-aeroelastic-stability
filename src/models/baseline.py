"""
Baseline models for comparison.

Implements simple baselines:
- Ridge regression
- Linear regression
"""

import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional


class RidgeBaseline:
    """
    Ridge regression baseline for scalar prediction.
    
    Trains separate ridge models for each output target.
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Args:
            alpha: Regularization strength
        """
        self.alpha = alpha
        self.models = {}
        self.scaler_X = StandardScaler()
        self.scaler_y = {}
        self.target_names = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, target_names: list):
        """
        Fit ridge models for each output.
        
        Args:
            X: Input features (n_samples, n_features)
            y: Output targets (n_samples, n_targets)
            target_names: List of target names
        """
        self.target_names = target_names
        
        # Normalize inputs
        X_scaled = self.scaler_X.fit_transform(X)
        
        # Train separate model for each target
        for i, name in enumerate(target_names):
            # Normalize output
            scaler_y = StandardScaler()
            y_scaled = scaler_y.fit_transform(y[:, i:i+1]).ravel()
            self.scaler_y[name] = scaler_y
            
            # Fit ridge model
            model = Ridge(alpha=self.alpha)
            model.fit(X_scaled, y_scaled)
            self.models[name] = model
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Predictions (n_samples, n_targets)
        """
        X_scaled = self.scaler_X.transform(X)
        
        predictions = []
        for name in self.target_names:
            y_pred_scaled = self.models[name].predict(X_scaled)
            y_pred = self.scaler_y[name].inverse_transform(
                y_pred_scaled.reshape(-1, 1)
            ).ravel()
            predictions.append(y_pred)
        
        return np.column_stack(predictions)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Compute R² score for each target.
        
        Args:
            X: Input features
            y: True targets
            
        Returns:
            Dictionary of R² scores
        """
        X_scaled = self.scaler_X.transform(X)
        
        scores = {}
        for i, name in enumerate(self.target_names):
            y_true_scaled = self.scaler_y[name].transform(y[:, i:i+1]).ravel()
            score = self.models[name].score(X_scaled, y_true_scaled)
            scores[name] = score
        
        return scores


class LinearBaseline:
    """
    Simple linear regression baseline (no regularization).
    """
    
    def __init__(self):
        self.model = LinearRegression()
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.target_names = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, target_names: list):
        """
        Fit linear regression model.
        
        Args:
            X: Input features (n_samples, n_features)
            y: Output targets (n_samples, n_targets)
            target_names: List of target names
        """
        self.target_names = target_names
        
        # Normalize
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        # Fit
        self.model.fit(X_scaled, y_scaled)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Predictions (n_samples, n_targets)
        """
        X_scaled = self.scaler_X.transform(X)
        y_pred_scaled = self.model.predict(X_scaled)
        return self.scaler_y.inverse_transform(y_pred_scaled)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute R² score.
        
        Args:
            X: Input features
            y: True targets
            
        Returns:
            Overall R² score
        """
        X_scaled = self.scaler_X.transform(X)
        y_scaled = self.scaler_y.transform(y)
        return self.model.score(X_scaled, y_scaled)
