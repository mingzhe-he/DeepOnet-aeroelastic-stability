import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

class SheddingRegimeModel:
    """
    Regime-aware Strouhal number predictor.
    
    1. Identifies regimes using K-Means on (Features + St).
    2. Trains a classifier to predict regime from Features.
    3. Trains a regressor per regime to predict St from Features.
    """
    def __init__(self, n_regimes=2, classifier=None, regressor_type="gp", random_state=42):
        self.n_regimes = n_regimes
        self.random_state = random_state
        
        if classifier is None:
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=random_state)
        else:
            self.classifier = classifier
            
        self.regressor_type = regressor_type
        self.regressors = {}
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_regimes, random_state=random_state, n_init=10)
        
    def fit(self, X, y):
        """
        Fit the model.
        
        Args:
            X: Input features (n_samples, n_features)
            y: Strouhal number targets
        """
        # Scale features for clustering
        X_scaled = self.scaler.fit_transform(X)
        
        # Cluster in (X, y) space to find regimes
        # We weight y to ensure it influences clustering
        y_scaled = (y - y.mean()) / y.std()
        data_for_clustering = np.column_stack([X_scaled, y_scaled.reshape(-1, 1)])
        
        self.kmeans.fit(data_for_clustering)
        labels = self.kmeans.labels_
        
        # Train classifier: X -> Regime
        self.classifier.fit(X, labels)
        
        # Train per-regime regressors
        for r in range(self.n_regimes):
            mask = labels == r
            if np.sum(mask) < 2:
                # Not enough data, use global fallback or skip
                continue
                
            X_r = X[mask]
            y_r = y[mask]
            
            if self.regressor_type == "gp":
                kernel = C(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-5)
                reg = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True, random_state=self.random_state)
            elif self.regressor_type == "ridge":
                reg = Ridge(alpha=1.0)
            else:
                raise ValueError(f"Unknown regressor type: {self.regressor_type}")
                
            reg.fit(X_r, y_r)
            self.regressors[r] = reg
            
        return self
        
    def predict(self, X):
        """
        Predict Strouhal number.
        """
        # Predict regime
        regimes = self.classifier.predict(X)
        
        y_pred = np.zeros(X.shape[0])
        
        for r in range(self.n_regimes):
            mask = regimes == r
            if np.sum(mask) > 0:
                if r in self.regressors:
                    y_pred[mask] = self.regressors[r].predict(X[mask])
                else:
                    # Fallback if regime has no regressor (shouldn't happen if trained well)
                    y_pred[mask] = np.nan 
                    
        return y_pred
        
    def from_dataframe(self, df, feature_cols, target_col="St_peak"):
        """
        Fit from DataFrame.
        """
        X = df[feature_cols].values
        y = df[target_col].values
        self.fit(X, y)
        return self
