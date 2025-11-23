import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler

class GallopingGPROM:
    """
    Gaussian Process ROM for mean force coefficients and Den Hartog criterion.
    """
    def __init__(self, kernel=None, n_restarts_optimizer=5, random_state=42):
        self.kernel_template = kernel
        self.n_restarts_optimizer = n_restarts_optimizer
        self.random_state = random_state
        
        self.gp_cl = None
        self.gp_cd = None
        self.scaler = StandardScaler()
        self.feature_cols = None
        
    def fit(self, X, y_cl, y_cd):
        """
        Fit GPs for Cl and Cd.
        
        Args:
            X: Input features (n_samples, n_features)
            y_cl: Mean Cl targets
            y_cd: Mean Cd targets
        """
        # Standardize X
        X_scaled = self.scaler.fit_transform(X)
        
        # Create kernel with ARD if not provided
        if self.kernel_template is None:
            n_features = X.shape[1]
            # ARD kernel: separate length scale for each feature
            kernel = C(1.0, (1e-2, 1e2)) * RBF(length_scale=[1.0]*n_features, length_scale_bounds=(1e-2, 1e2)) \
                     + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1))
        else:
            kernel = self.kernel_template
            
        self.gp_cl = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=self.n_restarts_optimizer,
            normalize_y=True,
            random_state=self.random_state
        )
        self.gp_cd = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=self.n_restarts_optimizer,
            normalize_y=True,
            random_state=self.random_state
        )
        
        self.gp_cl.fit(X_scaled, y_cl)
        self.gp_cd.fit(X_scaled, y_cd)
        return self
        
    def from_dataframe(self, df, feature_cols, cl_col="mean_Cl", cd_col="mean_Cd"):
        """
        Fit from DataFrame.
        """
        self.feature_cols = feature_cols
        X = df[feature_cols].values
        y_cl = df[cl_col].values
        y_cd = df[cd_col].values
        
        self.fit(X, y_cl, y_cd)
        return self
        
    def predict(self, X, return_std=False):
        """
        Predict mean Cl and Cd.
        """
        X_scaled = self.scaler.transform(X)
        
        if return_std:
            cl_pred, cl_std = self.gp_cl.predict(X_scaled, return_std=True)
            cd_pred, cd_std = self.gp_cd.predict(X_scaled, return_std=True)
            return (cl_pred, cd_pred), (cl_std, cd_std)
        else:
            cl_pred = self.gp_cl.predict(X_scaled)
            cd_pred = self.gp_cd.predict(X_scaled)
            return cl_pred, cd_pred
            
    def estimate_derivative_and_den_hartog(self, X, aoa_idx, delta_deg=0.1):
        """
        Estimate dCl/dAlpha and Den Hartog criterion using finite differences on the GP.
        
        Args:
            X: Input features (n_samples, n_features)
            aoa_idx: Index of the AoA feature (in radians)
            delta_deg: Step size in degrees for finite difference
            
        Returns:
            dCl_dAlpha, S_DH
        """
        delta_rad = np.radians(delta_deg)
        
        # Perturb X
        X_plus = X.copy()
        X_minus = X.copy()
        
        X_plus[:, aoa_idx] += delta_rad
        X_minus[:, aoa_idx] -= delta_rad
        
        # Predict
        # Predict (internally handles scaling)
        cl_plus = self.predict(X_plus)[0]
        cl_minus = self.predict(X_minus)[0]
        
        # Central difference
        dCl_dAlpha = (cl_plus - cl_minus) / (2 * delta_rad)
        
        # Predict Cd at center
        _, cd_pred = self.predict(X)
        
        # Den Hartog
        S_DH = dCl_dAlpha + cd_pred
        
        return dCl_dAlpha, S_DH
