"""
Feature extraction utilities for spectral descriptors.

This module contains functions to:
- Compute power spectral density (PSD) using Welch's method
- Extract spectral descriptors (peak Strouhal, amplitude, quality factor)
- Non-dimensionalize inputs and outputs
"""

from typing import Dict, Tuple
import numpy as np
from scipy.signal import welch, find_peaks


def compute_welch_psd(
    signal: np.ndarray,
    fs: float,
    nperseg: int = None,
    detrend: str = 'linear'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute power spectral density using Welch's method.
    
    Args:
        signal: Time series signal
        fs: Sampling frequency (Hz)
        nperseg: Length of each segment (default: len(signal)//8, max 8192)
        detrend: Detrending method ('linear', 'constant', or False)
        
    Returns:
        Tuple of (frequencies, PSD)
    """
    if nperseg is None:
        nperseg = min(len(signal) // 8, 8192)
    
    freq, psd = welch(
        signal,
        fs=fs,
        nperseg=nperseg,
        detrend=detrend,
        scaling='density'
    )
    
    return freq, psd


def compute_strouhal_from_freq(freq: np.ndarray, D: float, U: float) -> np.ndarray:
    """
    Convert frequency to Strouhal number.
    
    St = f * D / U
    
    Args:
        freq: Frequency array (Hz)
        D: Characteristic dimension (m)
        U: Reference velocity (m/s)
        
    Returns:
        Strouhal number array
    """
    return freq * D / U


def extract_spectral_descriptors(
    freq: np.ndarray,
    psd: np.ndarray,
    D: float,
    U: float,
    min_st: float = 0.01,
    max_st: float = 1.0,
    prominence_factor: float = 0.1
) -> Dict[str, float]:
    """
    Extract spectral descriptors from PSD.
    
    Extracts:
    - St_peak: Dominant Strouhal number
    - A_peak: PSD amplitude at peak
    - Q: Quality factor (peak width measure)
    - bandwidth: Frequency bandwidth at half-max
    
    Args:
        freq: Frequency array (Hz)
        psd: Power spectral density
        D: Characteristic dimension (m)
        U: Reference velocity (m/s)
        min_st: Minimum Strouhal number to consider
        max_st: Maximum Strouhal number to consider
        prominence_factor: Relative prominence for peak detection
        
    Returns:
        Dictionary with spectral descriptors
    """
    # Convert to Strouhal number
    strouhal = compute_strouhal_from_freq(freq, D, U)
    
    # Restrict to physical range
    mask = (strouhal >= min_st) & (strouhal <= max_st)
    st_range = strouhal[mask]
    psd_range = psd[mask]
    
    if len(st_range) == 0:
        return {
            'St_peak': 0.0,
            'A_peak': 0.0,
            'Q': 0.0,
            'bandwidth': 0.0,
            'freq_peak': 0.0,
        }
    
    # Find peaks
    prominence = prominence_factor * psd_range.max()
    peaks, properties = find_peaks(psd_range, prominence=prominence)
    
    if len(peaks) == 0:
        # No clear peak, use maximum
        peak_idx = psd_range.argmax()
        st_peak = st_range[peak_idx]
        a_peak = psd_range[peak_idx]
        q_factor = 0.0
        bandwidth = 0.0
    else:
        # Use highest peak
        highest_peak_idx = peaks[psd_range[peaks].argmax()]
        st_peak = st_range[highest_peak_idx]
        a_peak = psd_range[highest_peak_idx]
        
        # Estimate quality factor and bandwidth
        # Q ≈ f_peak / Δf, where Δf is full-width at half-maximum
        half_max = a_peak / 2.0
        
        # Find indices where PSD crosses half-maximum
        above_half = psd_range > half_max
        if above_half.sum() > 1:
            # Find contiguous regions above half-max
            diff = np.diff(above_half.astype(int))
            rising_edges = np.where(diff == 1)[0]
            falling_edges = np.where(diff == -1)[0]
            
            # Find the region containing the peak
            for i, (rise, fall) in enumerate(zip(rising_edges, falling_edges)):
                if rise <= highest_peak_idx <= fall:
                    st_low = st_range[rise]
                    st_high = st_range[fall]
                    bandwidth = st_high - st_low
                    if bandwidth > 0:
                        q_factor = st_peak / bandwidth
                    else:
                        q_factor = 0.0
                    break
            else:
                q_factor = 0.0
                bandwidth = 0.0
        else:
            q_factor = 0.0
            bandwidth = 0.0
    
    return {
        'St_peak': float(st_peak),
        'A_peak': float(a_peak),
        'Q': float(q_factor),
        'bandwidth': float(bandwidth),
        'freq_peak': float(st_peak * U / D),  # Convert back to Hz
    }


def extract_cl_spectral_features(
    df,
    D: float,
    U: float,
    settling_time: float = 80.0,
    fs: float = 4000.0
) -> Dict[str, float]:
    """
    Extract spectral features from lift coefficient time series.
    
    Args:
        df: DataFrame with time series (must have 't' and 'Cl' columns)
        D: Characteristic dimension (m)
        U: Reference velocity (m/s)
        settling_time: Time to exclude for settling (s)
        fs: Sampling frequency (Hz), default 4000 Hz
        
    Returns:
        Dictionary with spectral descriptors
    """
    # Extract stable portion
    mask = df['t'] > settling_time
    cl_stable = df[mask]['Cl'].values
    
    if len(cl_stable) == 0:
        raise ValueError(f"No data after settling time {settling_time}s")
    
    # Compute PSD
    freq, psd = compute_welch_psd(cl_stable, fs=fs)
    
    # Extract descriptors
    descriptors = extract_spectral_descriptors(freq, psd, D, U)
    
    return descriptors


def create_nondimensional_features(case_data: Dict) -> Dict[str, float]:
    """
    Create non-dimensional input features for ML models.
    
    Features include:
    - D: Characteristic dimension
    - H: Height
    - H/D: Aspect ratio
    - Re: Reynolds number
    - sin(α), cos(α): Angle of attack (periodic encoding)
    - α_rad: Angle of attack in radians (alternative)
    
    Args:
        case_data: Processed case data
        
    Returns:
        Dictionary with non-dimensional features
    """
    aoa_rad = case_data['aoa_rad']
    
    return {
        'D': case_data['D'],
        'H': case_data['H'],
        'H_over_D': case_data['H_over_D'],
        'Re': case_data['Re'],
        'aoa_rad': aoa_rad,
        'aoa_deg': case_data['aoa'],
        'sin_aoa': np.sin(aoa_rad),
        'cos_aoa': np.cos(aoa_rad),
        'U_ref': case_data['U_ref'],
    }
