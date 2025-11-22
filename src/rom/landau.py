import numpy as np

def make_time_array(duration, fs):
    """
    Create time array.
    """
    return np.arange(0, duration, 1.0/fs)

def reconstruct_cl_time_series(t, mean_cl, A_rms, St, U, D, phase=None):
    """
    Reconstruct Cl(t) using Landau limit cycle approximation.
    
    Cl(t) = mean_cl + A_rms * sqrt(2) * cos(2*pi*f*t + phi)
    
    where f = St * U / D
    
    Args:
        t: Time array
        mean_cl: Mean lift coefficient
        A_rms: RMS amplitude of fluctuations (std_Cl)
        St: Strouhal number
        U: Velocity
        D: Diameter
        phase: Phase angle (radians). If None, random phase is used.
        
    Returns:
        Cl time series
    """
    f = St * U / D
    
    if phase is None:
        phase = np.random.uniform(0, 2*np.pi)
        
    # A_rms is standard deviation.
    # For a pure sine wave A*sin(wt), RMS = A / sqrt(2) => A = RMS * sqrt(2)
    amplitude = A_rms * np.sqrt(2)
    
    cl_t = mean_cl + amplitude * np.cos(2 * np.pi * f * t + phase)
    
    return cl_t

def reconstruct_cd_time_series(t, mean_cd, A_rms_cd, St, U, D, phase=None):
    """
    Reconstruct Cd(t). Cd usually oscillates at 2*f_shedding.
    """
    f = St * U / D
    
    if phase is None:
        phase = np.random.uniform(0, 2*np.pi)
        
    amplitude = A_rms_cd * np.sqrt(2)
    
    # Drag oscillates at 2*f
    cd_t = mean_cd + amplitude * np.cos(2 * np.pi * (2*f) * t + phase)
    
    return cd_t
