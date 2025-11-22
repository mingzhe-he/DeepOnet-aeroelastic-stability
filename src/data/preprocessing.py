import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
from scipy import signal
from pathlib import Path
from tqdm import tqdm

# Constants
SHAPE_PARAMS = {
    "baseline": {"H": 1.5, "D": 3.0},
    "shorter": {"H": 1.0, "D": 3.0},
    "taller": {"H": 2.0, "D": 3.0},
    "higher": {"H": 2.0, "D": 3.0},  # Alias for taller
}

DEFAULT_DESIGN_U = 21.5
DEFAULT_LREF = 1.5
DEFAULT_AREF = 1.5**2  # 2.25
NU_AIR = 1.5e-5  # Kinematic viscosity of air approx

def parse_force_coeffs_header(filepath):
    """
    Parses the header of a forceCoeffs.dat file to extract reference values.
    """
    metadata = {}
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'):
                if 'lRef' in line:
                    try:
                        metadata['lRef'] = float(line.split(':')[1].strip().split()[0])
                    except:
                        pass
                if 'Aref' in line:
                    try:
                        metadata['Aref'] = float(line.split(':')[1].strip().split()[0])
                    except:
                        pass
                if 'magUInf' in line:
                    try:
                        metadata['magUInf'] = float(line.split(':')[1].strip().split()[0])
                    except:
                        pass
            else:
                break
    return metadata

def rescale_force_coefficients(df, metadata, shape_type, U_sim, design_U_default=21.5):
    """
    Rescales force coefficients to account for correct simulation velocity and shape depth.
    
    Original OpenFOAM normalisation often used fixed U_design and A_ref.
    We want coefficients based on actual U_sim and A_true = H * 1.0 (2D).
    
    Correction factor:
    C_corrected = C_raw * (U_design^2 * A_file) / (U_sim^2 * A_true)
    """
    # Get shape parameters
    if shape_type not in SHAPE_PARAMS:
        raise ValueError(f"Unknown shape type: {shape_type}")
    
    H = SHAPE_PARAMS[shape_type]["H"]
    # For 2D simulations, A_true is often H * 1.0 (unit depth) or similar.
    # However, the OpenFOAM Aref was likely H_baseline^2 = 1.5^2 = 2.25 or similar.
    # Let's assume the standard definition C = F / (0.5 * rho * U^2 * A)
    # The file contains C_file = F / (0.5 * rho * U_design^2 * A_file)
    # We want C_true = F / (0.5 * rho * U_sim^2 * A_true)
    # So C_true = C_file * (U_design^2 * A_file) / (U_sim^2 * A_true)
    
    # Extract file metadata or use defaults
    U_design = metadata.get('magUInf', design_U_default)
    # If magUInf is 0 or missing, assume it was normalized with the default design speed
    if U_design == 0: 
        U_design = design_U_default
        
    A_file = metadata.get('Aref', DEFAULT_AREF)
    lRef_file = metadata.get('lRef', DEFAULT_LREF)
    
    # True reference area for the shape (projected area)
    # For a prism of depth H and unit span, A = H * 1.0. 
    # BUT, check if OpenFOAM used Aref = lRef^2. 
    # Let's stick to the logic: A_true should be H * 1.0 if we want per-unit-span coeffs,
    # or if we want to match the "baseline" definition where A = 1.5 * 1.5, maybe we should use A_true = H * H?
    # The prompt says: A_true = H^2. Let's follow that.
    A_true = H**2
    
    # Velocity scaling
    # If U_sim is different from U_design, we need to correct.
    # Factor = (U_design / U_sim)^2
    vel_factor = (U_design / U_sim)**2
    
    # Area scaling
    area_factor = A_file / A_true
    
    total_factor = vel_factor * area_factor
    
    # Apply to Cl, Cd
    df['Cl'] = df['Cl'] * total_factor
    df['Cd'] = df['Cd'] * total_factor
    
    # For Cm, we also need to correct for reference length if it changed
    # Cm_file = M / (0.5 * rho * U_design^2 * A_file * lRef_file)
    # Cm_true = M / (0.5 * rho * U_sim^2 * A_true * H)  (assuming H is the characteristic length for Cm)
    # Factor_Cm = total_factor * (lRef_file / H)
    
    cm_factor = total_factor * (lRef_file / H)
    df['Cm'] = df['Cm'] * cm_factor
    
    return df

def compute_spectral_features(time, cl, settling_time=80.0):
    """
    Computes spectral features (St, A_peak, Q, etc.) from Cl time series.
    """
    # Filter stable region
    mask = time > settling_time
    if not np.any(mask):
        return {}
    
    t_stable = time[mask]
    cl_stable = cl[mask]
    
    # Detrend/Center
    cl_prime = cl_stable - np.mean(cl_stable)
    
    # Sampling parameters
    dt = np.mean(np.diff(t_stable))
    fs = 1.0 / dt
    
    # Welch's PSD
    freqs, psd = signal.welch(cl_prime, fs, nperseg=min(len(cl_prime), 4096))
    
    # Find peak (ignore low freq)
    valid_idx = freqs > 0.05
    if not np.any(valid_idx):
        return {}
    
    f_valid = freqs[valid_idx]
    p_valid = psd[valid_idx]
    
    peak_idx = np.argmax(p_valid)
    freq_peak = f_valid[peak_idx]
    psd_peak = p_valid[peak_idx]
    
    # Q factor (bandwidth)
    # Find half-power points
    half_power = psd_peak / 2.0
    # Simple search around peak
    # This is rough, can be improved
    
    return {
        "freq_peak": freq_peak,
        "psd_peak": psd_peak,
        "bandwidth": 0.0, # Placeholder
        "Q": 0.0 # Placeholder
    }

def process_case(case_path, settling_time=80.0):
    """
    Process a single simulation case.
    """
    path_obj = Path(case_path)
    # Expected structure: .../{shape_variant}/{AoA}/postProcessing/cylinder/0/forceCoeffs.dat
    
    try:
        aoa_dir = path_obj.parent.parent.parent.name
        shape_dir = path_obj.parent.parent.parent.parent.name
        
        # Parse AoA
        try:
            aoa = float(aoa_dir)
        except ValueError:
            # Handle cases like "angle0" or similar if needed
            aoa = float(re.findall(r"[-+]?\d*\.\d+|\d+", aoa_dir)[0])
            
        # Parse Shape and U_sim
        # shape_dir might be "baseline", "baseline_lowU", "shorter", etc.
        if "lowU" in shape_dir:
            u_sim = 5.0
            shape_base = shape_dir.replace("_lowU", "")
        elif "mediumU" in shape_dir:
            u_sim = 10.0
            shape_base = shape_dir.replace("_mediumU", "")
        else:
            u_sim = 21.5
            shape_base = shape_dir
            
        # Normalize shape name
        if shape_base == "higher":
            shape_base = "taller"
            
        if shape_base not in SHAPE_PARAMS:
            # Fallback or skip
            # print(f"Skipping unknown shape: {shape_base}")
            return None

        # Load data
        # Skip header lines (usually start with #)
        # forceCoeffs.dat usually has columns: Time, Cm, Cd, Cl, Cl(f), Cl(r)
        # We need to handle the header carefully.
        
        metadata = parse_force_coeffs_header(case_path)
        
        # Read data, assuming standard OpenFOAM format
        # Use pandas read_csv with flexible whitespace separator
        df = pd.read_csv(case_path, sep=r"\s+", comment='#', header=None)
        # Assign columns based on standard OpenFOAM output
        # Time	Cm	Cd	Cl	Cl(f)	Cl(r)
        if df.shape[1] >= 4:
            df.columns = ['Time', 'Cm', 'Cd', 'Cl'] + [f'Col{i}' for i in range(4, df.shape[1])]
        else:
            return None
            
        # Rescale
        df = rescale_force_coefficients(df, metadata, shape_base, u_sim)
        
        # Compute stats
        mask = df['Time'] > settling_time
        if not np.any(mask):
            return None
            
        df_stable = df[mask]
        
        mean_cl = df_stable['Cl'].mean()
        mean_cd = df_stable['Cd'].mean()
        mean_cm = df_stable['Cm'].mean()
        std_cl = df_stable['Cl'].std()
        std_cd = df_stable['Cd'].std()
        std_cm = df_stable['Cm'].std()
        
        # Spectral
        spec_feats = compute_spectral_features(df['Time'].values, df['Cl'].values, settling_time)
        
        # Geometry
        D = SHAPE_PARAMS[shape_base]["D"]
        H = SHAPE_PARAMS[shape_base]["H"]
        
        # Strouhal
        st_peak = spec_feats.get("freq_peak", np.nan) * D / u_sim
        
        # Re
        re_num = u_sim * D / NU_AIR
        
        return {
            "case_path": str(case_path),
            "shape_type": shape_base,
            "shape_variant": shape_dir, # Keep original folder name
            "aoa": aoa,
            "aoa_rad": np.radians(aoa),
            "D": D,
            "H": H,
            "H_over_D": H/D,
            "U_ref": u_sim,
            "Re": re_num,
            "mean_Cl": mean_cl,
            "mean_Cd": mean_cd,
            "mean_Cm": mean_cm,
            "std_Cl": std_cl,
            "std_Cd": std_cd,
            "std_Cm": std_cm,
            "St_peak": st_peak,
            "freq_peak": spec_feats.get("freq_peak", np.nan),
            "psd_peak": spec_feats.get("psd_peak", np.nan),
            # Derived
            "sin_aoa": np.sin(np.radians(aoa)),
            "cos_aoa": np.cos(np.radians(aoa)),
        }
        
    except Exception as e:
        print(f"Error processing {case_path}: {e}")
        return None

def discover_cases(base_path):
    """
    Recursively find all forceCoeffs.dat files.
    """
    # Pattern: base_path/**/forceCoeffs.dat
    # Use glob
    search_pattern = os.path.join(base_path, "**", "forceCoeffs.dat")
    files = glob.glob(search_pattern, recursive=True)
    return files

def build_summary_dataframe(base_path, settling_time=80.0):
    files = discover_cases(base_path)
    print(f"Found {len(files)} case files.")
    
    results = []
    for f in tqdm(files):
        res = process_case(f, settling_time)
        if res:
            results.append(res)
            
    df = pd.DataFrame(results)
    return df

def main():
    parser = argparse.ArgumentParser(description="Preprocess OpenFOAM data for Tier 1.")
    parser.add_argument("--raw-root", type=str, required=True, help="Path to raw data root")
    parser.add_argument("--output", type=str, required=True, help="Path to output parquet file")
    parser.add_argument("--settling-time", type=float, default=80.0, help="Settling time in seconds")
    
    args = parser.parse_args()
    
    print(f"Scanning {args.raw_root}...")
    df = build_summary_dataframe(args.raw_root, args.settling_time)
    
    print(f"Processed {len(df)} cases.")
    
    # Ensure output dir exists
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(out_path)
    print(f"Saved summary to {out_path}")

if __name__ == "__main__":
    main()
