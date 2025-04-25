#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
data_to_csv.py - Process EDF Data and Export Metrics to CSV

This module loads an EDF file, segments the data into fixed-length epochs,
computes basic metrics (mean power in Delta, Theta, Alpha, SMR, Beta, HighBeta)
for each channel in each epoch, and exports the results into a CSV file.
Each row corresponds to one epoch for one channel.

Usage:
    python data_to_csv.py --edf path/to/file.edf --epoch_length 2.0 --output output.csv

Ensure this file is saved with UTF-8 encoding.
"""

import argparse
import numpy as np
import pandas as pd
import mne
from modules import io_utils

# Define frequency bands
BANDS = {
    "Delta": (1, 4),
    "Theta": (4, 8),
    "Alpha": (8, 12),
    "SMR": (12, 15),
    "Beta": (15, 27),
    "HighBeta": (28, 38)
}


def compute_band_power(data, sfreq, band):
    """
    Compute the mean power in the given frequency band.

    Parameters:
        data : 1D numpy array
            Signal data.
        sfreq : float
            Sampling frequency.
        band : tuple (fmin, fmax)
            Frequency band limits.

    Returns:
        float: Mean power in the band.
    """
    fmin, fmax = band
    # Apply bandpass filtering using MNE's filter function
    filtered = mne.filter.filter_data(data, sfreq, fmin, fmax, verbose=False)
    power = np.mean(filtered ** 2)
    return power


def process_edf_to_csv(edf_path, epoch_length, output_csv):
    """
    Load an EDF file, segment the data into epochs of length epoch_length (in seconds),
    compute band power for each channel in each epoch, and write the results to a CSV file.

    Parameters:
        edf_path : str
            Path to the EDF file.
        epoch_length : float
            Duration (in seconds) for each epoch.
        output_csv : str
            Output CSV file path.
    """
    # Use io_utils.load_eeg_data to load and preprocess the EDF data.
    # Here, we do not need to apply CSD (use_csd=False) and we want to apply the high-pass filter.
    raw = io_utils.load_eeg_data(edf_path, use_csd=False, for_source=False, apply_filter=True)

    # Get sampling frequency
    sfreq = raw.info["sfreq"]

    # Create fixed-length events and epochs
    events = mne.make_fixed_length_events(raw, duration=epoch_length, verbose=False)
    epochs = mne.Epochs(raw, events, tmin=0, tmax=epoch_length, baseline=None, preload=True, verbose=False)

    rows = []
    # Loop over epochs
    for i, epoch in enumerate(epochs.get_data()):
        epoch_start = events[i, 0] / sfreq
        epoch_end = epoch_start + epoch_length
        # Loop over channels
        for ch_idx, ch in enumerate(epochs.ch_names):
            row = {
                "Channel": ch,
                "Epoch": i,
                "Start_Time": epoch_start,
                "End_Time": epoch_end
            }
            # Compute band power for each defined band
            for band_name, band_range in BANDS.items():
                power = compute_band_power(epoch[ch_idx, :], sfreq, band_range)
                row[band_name] = power
            rows.append(row)

    # Create a DataFrame and export to CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"CSV file saved to {output_csv}")


# --- New function to save pre-computed features --- 
def save_computed_features_to_csv(features: dict, info: mne.Info | None, output_csv: str) -> None:
    """Saves a dictionary of pre-computed features to a CSV file.

    Assumes features dictionary might contain nested structures like band powers per channel.
    Flattens the structure for CSV output.

    Args:
        features (dict): Dictionary containing computed features. 
                         Expected keys like 'band_powers_eo', 'zscores_eo', etc.
                         Values under these keys might be dicts mapping channel to band power dict,
                         or band to list of zscores.
        info (mne.Info | None): MNE Info object to get channel names if needed.
        output_csv (str): Path to the output CSV file.
    """
    rows = []
    ch_names = info.ch_names if info else []

    # --- Process Band Powers (Example: EO) ---
    bp_eo = features.get("band_powers_eo")
    if bp_eo and isinstance(bp_eo, dict):
        for ch, band_data in bp_eo.items():
            if isinstance(band_data, dict):
                 row = {"Subject_Feature": f"{ch}_EO_Power"}
                 row.update(band_data) # Add {band: power} pairs
                 rows.append(row)

    # --- Process Band Powers (Example: EC) ---
    bp_ec = features.get("band_powers_ec")
    if bp_ec and isinstance(bp_ec, dict):
        for ch, band_data in bp_ec.items():
             if isinstance(band_data, dict):
                 row = {"Subject_Feature": f"{ch}_EC_Power"}
                 row.update(band_data)
                 rows.append(row)

    # --- Process Z-Scores (Example: EO) ---
    zscores_eo = features.get("zscores_eo")
    if zscores_eo and isinstance(zscores_eo, dict):
        for band, z_list in zscores_eo.items():
            if isinstance(z_list, list) and len(z_list) == len(ch_names):
                 # Save z-scores per channel for this band
                 for i, ch in enumerate(ch_names):
                      row = {"Subject_Feature": f"{ch}_EO_ZScore_{band}", "Value": z_list[i]}
                      rows.append(row)
            else:
                 # Save average z-score if list format doesn't match channels
                 try:
                     avg_z = np.nanmean(z_list) if isinstance(z_list, list) else np.nan
                     row = {"Subject_Feature": f"Avg_EO_ZScore_{band}", "Value": avg_z}
                     rows.append(row)
                 except Exception:
                     pass # Ignore errors if averaging fails

    # --- Process Z-Scores (Example: EC) ---
    # (Add similar logic for zscores_ec if needed)
    
    # --- Add other features as needed --- 
    # Example: Single value features
    # if features.get('some_single_value'):
    #    rows.append({"Subject_Feature": "Some_Single_Value", "Value": features['some_single_value']})

    if not rows:
        print(f"No features processed for CSV export to {output_csv}. Skipping file creation.")
        return

    try:
        df = pd.DataFrame(rows)
        # Reorder columns nicely if possible
        cols = df.columns.tolist()
        if "Subject_Feature" in cols:
            cols.insert(0, cols.pop(cols.index("Subject_Feature")))
        if "Value" in cols: # For single value features
             cols.insert(1, cols.pop(cols.index("Value")))
        # Put band columns after Value if they exist
        band_cols = [b for b in BANDS if b in cols]
        for band in reversed(band_cols):
            if band in cols: # Check again as it might have been moved
                cols.insert(2, cols.pop(cols.index(band)))
                
        df = df[cols]
        df.to_csv(output_csv, index=False, na_rep='NaN')
        print(f"Computed features saved to {output_csv}")
    except Exception as e:
        print(f"Error saving computed features to CSV {output_csv}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Process an EDF file and export channel metrics per epoch to CSV"
    )
    parser.add_argument("--edf", required=True, help="Path to the EDF file")
    parser.add_argument("--epoch_length", type=float, default=2.0, help="Epoch length in seconds (default: 2.0)")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    args = parser.parse_args()

    process_edf_to_csv(args.edf, args.epoch_length, args.output)


if __name__ == "__main__":
    main()
