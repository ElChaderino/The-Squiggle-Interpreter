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
import mne
import pandas as pd

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
    # Load EDF data
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    # Set a standard montage and average reference
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, match_case=False)
    raw.set_eeg_reference("average", projection=True)
    
    sfreq = raw.info["sfreq"]
    
    # Create fixed-length epochs
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
    
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"CSV file saved to {output_csv}")

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
