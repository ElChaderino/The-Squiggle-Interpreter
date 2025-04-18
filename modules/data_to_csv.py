#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
data_to_csv.py - Process EDF Data and Export Comprehensive Metrics to CSV

This module loads EDF files (for EO and/or EC conditions), segments the data into fixed-length epochs,
computes a wide range of metrics (frequency-domain, time-domain, asymmetry, and vigilance-related) for each
channel in each epoch, and exports the results into CSV files. It supports overlapping epochs, artifact detection,
and differential analysis between EO and EC conditions.

Enhanced Features:
- Frequency-domain metrics: Absolute and relative band power, peak frequency, total power.
- Time-domain metrics: Mean amplitude, standard deviation, skewness, kurtosis.
- Asymmetry metrics: Frontal alpha asymmetry (F4-F3).
- Vigilance metrics: Alpha reactivity (EO/EC ratio), vigilance state estimation.
- Artifact detection: Flags epochs with potential artifacts based on variance and amplitude.
- Output: Detailed CSV per condition, summary CSV per channel, and differential CSV (EO-EC).

Usage:
    python data_to_csv.py --eo path/to/eo.edf --ec path/to/ec.edf --epoch_length 2.0 --overlap 0.5 --output output_dir

Ensure this file is saved with UTF-8 encoding.
"""

import argparse
import numpy as np
import pandas as pd
import mne
from scipy.stats import skew, kurtosis
from pathlib import Path
import logging
from modules import io_utils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("data_to_csv.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
        data (np.array): 1D signal data.
        sfreq (float): Sampling frequency.
        band (tuple): Frequency band limits (fmin, fmax).

    Returns:
        tuple: (mean_power, peak_freq, rel_power, total_power)
            - mean_power (float): Mean power in the band.
            - peak_freq (float): Frequency with maximum power in the band.
            - rel_power (float): Relative power (band power / total power).
            - total_power (float): Total power across 1-40 Hz.
    """
    fmin, fmax = band
    try:
        # Compute total power (1-40 Hz)
        freqs, psd = signal.welch(data, fs=sfreq, nperseg=min(int(sfreq * 0.5), len(data)))
        total_mask = (freqs >= 1) & (freqs <= 40)
        total_power = np.sum(psd[total_mask]) if np.any(total_mask) else 1e-10

        # Compute band power
        band_mask = (freqs >= fmin) & (freqs <= fmax)
        if not np.any(band_mask):
            logger.warning(f"No frequencies in range {band} for band power computation.")
            return 0.0, 0.0, 0.0, total_power
        band_psd = psd[band_mask]
        mean_power = np.mean(band_psd)
        peak_idx = np.argmax(band_psd)
        peak_freq = freqs[band_mask][peak_idx] if len(band_psd) > 0 else fmin
        rel_power = mean_power / total_power if total_power > 0 else 0.0

        return mean_power, peak_freq, rel_power, total_power
    except Exception as e:
        logger.error(f"Failed to compute band power for band {band}: {e}")
        return 0.0, 0.0, 0.0, 1e-10

def compute_time_domain_metrics(data):
    """
    Compute time-domain metrics for the given signal.

    Parameters:
        data (np.array): 1D signal data.

    Returns:
        dict: Dictionary with metrics (mean_amplitude, std_amplitude, skewness, kurtosis).
    """
    try:
        return {
            "Mean_Amplitude": np.mean(data),
            "Std_Amplitude": np.std(data),
            "Skewness": skew(data),
            "Kurtosis": kurtosis(data)
        }
    except Exception as e:
        logger.error(f"Failed to compute time-domain metrics: {e}")
        return {
            "Mean_Amplitude": 0.0,
            "Std_Amplitude": 0.0,
            "Skewness": 0.0,
            "Kurtosis": 0.0
        }

def detect_artifacts(data, threshold=5.0):
    """
    Detect potential artifacts in the signal based on amplitude and variance.

    Parameters:
        data (np.array): 1D signal data.
        threshold (float): Z-score threshold for artifact detection.

    Returns:
        bool: True if the epoch is flagged as an artifact, False otherwise.
    """
    try:
        # Compute z-scores of the absolute amplitude
        z_scores = np.abs(data - np.mean(data)) / np.std(data)
        # Flag if max z-score exceeds threshold or variance is unusually high
        if np.max(z_scores) > threshold or np.var(data) > 1e6:  # Arbitrary high variance threshold
            return True
        return False
    except Exception as e:
        logger.warning(f"Failed to detect artifacts: {e}")
        return False

def compute_asymmetry_metrics(data, ch_names, sfreq):
    """
    Compute asymmetry metrics, such as frontal alpha asymmetry (F4-F3).

    Parameters:
        data (np.array): 2D array (n_channels x n_samples).
        ch_names (list): List of channel names.
        sfreq (float): Sampling frequency.

    Returns:
        dict: Dictionary with asymmetry metrics.
    """
    metrics = {}
    try:
        if "F3" in ch_names and "F4" in ch_names:
            f3_idx = ch_names.index("F3")
            f4_idx = ch_names.index("F4")
            f3_alpha, _, _, _ = compute_band_power(data[f3_idx, :], sfreq, BANDS["Alpha"])
            f4_alpha, _, _, _ = compute_band_power(data[f4_idx, :], sfreq, BANDS["Alpha"])
            metrics["Frontal_Alpha_Asymmetry"] = f4_alpha - f3_alpha
        return metrics
    except Exception as e:
        logger.warning(f"Failed to compute asymmetry metrics: {e}")
        return {"Frontal_Alpha_Asymmetry": 0.0}

def compute_vigilance_metrics(eo_data, ec_data, sfreq):
    """
    Compute vigilance-related metrics, such as alpha reactivity (EO/EC ratio).

    Parameters:
        eo_data (np.array): 1D signal data for EO condition.
        ec_data (np.array): 1D signal data for EC condition.
        sfreq (float): Sampling frequency.

    Returns:
        dict: Dictionary with vigilance metrics.
    """
    metrics = {}
    try:
        if eo_data is not None and ec_data is not None:
            eo_alpha, _, _, _ = compute_band_power(eo_data, sfreq, BANDS["Alpha"])
            ec_alpha, _, _, _ = compute_band_power(ec_data, sfreq, BANDS["Alpha"])
            alpha_ratio = eo_alpha / ec_alpha if ec_alpha > 0 else 0.0
            metrics["Alpha_Reactivity"] = alpha_ratio
            # Simple vigilance state estimation based on alpha ratio
            if alpha_ratio < 0.5:
                metrics["Vigilance_State"] = "High Vigilance (EO Dominant)"
            elif alpha_ratio < 1.0:
                metrics["Vigilance_State"] = "Moderate Vigilance"
            else:
                metrics["Vigilance_State"] = "Low Vigilance (EC Dominant)"
        return metrics
    except Exception as e:
        logger.warning(f"Failed to compute vigilance metrics: {e}")
        return {"Alpha_Reactivity": 0.0, "Vigilance_State": "Unknown"}

def process_edf_to_csv(eo_path=None, ec_path=None, epoch_length=2.0, overlap=0.0, output_dir="output"):
    """
    Load EDF files (EO and/or EC), segment the data into epochs, compute comprehensive metrics,
    and export the results to CSV files.

    Parameters:
        eo_path (str, optional): Path to the EO EDF file.
        ec_path (str, optional): Path to the EC EDF file.
        epoch_length (float): Duration (in seconds) for each epoch.
        overlap (float): Overlap fraction between epochs (0.0 to 1.0).
        output_dir (str): Output directory for CSV files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    raw_eo = io_utils.load_eeg_data(eo_path, use_csd=False, for_source=False, apply_filter=True) if eo_path else None
    raw_ec = io_utils.load_eeg_data(ec_path, use_csd=False, for_source=False, apply_filter=True) if ec_path else None

    if raw_eo is None and raw_ec is None:
        logger.error("No valid EDF data provided. Exiting.")
        return

    # Process each condition (EO and EC)
    conditions = []
    if raw_eo:
        conditions.append(("EO", raw_eo))
    if raw_ec:
        conditions.append(("EC", raw_ec))

    detailed_rows = []
    summary_rows = []
    differential_rows = []

    for cond, raw in conditions:
        sfreq = raw.info["sfreq"]
        # Compute step size for overlapping epochs
        epoch_samples = int(epoch_length * sfreq)
        step_samples = int(epoch_samples * (1 - overlap))
        events = mne.make_fixed_length_events(raw, duration=epoch_length, overlap=overlap, verbose=False)
        epochs = mne.Epochs(raw, events, tmin=0, tmax=epoch_length, baseline=None, preload=True, verbose=False)

        cond_rows = []
        channel_summaries = {ch: [] for ch in epochs.ch_names}

        for i, epoch in enumerate(epochs.get_data()):
            epoch_start = events[i, 0] / sfreq
            epoch_end = epoch_start + epoch_length
            for ch_idx, ch in enumerate(epochs.ch_names):
                signal = epoch[ch_idx, :]
                # Detect artifacts
                is_artifact = detect_artifacts(signal)
                row = {
                    "Condition": cond,
                    "Channel": ch,
                    "Epoch": i,
                    "Start_Time": epoch_start,
                    "End_Time": epoch_end,
                    "Is_Artifact": is_artifact
                }
                # Frequency-domain metrics
                for band_name, band_range in BANDS.items():
                    mean_power, peak_freq, rel_power, total_power = compute_band_power(signal, sfreq, band_range)
                    row[f"{band_name}_Power"] = mean_power
                    row[f"{band_name}_Peak_Freq"] = peak_freq
                    row[f"{band_name}_Rel_Power"] = rel_power
                    if band_name == "Delta":  # Store total power once
                        row["Total_Power"] = total_power
                # Time-domain metrics
                time_metrics = compute_time_domain_metrics(signal)
                row.update(time_metrics)
                # Asymmetry metrics (per epoch)
                asymmetry_metrics = compute_asymmetry_metrics(epoch, epochs.ch_names, sfreq)
                row.update(asymmetry_metrics)

                cond_rows.append(row)
                channel_summaries[ch].append(row)

        # Compute summary per channel for this condition
        for ch, ch_rows in channel_summaries.items():
            if ch_rows:
                summary = {"Condition": cond, "Channel": ch}
                for key in ch_rows[0]:
                    if key not in ["Condition", "Channel", "Epoch", "Start_Time", "End_Time", "Is_Artifact"]:
                        values = [row[key] for row in ch_rows if not row["Is_Artifact"]]
                        summary[f"{key}_Mean"] = np.mean(values) if values else 0.0
                        summary[f"{key}_Std"] = np.std(values) if values else 0.0
                summary_rows.append(summary)

        detailed_rows.extend(cond_rows)

    # Compute differential metrics if both EO and EC are provided
    if raw_eo and raw_ec:
        eo_epochs = mne.Epochs(raw_eo, mne.make_fixed_length_events(raw_eo, duration=epoch_length, overlap=overlap, verbose=False),
                               tmin=0, tmax=epoch_length, baseline=None, preload=True, verbose=False)
        ec_epochs = mne.Epochs(raw_ec, mne.make_fixed_length_events(raw_ec, duration=epoch_length, overlap=overlap, verbose=False),
                               tmin=0, tmax=epoch_length, baseline=None, preload=True, verbose=False)
        min_epochs = min(len(eo_epochs), len(ec_epochs))
        for i in range(min_epochs):
            eo_epoch = eo_epochs.get_data()[i]
            ec_epoch = ec_epochs.get_data()[i]
            for ch_idx, ch in enumerate(eo_epochs.ch_names):
                if ch not in ec_epochs.ch_names:
                    continue
                eo_signal = eo_epoch[ch_idx, :]
                ec_signal = ec_epoch[ec_epochs.ch_names.index(ch), :]
                row = {
                    "Channel": ch,
                    "Epoch": i,
                    "Start_Time": i * (epoch_length * (1 - overlap))
                }
                # Compute vigilance metrics
                vigilance_metrics = compute_vigilance_metrics(eo_signal, ec_signal, sfreq)
                row.update(vigilance_metrics)
                # Compute differential band powers
                for band_name, band_range in BANDS.items():
                    eo_power, _, _, _ = compute_band_power(eo_signal, sfreq, band_range)
                    ec_power, _, _, _ = compute_band_power(ec_signal, sfreq, band_range)
                    row[f"{band_name}_Power_Diff"] = eo_power - ec_power
                differential_rows.append(row)

    # Export to CSV
    # Detailed CSV
    detailed_df = pd.DataFrame(detailed_rows)
    detailed_csv = output_dir / "detailed_metrics.csv"
    detailed_df.to_csv(detailed_csv, index=False)
    logger.info(f"Detailed metrics CSV saved to {detailed_csv}")

    # Summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_dir / "summary_metrics.csv"
    summary_df.to_csv(summary_csv, index=False)
    logger.info(f"Summary metrics CSV saved to {summary_csv}")

    # Differential CSV (if applicable)
    if differential_rows:
        differential_df = pd.DataFrame(differential_rows)
        differential_csv = output_dir / "differential_metrics.csv"
        differential_df.to_csv(differential_csv, index=False)
        logger.info(f"Differential metrics CSV saved to {differential_csv}")

def main():
    parser = argparse.ArgumentParser(
        description="Process EDF files (EO and/or EC) and export comprehensive metrics to CSV."
    )
    parser.add_argument("--eo", help="Path to the eyes-open EDF file")
    parser.add_argument("--ec", help="Path to the eyes-closed EDF file")
    parser.add_argument("--epoch_length", type=float, default=2.0, help="Epoch length in seconds (default: 2.0)")
    parser.add_argument("--overlap", type=float, default=0.0, help="Overlap fraction between epochs (0.0 to 1.0, default: 0.0)")
    parser.add_argument("--output", default="output", help="Output directory for CSV files (default: output)")
    args = parser.parse_args()

    if not (args.eo or args.ec):
        parser.error("At least one of --eo or --ec must be provided.")

    process_edf_to_csv(args.eo, args.ec, args.epoch_length, args.overlap, args.output)

if __name__ == "__main__":
    main()
