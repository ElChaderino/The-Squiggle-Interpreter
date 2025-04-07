#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
clinical_report.py - Comprehensive Clinical EEG Analysis and Reporting

This script generates a detailed EEG report from EDF files, including:
- Site-specific clinical metrics (e.g., Alpha Change, Theta/Beta Ratio, Alpha Peak Frequency, Frontal Asymmetry).
- Global metrics (e.g., Frontal Asymmetry).
- Detailed per-site, per-band plots (PSD overlays, waveform overlays, difference plots).
- Topomaps for absolute and relative power across frequency bands.
- A text report with clinical interpretations and pyramid model mappings.
- CSV summaries of all metrics.

Supports Eyes Open (EO) and Eyes Closed (EC) conditions with optional CSD and filtering.

The script operates relative to the 'The-Squiggle-Interpreter' project folder,
expecting 'pyramid_model.py' in the 'modules' subdirectory and saving outputs to 'outputs/report'.

Dependencies: mne, numpy, pandas, matplotlib, pathlib, scipy
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
from typing import Dict, Tuple, List
import mne
from . import pdf_report_builder
import re

from . import pyramid_model  # Relative import since both are in the 'modules' directory
from . import processing, plotting  # Assumes these modules exist in 'modules'
from modules.io_utils import load_eeg_data as load_data  # ✅ Using enhanced version from io_utils.py

# Global Parameters
BANDS = {
    "Delta": (1, 4),
    "Theta": (4, 8),
    "Alpha": (8, 12),
    "SMR": (12, 15),
    "Beta": (15, 27),
    "HighBeta": (28, 38),
}

NORM_VALUES = {
    "Delta": {"mean": 20.0, "sd": 10.0},
    "Theta": {"mean": 15.0, "sd": 7.0},
    "Alpha": {"mean": 18.0, "sd": 6.0},
    "SMR": {"mean": 6.0, "sd": 2.5},
    "Beta": {"mean": 5.0, "sd": 2.0},
    "HighBeta": {"mean": 3.5, "sd": 1.5},
    "Alpha_Change": {"mean": 50.0, "sd": 15.0},  # Percentage change EO → EC
    "Theta_Beta_Ratio": {"mean": 1.5, "sd": 0.5},  # EO
    "Alpha_Peak_Freq": {"mean": 10.0, "sd": 1.0},  # Hz, EC
    "Delta_Power": {"mean": 20.0, "sd": 10.0},  # µV², EO
    "SMR_Power": {"mean": 6.0, "sd": 2.5},  # µV², EO
    "Frontal_Asymmetry": {"mean": 0.0, "sd": 0.1},  # Log(F4/F3 Alpha), EO
    "Total_Power": {"mean": 70.0, "sd": 20.0},  # µV², EO/EC
}

THRESHOLDS = {
    "CZ_Alpha_Percent": {"low": 30, "high": 25},
    "Theta_Beta_Ratio": {"threshold": 2.2, "severe": 3.0},
    "Total_Amplitude": {"max": 60},
    "O1_Alpha_EC": {"low": 50, "high": 150},
    "O1_Theta_Beta_Ratio": {"threshold": 1.8},
    "F3F4_Theta_Beta_Ratio": {"threshold": 2.2},
    "FZ_Delta": {"min": 9.0},
    "Alpha_Change": {"low": 30, "high": 70},
    "Alpha_Peak_Freq": {"low": 8.0, "high": 12.0},
    "Delta_Power": {"high": 30.0},
    "SMR_Power": {"low": 3.0},
    "Frontal_Asymmetry": {"low": -0.2, "high": 0.2},
    "Total_Power": {"high": 100.0},
}

DETAILED_SITES = {"CZ", "O1", "F3", "F4", "FZ", "PZ", "T3", "T4", "O2"}


# Helper Functions
def get_project_root() -> Path:
    """
    Determine the project root directory ('The-Squiggle-Interpreter') by checking
    if the current directory or its parents match the expected folder name.

    Returns:
        Path: Path to the project root directory.

    Raises:
        SystemExit: If the project root cannot be found.
    """
    current_dir = Path.cwd()
    expected_root = "The-Squiggle-Interpreter"

    if current_dir.name == expected_root:
        return current_dir
    if current_dir.name.startswith(expected_root):
        return current_dir

    for parent in current_dir.parents:
        if parent.name == expected_root or parent.name.startswith(expected_root):
            return parent

    sys.exit(
        f"Error: Could not find 'The-Squiggle-Interpreter' in the current directory ({current_dir}) "
        "or its parents. Please run this script from within the project directory."
    )


def find_edf_files(directory: Path) -> Dict[str, Path | None]:
    """
    Scan directory for EO and EC EDF files.

    Parameters:
        directory (Path): Directory to search for EDF files.

    Returns:
        Dict[str, Path | None]: Dictionary with keys "EO" and "EC" mapping to file paths or None.
    """
    edf_files = list(directory.glob("*.edf"))
    files = {"EO": None, "EC": None}
    for file in edf_files:
        name = file.name.strip().lower()
        if "eo" in name:
            files["EO"] = file
        elif "ec" in name:
            files["EC"] = file
    return files


def clean_channel_name_dynamic(ch: str) -> str:
    """
    Cleans a channel name by stripping common suffixes and prefixes.

    Parameters:
        ch (str): Original channel name.

    Returns:
        str: Cleaned channel name in uppercase.
    """
    ch_base = ch.upper()
    ch_base = re.sub(r'[-._]?(LE|RE|AVG|M1|M2|A1|A2|REF|CZREF|AV|AVERAGE|LINKED|MASTOID)$', '', ch_base)
    ch_base = re.sub(r'^(EEG\s*)?', '', ch_base)
    return ch_base


def compute_band_power(data: np.ndarray, sfreq: float, band: Tuple[float, float]) -> float:
    """
    Compute mean power for a frequency band.

    Parameters:
        data (np.ndarray): EEG signal data.
        sfreq (float): Sampling frequency (Hz).
        band (Tuple[float, float]): Frequency band range (fmin, fmax).

    Returns:
        float: Mean power in the specified band (µV²).
    """
    fmin, fmax = band
    data_filt = mne.filter.filter_data(data, sfreq, fmin, fmax, verbose=False)
    return float(np.mean(data_filt ** 2))


def compute_all_band_powers(raw: mne.io.Raw) -> Dict[str, Dict[str, float]]:
    """
    Compute band powers for all channels.

    Parameters:
        raw (mne.io.Raw): Raw EEG data.

    Returns:
        Dict[str, Dict[str, float]]: Band powers per channel {channel: {band: power, ...}}.
    """
    sfreq = raw.info["sfreq"]
    data = raw.get_data() * 1e6  # Convert to µV
    return {
        ch: {band: compute_band_power(data[i], sfreq, range_) for band, range_ in BANDS.items()}
        for i, ch in enumerate(raw.ch_names)
    }


def compute_alpha_peak_frequency(data: np.ndarray, sfreq: float, freq_range: Tuple[float, float]) -> float:
    """
    Compute the Alpha peak frequency within the specified range.

    Parameters:
        data (np.ndarray): EEG signal data (µV).
        sfreq (float): Sampling frequency (Hz).
        freq_range (Tuple[float, float]): Frequency range for Alpha band (e.g., (8, 12)).

    Returns:
        float: Alpha peak frequency in Hz, or NaN if not found.
    """
    fmin, fmax = freq_range
    freqs, psd = welch(data, fs=sfreq, nperseg=int(sfreq * 2), noverlap=int(sfreq))
    alpha_mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(alpha_mask):
        return np.nan
    alpha_freqs = freqs[alpha_mask]
    alpha_psd = psd[alpha_mask]
    return float(alpha_freqs[np.argmax(alpha_psd)])


def compute_frontal_asymmetry(bp_EO: Dict[str, Dict[str, float]], ch_left: str = "F3", ch_right: str = "F4") -> float:
    """
    Compute frontal asymmetry as log(right/left) Alpha power for EO condition.

    Parameters:
        bp_EO (Dict): Band power dictionary for EO.
        ch_left (str): Left frontal channel (default: F3).
        ch_right (str): Right frontal channel (default: F4).

    Returns:
        float: Frontal asymmetry value, or NaN if channels are missing or power is zero.
    """
    try:
        alpha_left = bp_EO[ch_left]["Alpha"]
        alpha_right = bp_EO[ch_right]["Alpha"]
        if alpha_left == 0 or alpha_right == 0:
            return np.nan
        return float(np.log(alpha_right / alpha_left))
    except KeyError:
        return np.nan


def compute_site_metrics(raw_eo: mne.io.Raw, raw_ec: mne.io.Raw, bp_EO: Dict, bp_EC: Dict) -> Tuple[Dict, Dict]:
    """
    Compute site-specific and global metrics based on band power and raw data for EO and EC.

    Metrics computed per channel:
        - Alpha Change (EO → EC, %)
        - Theta/Beta Ratio (EO)
        - Alpha Peak Frequency (EC, Hz)
        - Delta Power (EO, µV²)
        - SMR Power (EO, µV²)
        - Total Power (EO and EC, µV²)
        - Total Amplitude (Alpha EO, µV²)

    Global metrics:
        - Frontal Asymmetry (F3/F4 Alpha, EO)

    Parameters:
        raw_eo (mne.io.Raw): Raw data for Eyes Open.
        raw_ec (mne.io.Raw): Raw data for Eyes Closed.
        bp_EO (Dict): Band power dictionary for EO {channel: {band: power, ...}}.
        bp_EC (Dict): Band power dictionary for EC {channel: {band: power, ...}}.

    Returns:
        Tuple[Dict, Dict]: (site_metrics, global_metrics)
            - site_metrics: {channel: {metric_name: value, ...}, ...}
            - global_metrics: {metric_name: value, ...}
    """
    site_metrics = {}
    global_metrics = {}
    sfreq = raw_eo.info["sfreq"]

    missing_ec_channels = set(bp_EO.keys()) - set(bp_EC.keys())
    if missing_ec_channels:
        with open("missing_channels_log.txt", "a") as f:
            for ch in missing_ec_channels:
                f.write(f"{ch} missing from EC\n")

    for ch in bp_EO.keys():
        if ch not in bp_EC:
            with open("missing_channels_log.txt", "a") as f:
                f.write(f"{ch} missing from EC\n")
            continue

        site_metrics[ch] = {}
        alpha_EO = bp_EO[ch].get("Alpha", 0)
        alpha_EC = bp_EC[ch].get("Alpha", 0)
        site_metrics[ch]["Alpha_Change"] = (
            ((alpha_EC - alpha_EO) / alpha_EO) * 100 if alpha_EO != 0 else np.nan
        )

        theta = bp_EO[ch].get("Theta", np.nan)
        beta = bp_EO[ch].get("Beta", np.nan)
        site_metrics[ch]["Theta_Beta_Ratio"] = theta / beta if beta != 0 else np.nan

        ec_sig = raw_ec.get_data(picks=[ch])[0] * 1e6
        site_metrics[ch]["Alpha_Peak_Freq"] = compute_alpha_peak_frequency(ec_sig, sfreq, BANDS["Alpha"])

        site_metrics[ch]["Delta_Power"] = bp_EO[ch].get("Delta", np.nan)
        site_metrics[ch]["SMR_Power"] = bp_EO[ch].get("SMR", np.nan)
        site_metrics[ch]["Total_Power_EO"] = sum(bp_EO[ch].values())
        site_metrics[ch]["Total_Power_EC"] = sum(bp_EC[ch].values())
        site_metrics[ch]["Total_Amplitude"] = alpha_EO

    global_metrics["Frontal_Asymmetry"] = compute_frontal_asymmetry(bp_EO, "F3", "F4")
    return site_metrics, global_metrics


def flag_abnormal(value: float, metric: str) -> str:
    """
    Flag if a metric value is outside normative range (±2 SD or specific thresholds).

    Parameters:
        value (float): Metric value to evaluate.
        metric (str): Name of the metric (e.g., "Alpha_Change").

    Returns:
        str: Description of abnormality (e.g., "Above normative range").
    """
    if np.isnan(value):
        return "Not computed"

    if metric in NORM_VALUES:
        norm_mean = NORM_VALUES[metric]["mean"]
        norm_sd = NORM_VALUES[metric]["sd"]
        if value < norm_mean - 2 * norm_sd:
            return "Below normative range"
        if value > norm_mean + 2 * norm_sd:
            return "Above normative range"
    return "Within normative range"


def generic_interpretation(band: str, value: float) -> str:
    """
    Provide generic clinical interpretation for a band.

    Parameters:
        band (str): Frequency band name (e.g., "Alpha").
        value (float): Power value for the band (µV²).

    Returns:
        str: Clinical interpretation string.
    """
    norm_mean, norm_sd = NORM_VALUES[band]["mean"], NORM_VALUES[band]["sd"]
    flag = flag_abnormal(value, band)
    if flag == "Within normative range":
        return "Activity is within normative limits."

    interpretations = {
        "Below normative range": {
            "Delta": "Reduced slow-wave activity, possibly impacting recovery.",
            "Theta": "Low Theta may indicate hyperarousal or difficulty relaxing.",
            "Alpha": "Low Alpha may suggest stress or impaired visual processing.",
            "SMR": "Low SMR may reflect reduced calm focus and increased impulsivity.",
            "Beta": "Low Beta may indicate diminished cognitive processing or alertness.",
            "HighBeta": "Lower HighBeta is generally not concerning.",
        },
        "Above normative range": {
            "Delta": "High Delta may indicate cognitive impairment or attentional deficits.",
            "Theta": "Elevated Theta may be linked to drowsiness or inattention.",
            "Alpha": "Excessive Alpha may suggest over-relaxation or disengagement.",
            "SMR": "High SMR might reflect overcompensation in motor control.",
            "Beta": "Elevated Beta may relate to anxiety, stress, or hyperarousal.",
            "HighBeta": "High HighBeta may indicate excessive stress or cortical overactivation.",
        },
    }
    return interpretations.get(flag, {}).get(band, "")


def interpret_metrics(site_metrics: Dict, global_metrics: Dict, bp_EO: Dict, bp_EC: Dict) -> List[str]:
    """
    Generate clinical interpretations for each metric and band power.

    Parameters:
        site_metrics (Dict): Site-specific metrics from compute_site_metrics.
        global_metrics (Dict): Global metrics from compute_site_metrics.
        bp_EO (Dict): Band power dictionary for EO.
        bp_EC (Dict): Band power dictionary for EC.

    Returns:
        List[str]: List of interpretation strings for the report.
    """
    interpretations = []
    for ch in bp_EO:
        interpretations.append(f"\n=== Site: {ch} ===")
        site_upper = ch.upper()

        interpretations.append("Frequency Band Powers (µV²):")
        for band, power in bp_EO[ch].items():
            flag = flag_abnormal(power, band)
            interpretations.append(f"  {band}: {power:.2f} ({flag})")
            if flag != "Within normative range":
                interpretations.append(f"    {generic_interpretation(band, power)}")

        if site_upper in DETAILED_SITES:
            metrics = site_metrics[ch]
            alpha_change = metrics["Alpha_Change"]
            flag = flag_abnormal(alpha_change, "Alpha_Change")
            interpretations.append(f"Alpha Change (EO→EC): {alpha_change:.1f}% ({flag})")
            if site_upper == "CZ":
                if alpha_change < THRESHOLDS["CZ_Alpha_Percent"]["low"]:
                    interpretations.extend([
                        "  Implications: Deficits in visual processing, memory retention, and recall.",
                        "                Poor cortical engagement during state transitions.",
                        "  Recommendations: Enhance Alpha during EC for memory consolidation.",
                        "                  Optimize modulation for cognitive flexibility.",
                    ])
                elif alpha_change > THRESHOLDS["CZ_Alpha_Percent"]["high"]:
                    interpretations.extend([
                        "  Implications: Cognitive fog or inefficiencies in cortical regulation.",
                        "  Recommendations: Enhance Beta (15-18 Hz) and SMR (12-15 Hz), reduce Theta/Delta.",
                        "                  Consider coherence training between frontal and central regions.",
                    ])
            elif site_upper == "O1":
                if alpha_change < THRESHOLDS["O1_Alpha_EC"]["low"]:
                    interpretations.extend([
                        "  Implications: May indicate traumatic stress or unresolved psychological trauma.",
                        "  Recommendations: Enhance Alpha (8-12 Hz), inhibit Theta (4-7 Hz) during EC.",
                    ])
                elif alpha_change > THRESHOLDS["O1_Alpha_EC"]["high"]:
                    interpretations.extend([
                        "  Implications: Suggests enhanced artistic interest or introspection.",
                        "  Recommendations: Use Alpha training to balance creativity with emotional stability.",
                    ])
            elif site_upper == "PZ":
                if alpha_change < THRESHOLDS["CZ_Alpha_Percent"]["low"]:
                    interpretations.extend([
                        "  Implications: Issues with sensory integration and attention load management.",
                        "  Recommendations: Enhance Alpha modulation, cross-check with FZ and O1.",
                    ])
                elif alpha_change > THRESHOLDS["CZ_Alpha_Percent"]["high"]:
                    interpretations.extend([
                        "  Implications: Over-relaxation or reduced sensory processing efficiency.",
                        "  Recommendations: Balance with Beta training.",
                    ])
            elif site_upper == "O2":
                if alpha_change < THRESHOLDS["O1_Alpha_EC"]["low"]:
                    interpretations.extend([
                        "  Implications: Visual processing issues or stress-related disturbances.",
                        "  Recommendations: Enhance Alpha, assess with O1 and Pz.",
                    ])
                elif alpha_change > THRESHOLDS["O1_Alpha_EC"]["high"]:
                    interpretations.extend([
                        "  Implications: Possible overcompensation in visual processing.",
                        "  Recommendations: Stabilize Alpha, reduce excessive activity.",
                    ])

            tb_ratio = metrics["Theta_Beta_Ratio"]
            flag = flag_abnormal(tb_ratio, "Theta_Beta_Ratio")
            interpretations.append(f"Theta/Beta Ratio (EO): {tb_ratio:.2f} ({flag})")
            if site_upper in {"CZ", "O1", "F3", "F4", "T3", "T4"}:
                if tb_ratio > THRESHOLDS["Theta_Beta_Ratio"]["severe"]:
                    interpretations.extend([
                        "  Severe: Indicative of ADHD-like symptoms (hyperactivity, impulsivity).",
                        "  Recommendation: Inhibit Theta (4-8 Hz), enhance Beta (15-27 Hz).",
                    ])
                elif tb_ratio > THRESHOLDS["Theta_Beta_Ratio"]["threshold"]:
                    interpretations.extend([
                        "  Suggestive of attention regulation challenges.",
                        "  Recommendation: Monitor and consider Theta/Beta training.",
                    ])
            if site_upper == "O1" and tb_ratio < THRESHOLDS["O1_Theta_Beta_Ratio"]["threshold"]:
                interpretations.extend([
                    "  Implications: Reflects poor stress tolerance or heightened anxiety.",
                    "  Recommendations: Promote Theta stabilization, inhibit excessive Beta.",
                ])
            elif site_upper in {"F3", "F4"} and tb_ratio > THRESHOLDS["F3F4_Theta_Beta_Ratio"]["threshold"]:
                interpretations.extend([
                    "  Implications: Cognitive deficiencies, emotional volatility, or poor impulse control.",
                    "  Recommendations: Inhibit Theta, enhance Alpha for calm alertness and executive function.",
                ])
            elif site_upper in {"T3", "T4"} and tb_ratio > THRESHOLDS["Theta_Beta_Ratio"]["threshold"]:
                interpretations.extend([
                    "  Implications: Auditory processing deficits or emotional dysregulation.",
                    "  Recommendations: Balance Theta/Beta, target auditory processing training.",
                ])

            apf = metrics["Alpha_Peak_Freq"]
            flag = flag_abnormal(apf, "Alpha_Peak_Freq")
            interpretations.append(f"Alpha Peak Frequency (EC): {apf:.2f} Hz ({flag})")
            if apf < THRESHOLDS["Alpha_Peak_Freq"]["low"]:
                interpretations.extend([
                    "  Implication: Slowed Alpha peak, may indicate hypoarousal or cognitive slowing.",
                    "  Recommendation: Enhance Alpha frequency through training.",
                ])
            elif apf > THRESHOLDS["Alpha_Peak_Freq"]["high"]:
                interpretations.extend([
                    "  Implication: Fast Alpha peak, may indicate hyperarousal or anxiety.",
                    "  Recommendation: Stabilize Alpha frequency, reduce stress.",
                ])

            delta = metrics["Delta_Power"]
            flag = flag_abnormal(delta, "Delta_Power")
            interpretations.append(f"Delta Power (EO): {delta:.2f} µV² ({flag})")
            if site_upper == "FZ" and delta > THRESHOLDS["FZ_Delta"]["min"]:
                interpretations.extend([
                    "  Implications: Suggests cognitive deficits, poor concentration, or delayed development.",
                    "  Recommendations: Inhibit Delta, enhance SMR for cognitive clarity.",
                ])
            elif delta > THRESHOLDS["Delta_Power"]["high"]:
                interpretations.extend([
                    "  Implication: Excessive slow-wave activity, possible cognitive deficits.",
                    "  Recommendation: Inhibit Delta (1-4 Hz), enhance SMR/Beta.",
                ])

            smr = metrics["SMR_Power"]
            flag = flag_abnormal(smr, "SMR_Power")
            interpretations.append(f"SMR Power (EO): {smr:.2f} µV² ({flag})")
            if smr < THRESHOLDS["SMR_Power"]["low"]:
                interpretations.extend([
                    "  Implication: Low SMR, may reflect reduced calm focus or motor control.",
                    "  Recommendation: Enhance SMR (12-15 Hz) training.",
                ])

            total_eo = metrics["Total_Power_EO"]
            total_ec = metrics["Total_Power_EC"]
            flag_eo = flag_abnormal(total_eo, "Total_Power")
            flag_ec = flag_abnormal(total_ec, "Total_Power")
            interpretations.extend([
                f"Total Power (EO): {total_eo:.2f} µV² ({flag_eo})",
                f"Total Power (EC): {total_ec:.2f} µV² ({flag_ec})",
            ])
            if total_eo > THRESHOLDS["Total_Power"]["high"]:
                interpretations.extend([
                    "  Implication: High overall power (EO), possible developmental delays.",
                    "  Recommendation: Inhibit slow-wave activity, assess for artifacts.",
                ])

            total_amplitude = metrics["Total_Amplitude"]
            if site_upper == "CZ" and total_amplitude > THRESHOLDS["Total_Amplitude"]["max"]:
                interpretations.extend([
                    f"Total Amplitude (Alpha EO): {total_amplitude:.2f} µV²",
                    "  Implications: Potential developmental delays or cognitive deficits.",
                    "  Recommendations: Inhibit slow-wave activity, enhance SMR/Beta.",
                ])

    interpretations.append("\n=== Global Metrics ===")
    fa = global_metrics["Frontal_Asymmetry"]
    flag = flag_abnormal(fa, "Frontal_Asymmetry")
    interpretations.append(f"Frontal Asymmetry (F4/F3 Alpha, EO): {fa:.2f} ({flag})")
    if fa < THRESHOLDS["Frontal_Asymmetry"]["low"]:
        interpretations.extend([
            "  Implication: Left-dominant asymmetry, may indicate depressive tendencies.",
            "  Recommendation: Enhance right frontal Alpha, monitor emotional regulation.",
        ])
    elif fa > THRESHOLDS["Frontal_Asymmetry"]["high"]:
        interpretations.extend([
            "  Implication: Right-dominant asymmetry, may indicate withdrawal behavior.",
            "  Recommendation: Enhance left frontal Alpha, assess emotional state.",
        ])

    return interpretations


def save_site_metrics(site_metrics: Dict, global_metrics: Dict, output_path: Path) -> None:
    """
    Save computed site-specific and global metrics to a CSV file.

    Parameters:
        site_metrics (Dict): Site-specific metrics from compute_site_metrics.
        global_metrics (Dict): Global metrics from compute_site_metrics.
        output_path (Path): File path to save the CSV.
    """
    rows = []
    for ch, met in site_metrics.items():
        row = {"Channel": ch}
        row.update({band: power for band, power in met.items()})
        rows.append(row)
    global_row = {"Channel": "Global"}
    global_row.update(global_metrics)
    rows.append(global_row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Clinical metrics saved to: {output_path}")


def remove_overlapping_channels(info: mne.Info, tol: float = 0.05) -> Tuple[mne.Info, List[int]]:
    """
    Remove channels with overlapping positions for topomap visualization.

    Parameters:
        info (mne.Info): MNE info object containing channel locations.
        tol (float): Tolerance for considering positions as overlapping (default: 0.05).

    Returns:
        Tuple[mne.Info, List[int]]: Cleaned info object and indices of unique channels.
    """
    ch_names = info["ch_names"]
    pos = np.array([info["chs"][i]["loc"][:3] for i in range(len(ch_names))])
    unique_idx = []
    for i in range(len(ch_names)):
        if not any(np.linalg.norm(pos[i] - pos[j]) < tol for j in unique_idx):
            unique_idx.append(i)
    info_clean = mne.pick_info(info, sel=unique_idx)
    return info_clean, unique_idx


CRITICAL_SITES = {"F3", "CZ", "O1", "FZ", "PZ", "F4", "O2", "T3", "T4"}


def plot_topomap_abs_rel(
    abs_vals: np.ndarray,
    rel_vals: np.ndarray,
    info: mne.Info,
    band_name: str,
    cond_name: str,
    output_path: Path,
    cmap: str = "viridis",
    figsize: Tuple[float, float] = (10, 4),
    dpi: int = 300,
    normalize: bool = False,
    show_ch_names: bool = True
) -> None:
    """
    Plot side-by-side topomaps of absolute and relative power for a given band.

    Parameters:
        abs_vals (np.ndarray): Absolute power values for each channel (µV²).
        rel_vals (np.ndarray): Relative power values for each channel (%).
        info (mne.Info): MNE info object with channel locations.
        band_name (str): Name of the frequency band (e.g., "Alpha").
        cond_name (str): Condition name (e.g., "EO" or "EC").
        output_path (Path): Path to save the plot.
        cmap (str): Colormap for topomaps (default: "viridis").
        figsize (Tuple[float, float]): Figure size in inches (default: (10, 4)).
        dpi (int): Resolution for saved image (default: 300).
        normalize (bool): Normalize power values to [0, 1] range (default: False).
        show_ch_names (bool): Overlay channel names on topomaps (default: True).
    """
    # Validate channel locations safely
    pos = np.array([ch["loc"][:3] for ch in info["chs"] if ch.get("loc") is not None])
    if len(pos) == 0 or not np.any(pos):
        print(f"Warning: No valid sensor positions for topomap ({cond_name}, {band_name}). Skipping.")
        with open("missing_channels_log.txt", "a") as log:
            log.write(f"\n=== Topomap Failure ({cond_name}, {band_name}) ===\n")
            log.write("No valid sensor positions. Skipping.\n")
        return

    info_clean, sel_idx = remove_overlapping_channels(info)
    if not sel_idx:
        print(f"Warning: No unique channel positions after overlap removal ({cond_name}, {band_name}). Skipping.")
        with open("missing_channels_log.txt", "a") as log:
            log.write(f"\n=== Topomap Failure ({cond_name}, {band_name}) ===\n")
            log.write("All channel positions overlap or are invalid. Topomap skipped.\n")
        return

    # Abort if critical channels are missing
    CRITICAL_SITES = {"F3", "CZ", "O1", "FZ", "PZ", "F4", "O2", "T3", "T4"}
    cleaned_ch_names = set(info_clean["ch_names"])
    missing_critical = CRITICAL_SITES - cleaned_ch_names
    if missing_critical:
        print(f"[!] Skipping topomap: Missing critical electrodes {missing_critical} for {band_name} ({cond_name})")
        with open("missing_channels_log.txt", "a") as log:
            log.write(f"\n=== Topomap Skipped ({cond_name}, {band_name}) ===\n")
            log.write(f"Missing critical electrodes: {missing_critical}\n")
        return

    # Final check for cleaned info
    final_pos = np.array([ch["loc"][:3] for ch in info_clean["chs"] if ch.get("loc") is not None])
    if len(final_pos) == 0 or not np.any(final_pos):
        print(f"[!] Skipping topomap: info_clean has no valid sensor positions for {band_name} ({cond_name})")
        with open("missing_channels_log.txt", "a") as log:
            log.write(f"\n=== Topomap Skipped ({cond_name}, {band_name}) ===\n")
            log.write("No valid sensor positions in cleaned info. Skipping.\n")
        return

    # Subset power values
    abs_vals_subset = abs_vals[sel_idx]
    rel_vals_subset = rel_vals[sel_idx]

    # Optional normalization
    if normalize:
        abs_vals_subset = (abs_vals_subset - abs_vals_subset.min()) / (abs_vals_subset.max() - abs_vals_subset.min() + 1e-10)
        rel_vals_subset = (rel_vals_subset - rel_vals_subset.min()) / (rel_vals_subset.max() - rel_vals_subset.min() + 1e-10)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor="black")
    fig.patch.set_facecolor("black")

    # Absolute power topomap
    ax_abs = axes[0]
    ax_abs.set_facecolor("black")
    vmin_abs, vmax_abs = np.min(abs_vals_subset), np.max(abs_vals_subset)
    im_abs, _ = mne.viz.plot_topomap(
        abs_vals_subset, info_clean, axes=ax_abs, show=False, cmap=cmap,
        vlim=(vmin_abs, vmax_abs), names=info_clean["ch_names"] if show_ch_names else None
    )
    ax_abs.set_title(f"{band_name} Abs Power ({cond_name})", color="white", fontsize=10)
    cbar_abs = plt.colorbar(im_abs, ax=ax_abs, orientation="horizontal", fraction=0.05, pad=0.08)
    cbar_abs.set_label("µV²" if not normalize else "Normalized", color="white")
    cbar_abs.ax.tick_params(colors="white")

    # Relative power topomap
    ax_rel = axes[1]
    ax_rel.set_facecolor("black")
    vmin_rel, vmax_rel = np.min(rel_vals_subset), np.max(rel_vals_subset)
    im_rel, _ = mne.viz.plot_topomap(
        rel_vals_subset, info_clean, axes=ax_rel, show=False, cmap=cmap,
        vlim=(vmin_rel, vmax_rel), names=info_clean["ch_names"] if show_ch_names else None
    )
    ax_rel.set_title(f"{band_name} Rel Power ({cond_name})", color="white", fontsize=10)
    cbar_rel = plt.colorbar(im_rel, ax=ax_rel, orientation="horizontal", fraction=0.05, pad=0.08)
    cbar_rel.set_label("%" if not normalize else "Normalized", color="white")
    cbar_rel.ax.tick_params(colors="white")

    # Figure title and layout
    fig.suptitle(f"Global Topomap (Abs & Rel) - {band_name} ({cond_name})", color="white", fontsize=12)
    fig.tight_layout()

    # Save and clean up
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"Saved topomap for {band_name} ({cond_name}) to {output_path}")


def generate_full_site_plots(raw_eo: mne.io.Raw, raw_ec: mne.io.Raw, output_dir: Path) -> None:
    """
    Generate detailed per-site, per-band plots for EO vs. EC.

    For each channel (site) and frequency band, this function:
        - Creates a subfolder for the channel.
        - Generates a PSD overlay plot comparing EO and EC.
        - Generates a waveform overlay plot comparing EO and EC.
        - Generates a difference bar plot comparing EO vs. EC band power.

    Parameters:
        raw_eo (mne.io.Raw): Raw data for Eyes Open.
        raw_ec (mne.io.Raw): Raw data for Eyes Closed.
        output_dir (Path): Base directory where per-site plots will be saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    channels = raw_eo.ch_names
    sfreq = raw_eo.info["sfreq"]

    for ch in channels:
        if ch not in raw_ec.ch_names or ch not in raw_eo.ch_names:
            with open("missing_channels_log.txt", "a") as f:
                f.write(f"{ch} missing in EO or EC for plotting\n")
            continue

        ch_folder = output_dir / ch
        psd_folder = ch_folder / "PSD_Overlay"
        wave_folder = ch_folder / "Waveform_Overlay"
        diff_folder = ch_folder / "Difference"
        psd_folder.mkdir(parents=True, exist_ok=True)
        wave_folder.mkdir(parents=True, exist_ok=True)
        diff_folder.mkdir(parents=True, exist_ok=True)

        eo_sig = raw_eo.get_data(picks=[ch])[0] * 1e6
        ec_sig = raw_ec.get_data(picks=[ch])[0] * 1e6

        for band_name, band_range in BANDS.items():
            fig_psd = plotting.plot_band_psd_overlay(
                eo_sig, ec_sig, sfreq, band_range, ch, band_name, colors=("cyan", "magenta")
            )
            psd_path = psd_folder / f"{ch}_PSD_{band_name}.png"
            fig_psd.savefig(psd_path, facecolor="black")
            plt.close(fig_psd)

            fig_wave = plotting.plot_band_waveform_overlay(
                eo_sig, ec_sig, sfreq, band_range, ch, band_name, colors=("cyan", "magenta"), epoch_length=10
            )
            wave_path = wave_folder / f"{ch}_Waveform_{band_name}.png"
            fig_wave.savefig(wave_path, facecolor="black")
            plt.close(fig_wave)

            power_eo = processing.compute_band_power(eo_sig, sfreq, band_range)
            power_ec = processing.compute_band_power(ec_sig, sfreq, band_range)
            fig_diff, ax = plt.subplots(figsize=(4, 4), facecolor="black")
            ax.bar(["EO", "EC"], [power_eo, power_ec], color=["cyan", "magenta"])
            ax.set_title(f"{ch} {band_name} Difference", color="white", fontsize=10)
            ax.set_ylabel("Power (µV²)", color="white")
            ax.tick_params(colors="white")
            ax.set_facecolor("black")
            fig_diff.tight_layout()
            diff_path = diff_folder / f"{ch}_Difference_{band_name}.png"
            fig_diff.savefig(diff_path, facecolor="black")
            plt.close(fig_diff)

            print(f"Saved detailed plots for channel {ch}, band {band_name} in {ch_folder}")


def get_pyramid_mapping_section() -> str:
    """
    Format pyramid model mappings as text for inclusion in the clinical report.

    Returns:
        str: Formatted string containing pyramid model mappings.
    """
    lines = ["=== Refined Clinical Mapping (Pyramid Levels) ==="]
    for level, mapping in pyramid_model.list_all_refined_mappings():
        lines.extend([
            mapping["level_name"],
            f"  EEG Patterns: {', '.join(mapping['eeg_patterns'])}",
            f"  Cognitive/Behavioral: {mapping['cognitive_behavior']}",
            f"  Protocols: {mapping['protocols']}",
            "",
        ])
    lines.append("=== EEG Connectivity Mapping ===")
    for level, mapping in pyramid_model.list_all_connectivity_mappings():
        lines.extend([
            mapping["level_name"],
            f"  EEG Patterns: {', '.join(mapping['eeg_patterns'])}",
            f"  Differentiators: {mapping['differentiators']}",
            f"  Cognitive/Behavioral: {mapping['cognition_behavior']}",
            f"  Vigilance Stage: {mapping['vigilance_stage']}",
            f"  Neurofeedback Targets: {'; '.join(mapping['neurofeedback_targets'])}",
            "",
        ])
    return "\n".join(lines)


def _generate_clinical_report(report_output_dir: Path, use_csd: bool, apply_filter: bool) -> None:
    """
    Generate and save the full clinical report with topomaps and per-site plots.

    Parameters:
        report_output_dir (Path): Directory to save the report and associated files.
        use_csd (bool): Whether to apply CSD transform.
        apply_filter (bool): Whether to apply high-pass filter.
    """
    report_output_dir.mkdir(parents=True, exist_ok=True)
    topomap_dir = report_output_dir / "topomaps"
    site_plot_dir = report_output_dir / "site_plots"
    topomap_dir.mkdir(parents=True, exist_ok=True)
    site_plot_dir.mkdir(parents=True, exist_ok=True)

    project_root = get_project_root()
    edf_dict = find_edf_files(project_root)

    if not any(edf_dict.values()):
        sys.exit("Error: No EDF files found in the project directory.")
    if edf_dict["EO"] is None:
        edf_dict["EO"] = edf_dict["EC"]
        print("Using EC file for EO.")
    if edf_dict["EC"] is None:
        edf_dict["EC"] = edf_dict["EO"]
        print("Using EO file for EC.")

    raw_eo = load_data(edf_dict["EO"], use_csd, apply_filter)
    raw_ec = load_data(edf_dict["EC"], use_csd, apply_filter)
    bp_EO = compute_all_band_powers(raw_eo)
    bp_EC = compute_all_band_powers(raw_ec)

    site_metrics, global_metrics = compute_site_metrics(raw_eo, raw_ec, bp_EO, bp_EC)
    interpretations = interpret_metrics(site_metrics, global_metrics, bp_EO, bp_EC)

    csv_path = report_output_dir / "clinical_metrics.csv"
    report_text_path = report_output_dir / "clinical_report.txt"

    save_site_metrics(site_metrics, global_metrics, csv_path)
    with open(report_text_path, "w", encoding="utf-8") as f:
        f.write("\n".join([
            "==============================================",
            "           Clinical EEG Report",
            "==============================================",
            *interpretations,
            "\n",
            get_pyramid_mapping_section(),
        ]))

    generate_full_site_plots(raw_eo, raw_ec, site_plot_dir)

    common_channels = sorted(list(set(bp_EO.keys()) & set(bp_EC.keys())))
    raw_eo_common = raw_eo.copy().pick_channels(common_channels)
    raw_ec_common = raw_ec.copy().pick_channels(common_channels)

    for band in BANDS:
        eo_abs = np.array([bp_EO[ch][band] for ch in common_channels])
        ec_abs = np.array([bp_EC[ch][band] for ch in common_channels])
        eo_total = np.array([sum(bp_EO[ch].values()) for ch in common_channels])
        ec_total = np.array([sum(bp_EC[ch].values()) for ch in common_channels])
        eo_rel = eo_abs / eo_total * 100 if np.any(eo_total) else np.zeros_like(eo_abs)
        ec_rel = ec_abs / ec_total * 100 if np.any(ec_total) else np.zeros_like(ec_abs)

        plot_topomap_abs_rel(eo_abs, eo_rel, raw_eo_common.info, band, "EO", topomap_dir / f"topomap_{band}_EO.png")
        plot_topomap_abs_rel(ec_abs, ec_rel, raw_ec_common.info, band, "EC", topomap_dir / f"topomap_{band}_EC.png")

    pdf_report_builder.build_pdf_report(report_output_dir)

    print(f"\n✅ Clinical report saved in: {report_output_dir}")
    print(f"   - PDF: clinical_report.pdf")
    print(f"   - Report: clinical_report.txt")
    print(f"   - Metrics CSV: clinical_metrics.csv")
    print(f"   - Topomaps: {topomap_dir}")
    print(f"   - Site plots: {site_plot_dir}")


def generate_full_clinical_report(*args, **kwargs):
    """
    Supports both CLI-style (Path, use_csd, apply_filter) and main.py-style
    (use_csd, apply_filter, subject_folder, subject).

    Parameters:
        *args: Variable positional arguments based on calling context.
        **kwargs: Variable keyword arguments (not used currently).

    Raises:
        TypeError: If arguments do not match expected patterns.
    """
    if len(args) == 3 and isinstance(args[0], Path):
        return _generate_clinical_report(args[0], args[1], args[2])
    elif len(args) == 4:
        use_csd, apply_filter, subject_folder, subject = args
        output_dir = Path(subject_folder)
        return _generate_clinical_report(output_dir, use_csd, apply_filter)
    else:
        raise TypeError("Invalid arguments for generate_full_clinical_report")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the script.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Clinical EEG Report Generator")
    parser.add_argument("--no-csd", action="store_true", help="Disable CSD transform")
    parser.add_argument("--no-filter", action="store_true", help="Disable high-pass filter")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory for all report files")
    return parser.parse_args()


if __name__ == "__main__":
    """
    Main entry point for command-line execution.
    """
    args = parse_args()
    generate_full_clinical_report(Path(args.outdir), use_csd=not args.no_csd, apply_filter=not args.no_filter)
