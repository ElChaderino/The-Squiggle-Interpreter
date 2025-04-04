#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
clinical_report.py - Comprehensive Clinical EEG Report with Enhanced Data Cleaning & Pyramid Mapping

This module:
  • Scans the current directory for EDF files (using pathlib for robust handling of filenames with spaces)
    and assigns them as Eyes Open (EO) or Eyes Closed (EC).
  • Loads both EDF files, renames channels (removing "-LE"), sets a standard 10–20 montage, applies an average reference,
    optionally applies a high-pass filter to remove slow drifts, and (if enabled) applies a current source density (CSD) transform.
  • Computes mean power in key frequency bands (Delta, Theta, Alpha, SMR, Beta, HighBeta) for each channel.
  • Derives key metrics per channel including:
       - Percentage change in Alpha power (EO → EC)
       - Theta/Beta ratio (EO)
       - Total amplitude (using Alpha power as a proxy)
  • Provides detailed clinical interpretations and neurofeedback recommendations for key sites,
    with generic analysis for other channels.
  • Imports refined clinical and connectivity mappings from the pyramid_model module and appends them to the report.
  • Exports a full UTF‑8 text report (so emojis display correctly) and a CSV summary.

Dependencies: mne, numpy, pandas, matplotlib, pathlib

Ensure that both this file and pyramid_model.py are saved in UTF‑8.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

# Import the pyramid mapping module (make sure pyramid_model.py is in the same folder)
from modules import pyramid_model

# ------------------------ GLOBAL PARAMETERS ------------------------ #
USE_CSD = True       # Apply current source density (CSD) transform for enhanced spatial resolution.
APPLY_FILTER = True  # Apply a high-pass filter (e.g., 1 Hz) to remove slow drifts.

# ------------------------ EDF FILE DISCOVERY ------------------------ #
def find_edf_files(directory):
    """
    Scan the given directory for .edf files using pathlib.
    The function returns a dictionary with keys "EO" and "EC" where filenames are checked
    in a case-insensitive manner and extra whitespace is stripped.
    """
    p = Path(directory)
    edf_files = list(p.glob("*.edf"))
    files = {"EO": None, "EC": None}
    for f in edf_files:
        name = f.name.strip().lower()
        if "eo" in name:
            files["EO"] = f  # f is a Path object
        elif "ec" in name:
            files["EC"] = f
    return files

edf_dict = find_edf_files(Path.cwd())
print("Found EDF files:", {k: str(v) for k, v in edf_dict.items()})
if edf_dict["EO"] is None and edf_dict["EC"] is not None:
    print("Only an EC file was found; using it for both EO and EC processing.")
    edf_dict["EO"] = edf_dict["EC"]
elif edf_dict["EC"] is None and edf_dict["EO"] is not None:
    print("Only an EO file was found; using it for both EO and EC processing.")
    edf_dict["EC"] = edf_dict["EO"]
elif edf_dict["EO"] is None and edf_dict["EC"] is None:
    sys.exit("Error: No EDF files found in the current directory.")

edf_eo = edf_dict["EO"]
edf_ec = edf_dict["EC"]

# ------------------------ SETTINGS ------------------------ #
# Frequency bands (Hz)
bands = {
    "Delta": (1, 4),
    "Theta": (4, 8),
    "Alpha": (8, 12),
    "SMR": (12, 15),
    "Beta": (15, 27),
    "HighBeta": (28, 38),
}

# Normative values (example values; adjust with published norms as needed)
norm_values = {
    "Delta": {"mean": 20.0, "sd": 10.0},
    "Theta": {"mean": 15.0, "sd": 7.0},
    "Alpha": {"mean": 18.0, "sd": 6.0},
    "SMR": {"mean": 6.0, "sd": 2.5},
    "Beta": {"mean": 5.0, "sd": 2.0},
    "HighBeta": {"mean": 3.5, "sd": 1.5},
}

# Example thresholds & recommendations for key sites
thresholds = {
    "CZ_Alpha_Percent": {"low": 30, "high": 25},
    "Theta_Beta_Ratio": {"threshold": 2.2, "severe": 3.0},
    "Total_Amplitude": {"max": 60},
    "O1_Alpha_EC": {"low": 50, "high": 150},
    "O1_Theta_Beta_Ratio": {"threshold": 1.8},
    "F3F4_Theta_Beta_Ratio": {"threshold": 2.2},
    "FZ_Delta": {"min": 9.0},
}

# List of key sites for detailed analysis
detailed_sites = {"CZ", "O1", "F3", "F4", "FZ", "PZ", "T3", "T4", "O2"}

# Base output folder
report_output_dir = Path("outputs") / "report"
report_output_dir.mkdir(parents=True, exist_ok=True)

# ------------------------ HELPER FUNCTIONS ------------------------ #
def load_data(edf_file):
    """
    Load an EDF file given as a Path object.
    Removes '-LE' suffix from channel names, sets the standard montage, applies an average reference,
    optionally applies a high-pass filter, and optionally computes CSD.
    """
    raw = mne.io.read_raw_edf(str(edf_file), preload=True, verbose=False)
    raw.rename_channels({ch: ch.replace('-LE', '') for ch in raw.ch_names})
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, match_case=False)
    raw.set_eeg_reference("average", projection=True)
    if APPLY_FILTER:
        raw.filter(l_freq=1.0, h_freq=None, verbose=False)
    if USE_CSD:
        try:
            raw = mne.preprocessing.compute_current_source_density(raw)
            print("CSD transform applied successfully to", edf_file)
        except Exception as e:
            print("CSD transform failed on", edf_file, ":", e)
    return raw

def compute_band_power(data, sfreq, band):
    """Filter data to the given band and return mean power (average squared amplitude)."""
    fmin, fmax = band
    data_filt = mne.filter.filter_data(data, sfreq, fmin, fmax, verbose=False)
    return np.mean(data_filt ** 2)

def compute_all_band_powers(raw):
    """
    Compute mean power for each channel and each frequency band.
    Returns a dict: {channel: {band: power, ...}, ...}
    """
    sfreq = raw.info["sfreq"]
    data = raw.get_data() * 1e6  # convert to µV
    results = {}
    for i, ch in enumerate(raw.ch_names):
        results[ch] = {}
        for band_name, band_range in bands.items():
            results[ch][band_name] = compute_band_power(data[i], sfreq, band_range)
    return results

def compute_percentage_change(power_EO, power_EC):
    """Compute percentage change from EO to EC."""
    if power_EO == 0:
        return np.nan
    return ((power_EC - power_EO) / power_EO) * 100

def compute_theta_beta_ratio(raw, ch):
    """Compute Theta/Beta ratio (EO) for a given channel."""
    sfreq = raw.info["sfreq"]
    data = raw.get_data(picks=[ch])[0] * 1e6
    theta = compute_band_power(data, sfreq, bands["Theta"])
    beta = compute_band_power(data, sfreq, bands["Beta"])
    if beta == 0:
        return np.nan
    return theta / beta

def flag_abnormal(value, norm_mean, norm_sd):
    """Return a flag if value is below or above ±2 SD of the normative value."""
    if value < norm_mean - 2 * norm_sd:
        return "Below normative range"
    elif value > norm_mean + 2 * norm_sd:
        return "Above normative range"
    else:
        return "Within normative range"

def generic_interpretation(band, value):
    """Return a generic clinical interpretation for a given band if abnormal."""
    norm_mean = norm_values[band]["mean"]
    norm_sd = norm_values[band]["sd"]
    flag = flag_abnormal(value, norm_mean, norm_sd)
    if flag == "Within normative range":
        return "Activity is within normative limits."
    elif flag == "Below normative range":
        if band == "Delta":
            return "Lower than expected Delta may suggest reduced slow-wave activity, possibly impacting recovery."
        elif band == "Theta":
            return "Low Theta may indicate hyperarousal or difficulty in relaxation."
        elif band == "Alpha":
            return "Low Alpha may be associated with stress or impaired visual processing."
        elif band == "SMR":
            return "Low SMR may reflect reduced calm focus and increased impulsivity."
        elif band == "Beta":
            return "Low Beta can indicate diminished cognitive processing or alertness."
        elif band == "HighBeta":
            return "Lower HighBeta is generally not concerning."
    elif flag == "Above normative range":
        if band == "Delta":
            return "High Delta may indicate cognitive impairment or attentional deficits."
        elif band == "Theta":
            return "Elevated Theta may be linked to drowsiness or inattention."
        elif band == "Alpha":
            return "Excessive Alpha may suggest over-relaxation or disengagement."
        elif band == "SMR":
            return "High SMR might be associated with overcompensation in motor control."
        elif band == "Beta":
            return "Elevated Beta may be related to anxiety, stress, or hyperarousal."
        elif band == "HighBeta":
            return "High HighBeta may reflect excessive stress or cortical overactivation."
    return ""

# --- Helper to remove overlapping channels for topomap visualization ---
def remove_overlapping_channels(info, tol=0.05):
    """
    Return a new info object with only channels that have unique positions (within tolerance tol)
    and also return the indices of these channels for subsetting data arrays.
    """
    ch_names = info["ch_names"]
    pos = np.array([info["chs"][i]["loc"][:3] for i in range(len(ch_names))])
    unique_idx = []
    for i in range(len(ch_names)):
        duplicate = False
        for j in unique_idx:
            if np.linalg.norm(pos[i] - pos[j]) < tol:
                duplicate = True
                break
        if not duplicate:
            unique_idx.append(i)
    info_clean = mne.pick_info(info, sel=unique_idx)
    return info_clean, unique_idx

def plot_topomap_abs_rel(abs_vals, rel_vals, info, band_name, cond_name):
    """Plot side-by-side topomaps of absolute and relative power for a given band."""
    info_clean, sel_idx = remove_overlapping_channels(info)
    abs_vals_subset = abs_vals[sel_idx]
    rel_vals_subset = rel_vals[sel_idx]
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor='black')
    fig.patch.set_facecolor('black')
    
    ax_abs = axes[0]
    ax_abs.set_facecolor('black')
    im_abs, _ = mne.viz.plot_topomap(abs_vals_subset, info_clean, axes=ax_abs, show=False, cmap='viridis')
    ax_abs.set_title(f"{band_name} Abs Power ({cond_name})", color='white', fontsize=10)
    cbar_abs = plt.colorbar(im_abs, ax=ax_abs, orientation='horizontal', fraction=0.05, pad=0.08)
    cbar_abs.ax.tick_params(colors='white')
    
    ax_rel = axes[1]
    ax_rel.set_facecolor('black')
    im_rel, _ = mne.viz.plot_topomap(rel_vals_subset, info_clean, axes=ax_rel, show=False, cmap='viridis')
    ax_rel.set_title(f"{band_name} Rel Power ({cond_name})", color='white', fontsize=10)
    cbar_rel = plt.colorbar(im_rel, ax=ax_rel, orientation='horizontal', fraction=0.05, pad=0.08)
    cbar_rel.ax.tick_params(colors='white')
    
    fig.suptitle(f"Global Topomap (Abs & Rel) - {band_name} ({cond_name})", color='white')
    fig.tight_layout()
    return fig

def generate_site_report(eo_results, ec_results, raw_eo):
    """
    Generate a clinical report summary for each channel.
    Returns a tuple (list of text report lines, DataFrame for CSV summary).
    """
    channels = list(eo_results.keys())
    report_lines = []
    rows = []
    for ch in channels:
        line = f"Site: {ch}\n"
        # Detailed analysis for key sites
        if ch.upper() in detailed_sites:
            if ch.upper() == "CZ":
                alpha_EO = eo_results[ch]["Alpha"]
                alpha_EC = ec_results[ch]["Alpha"]
                alpha_change = compute_percentage_change(alpha_EO, alpha_EC)
                line += f"  CZ Alpha Power Change (EO→EC): {alpha_change:.1f}%\n"
                if alpha_change < thresholds["CZ_Alpha_Percent"]["low"]:
                    line += ("  Implications: Potential deficits in visual processing, memory retention, and short-term recall.\n"
                             "                Poor cortical engagement during state transitions (difficulty downregulating Alpha).\n"
                             "  Recommendations: Enhance Alpha during EC for memory consolidation; optimize modulation to improve cognitive flexibility.\n"
                             "                  Cross-reference with O1 Alpha transitions for visual-spatial and memory challenges.\n")
                elif alpha_change > thresholds["CZ_Alpha_Percent"]["high"]:
                    line += ("  Implications: Possible cognitive fog or inefficiencies in cortical regulation; overactivation in frontal/central lobes.\n"
                             "  Recommendations: Balance activity by enhancing Beta (15-18 Hz) and SMR (12-15 Hz) while reducing excessive Theta/Delta.\n"
                             "                  Consider coherence training between frontal and central regions.\n")
                tb_ratio = compute_theta_beta_ratio(raw_eo, ch)
                line += f"  Theta/Beta Ratio (EO): {tb_ratio:.2f}\n"
                if tb_ratio > thresholds["Theta_Beta_Ratio"]["severe"]:
                    line += "  Severe: Indicative of ADHD-like symptoms (hyperactivity, impulsivity).\n"
                elif tb_ratio > thresholds["Theta_Beta_Ratio"]["threshold"]:
                    line += "  Suggestive of attention regulation challenges.\n"
                total_amp = alpha_EO  # using Alpha as a proxy
                line += f"  Total Amplitude (Alpha EO): {total_amp:.2f} µV²\n"
                if total_amp > thresholds["Total_Amplitude"]["max"]:
                    line += ("  Implications: Indicates potential developmental delays or cognitive deficits.\n"
                             "  Recommendations: Inhibit slow-wave activity and enhance SMR/Beta for improved processing.\n")
            elif ch.upper() == "O1":
                alpha_EO = eo_results[ch]["Alpha"]
                alpha_EC = ec_results[ch]["Alpha"]
                alpha_change = compute_percentage_change(alpha_EO, alpha_EC)
                line += f"  O1 Alpha Power Change (EO→EC): {alpha_change:.1f}%\n"
                if alpha_change < 50:
                    line += ("  Implications: May indicate traumatic stress or unresolved psychological trauma.\n"
                             "  Recommendations: Enhance Alpha (8-12 Hz) and inhibit Theta (4-7 Hz) during EC.\n")
                elif alpha_change > 150:
                    line += ("  Implications: Suggests enhanced artistic interest or introspection.\n"
                             "  Recommendations: Use Alpha training to balance creativity with emotional stability.\n")
                tb_ratio = compute_theta_beta_ratio(raw_eo, ch)
                line += f"  Theta/Beta Ratio (EO): {tb_ratio:.2f}\n"
                if tb_ratio < thresholds["O1_Theta_Beta_Ratio"]["threshold"]:
                    line += ("  Implications: Reflects poor stress tolerance or heightened anxiety.\n"
                             "  Recommendations: Promote Theta stabilization and inhibit excessive Beta.\n")
            elif ch.upper() in {"F3", "F4"}:
                tb_ratio = compute_theta_beta_ratio(raw_eo, ch)
                line += f"  Theta/Beta Ratio (EO): {tb_ratio:.2f}\n"
                if tb_ratio > thresholds["F3F4_Theta_Beta_Ratio"]["threshold"]:
                    line += ("  Implications: Reflects cognitive deficiencies, emotional volatility, or poor impulse control.\n"
                             "  Recommendations: Inhibit Theta and enhance Alpha to foster calm alertness and improve executive function.\n")
            elif ch.upper() == "FZ":
                delta_power = eo_results[ch]["Delta"]
                line += f"  Delta Power (EO): {delta_power:.2f} µV²\n"
                if delta_power > thresholds["FZ_Delta"]["min"]:
                    line += ("  Implications: Suggests cognitive deficits, poor concentration, or delayed neurological development.\n"
                             "  Recommendations: Inhibit Delta and enhance SMR for improved cognitive clarity.\n")
            elif ch.upper() == "PZ":
                alpha_EO = eo_results[ch]["Alpha"]
                alpha_EC = ec_results[ch]["Alpha"]
                alpha_change = compute_percentage_change(alpha_EO, alpha_EC)
                line += f"  Pz Alpha Power Change (EO→EC): {alpha_change:.1f}%\n"
                line += ("  Implications: May indicate issues with sensory integration and attention load management.\n"
                         "  Recommendations: Evaluate further with cross-site comparisons (e.g., with FZ, O1).\n")
            elif ch.upper() in {"T3", "T4"}:
                tb_ratio = compute_theta_beta_ratio(raw_eo, ch)
                line += f"  Theta/Beta Ratio (EO): {tb_ratio:.2f}\n"
                line += ("  Implications: Abnormal ratios may reflect auditory processing deficits or emotional dysregulation.\n"
                         "  Recommendations: Consider targeted training to balance auditory and language processing.\n")
            elif ch.upper() == "O2":
                alpha_EO = eo_results[ch]["Alpha"]
                alpha_EC = ec_results[ch]["Alpha"]
                alpha_change = compute_percentage_change(alpha_EO, alpha_EC)
                line += f"  O2 Alpha Power Change (EO→EC): {alpha_change:.1f}%\n"
                line += ("  Implications: May indicate visual processing or stress-related visual disturbances.\n"
                         "  Recommendations: Assess in conjunction with O1 and Pz for comprehensive evaluation.\n")
        else:
            # Generic analysis for channels without specific detailed logic
            line += "  Frequency Band Powers (µV²):\n"
            for band_name in bands.keys():
                power_val = eo_results[ch][band_name]
                norm_mean = norm_values[band_name]["mean"]
                norm_sd = norm_values[band_name]["sd"]
                flag = flag_abnormal(power_val, norm_mean, norm_sd)
                interp = generic_interpretation(band_name, power_val)
                line += f"    {band_name}: {power_val:.2f} µV² ({flag}; Norm: {norm_mean}±{norm_sd})\n"
                if flag != "Within normative range":
                    line += f"      Interpretation: {interp}\n"
        rows.append({
            "Site": ch,
            "Delta_EO": eo_results[ch]["Delta"],
            "Theta_EO": eo_results[ch]["Theta"],
            "Alpha_EO": eo_results[ch]["Alpha"],
            "SMR_EO": eo_results[ch]["SMR"],
            "Beta_EO": eo_results[ch]["Beta"],
            "HighBeta_EO": eo_results[ch]["HighBeta"],
            "Alpha_EC": ec_results[ch]["Alpha"],
            "Alpha_Change_%": (compute_percentage_change(eo_results[ch]["Alpha"], ec_results[ch]["Alpha"])
                               if ch.upper() in {"CZ", "O1", "PZ"} else np.nan),
            "Theta/Beta_Ratio_EO": (compute_theta_beta_ratio(raw_eo, ch)
                                    if ch.upper() in {"CZ", "F3", "F4", "T3", "T4"} else np.nan),
        })
        report_lines.append(line)
    report_df = pd.DataFrame(rows)
    return report_lines, report_df

# -------------------- INTEGRATE PYRAMID MAPPING -------------------- #
def get_pyramid_mapping_section():
    """
    Retrieve the refined clinical and connectivity mappings from pyramid_model,
    formatting them as text.
    """
    section_lines = []
    section_lines.append("=== Refined Clinical Mapping (Pyramid Levels) ===")
    for level, mapping in pyramid_model.list_all_refined_mappings():
        section_lines.append(f"{mapping['level_name']}")
        section_lines.append("  EEG Patterns: " + ", ".join(mapping["eeg_patterns"]))
        section_lines.append("  Cognitive/Behavioral Implications: " + mapping["cognitive_behavior"])
        section_lines.append("  Protocols: " + mapping["protocols"])
        section_lines.append("")
    section_lines.append("=== EEG Connectivity Mapping ===")
    for level, mapping in pyramid_model.list_all_connectivity_mappings():
        section_lines.append(f"{mapping['level_name']}")
        section_lines.append("  EEG Patterns: " + ", ".join(mapping["eeg_patterns"]))
        section_lines.append("  Differentiators: " + mapping["differentiators"])
        section_lines.append("  Cognitive/Behavioral Implications: " + mapping["cognition_behavior"])
        section_lines.append("  Vigilance Stage: " + mapping["vigilance_stage"])
        section_lines.append("  Neurofeedback Targets: " + "; ".join(mapping["neurofeedback_targets"]))
        section_lines.append("")
    return "\n".join(section_lines)

# -------------------- MAIN REPORT GENERATION -------------------- #
def generate_full_clinical_report():
    # Discover EDF files in the current directory
    edf_dict = find_edf_files(Path.cwd())
    print("Found EDF files:", {k: str(v) for k, v in edf_dict.items()})
    if edf_dict["EO"] is None and edf_dict["EC"] is not None:
        print("Only an EC file was found; using it for both EO and EC processing.")
        edf_dict["EO"] = edf_dict["EC"]
    elif edf_dict["EC"] is None and edf_dict["EO"] is not None:
        print("Only an EO file was found; using it for both EO and EC processing.")
        edf_dict["EC"] = edf_dict["EO"]
    elif edf_dict["EO"] is None and edf_dict["EC"] is None:
        sys.exit("Error: No EDF files found in the current directory.")

    # Load EO and EC data using our updated load_data function
    raw_eo = load_data(edf_dict["EO"])
    raw_ec = load_data(edf_dict["EC"])
    
    # Compute band powers for each channel
    eo_results = compute_all_band_powers(raw_eo)
    ec_results = compute_all_band_powers(raw_ec)
    
    # Generate clinical report summary (text and CSV)
    report_lines, report_df = generate_site_report(eo_results, ec_results, raw_eo)
    
    # Retrieve pyramid mapping section from pyramid_model
    pyramid_section = get_pyramid_mapping_section()
    
    # Combine all sections into the full report
    full_report_lines = []
    full_report_lines.append("==============================================")
    full_report_lines.append("           Clinical EEG Report")
    full_report_lines.append("==============================================\n")
    full_report_lines.extend(report_lines)
    full_report_lines.append("\n")
    full_report_lines.append(pyramid_section)
    
    # Define output paths
    report_text_path = report_output_dir / "clinical_report.txt"
    report_csv_path = report_output_dir / "clinical_report.csv"
    
    # Write the full text report (UTF-8 for emoji support)
    with open(report_text_path, "w", encoding="utf-8") as f:
        for line in full_report_lines:
            f.write(line + "\n")
    
    # Export the CSV summary
    report_df.to_csv(report_csv_path, index=False)
    
    print("✅ Clinical report generated:")
    print(f"  Text Report: {report_text_path}")
    print(f"  CSV Summary: {report_csv_path}")

if __name__ == "__main__":
    generate_full_clinical_report()
