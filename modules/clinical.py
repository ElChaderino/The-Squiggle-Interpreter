"""
clinical.py

This module provides functions for site‐specific clinical analyses and detailed plotting.
It computes clinical metrics for each channel (site), saves CSV summaries, and generates
detailed per-site, per-band plots (including PSD overlays, waveform overlays, and difference plots)
for both EO and EC conditions.

Adjust normative values and thresholds as needed.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modules import processing, plotting

def compute_site_metrics(bp_EO, bp_EC):
    """
    Compute site-specific metrics based on band power for EO and EC.
    
    For each channel, compute:
      - Percentage change in Alpha power (EO → EC)
      - Theta/Beta ratio (using EO data)
    
    Parameters:
      bp_EO (dict): Dictionary of band powers for EO, structured as {channel: {band: power, ...}}
      bp_EC (dict): Same structure as bp_EO, for EC.
      
    Returns:
      dict: {channel: {metric_name: value, ...}, ...}
    """
    metrics = {}
    for ch in bp_EO.keys():
        metrics[ch] = {}
        # Percentage change in Alpha power:
        alpha_EO = bp_EO[ch].get("Alpha", 0)
        alpha_EC = bp_EC[ch].get("Alpha", 0)
        if alpha_EO != 0:
            metrics[ch]["Alpha_Change"] = ((alpha_EC - alpha_EO) / alpha_EO) * 100
        else:
            metrics[ch]["Alpha_Change"] = np.nan
        
        # Theta/Beta ratio using EO data:
        theta = bp_EO[ch].get("Theta", np.nan)
        beta = bp_EO[ch].get("Beta", np.nan)
        if beta and beta != 0:
            metrics[ch]["Theta_Beta_Ratio"] = theta / beta
        else:
            metrics[ch]["Theta_Beta_Ratio"] = np.nan
    return metrics

def save_site_metrics(metrics, output_path):
    """
    Save computed site-specific metrics to a CSV file.
    
    Parameters:
      metrics (dict): Output from compute_site_metrics.
      output_path (str): File path to save the CSV.
    """
    rows = []
    for ch, met in metrics.items():
        row = {"Channel": ch}
        row.update(met)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Clinical metrics saved to: {output_path}")

def generate_site_reports(bp_EO, bp_EC, output_dir):
    """
    Compute clinical metrics for each channel and save a CSV summary.
    
    Parameters:
      bp_EO (dict): Band power dictionary for EO.
      bp_EC (dict): Band power dictionary for EC.
      output_dir (str): Directory where the CSV will be saved.
    """
    metrics = compute_site_metrics(bp_EO, bp_EC)
    csv_path = os.path.join(output_dir, "clinical_metrics.csv")
    save_site_metrics(metrics, csv_path)

def generate_full_site_reports(raw_eo, raw_ec, output_dir):
    """
    Generate detailed per-site, per-band plots for EO vs. EC.
    
    For each channel (site) and each frequency band (from processing.BANDS), this function:
      - Creates a subfolder for the channel.
      - Generates a PSD overlay plot comparing EO and EC.
      - Generates a waveform overlay plot comparing EO and EC.
      - Generates a difference bar plot comparing EO vs. EC band power.
      
    All plots are generated in dark mode and saved in an organized folder structure.
    
    Parameters:
      raw_eo (mne.io.Raw): Raw data for Eyes Open.
      raw_ec (mne.io.Raw): Raw data for Eyes Closed.
      output_dir (str): Base directory where per-site plots will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    channels = raw_eo.ch_names
    sfreq = raw_eo.info['sfreq']
    
    for ch in channels:
        # Create a folder for each channel/site.
        ch_folder = os.path.join(output_dir, ch)
        psd_folder = os.path.join(ch_folder, "PSD_Overlay")
        wave_folder = os.path.join(ch_folder, "Waveform_Overlay")
        diff_folder = os.path.join(ch_folder, "Difference")
        os.makedirs(psd_folder, exist_ok=True)
        os.makedirs(wave_folder, exist_ok=True)
        os.makedirs(diff_folder, exist_ok=True)
        
        # Extract signals for this channel (convert to microvolts).
        eo_sig = raw_eo.get_data(picks=[ch])[0] * 1e6
        ec_sig = raw_ec.get_data(picks=[ch])[0] * 1e6
        
        # Loop through each frequency band defined in processing.BANDS.
        for band_name, band_range in processing.BANDS.items():
            # Generate PSD overlay plot.
            fig_psd = plotting.plot_band_psd_overlay(eo_sig, ec_sig, sfreq, band_range, ch, band_name, colors=("cyan", "magenta"))
            psd_path = os.path.join(psd_folder, f"{ch}_PSD_{band_name}.png")
            fig_psd.savefig(psd_path, facecolor='black')
            plt.close(fig_psd)
            
            # Generate waveform overlay plot.
            fig_wave = plotting.plot_band_waveform_overlay(eo_sig, ec_sig, sfreq, band_range, ch, band_name, colors=("cyan", "magenta"), epoch_length=10)
            wave_path = os.path.join(wave_folder, f"{ch}_Waveform_{band_name}.png")
            fig_wave.savefig(wave_path, facecolor='black')
            plt.close(fig_wave)
            
            # Generate a difference bar plot comparing EO vs. EC band power.
            power_eo = processing.compute_band_power(eo_sig, sfreq, band_range)
            power_ec = processing.compute_band_power(ec_sig, sfreq, band_range)
            fig_diff, ax = plt.subplots(figsize=(4, 4), facecolor='black')
            ax.bar(["EO", "EC"], [power_eo, power_ec], color=["cyan", "magenta"])
            ax.set_title(f"{ch} {band_name} Difference", color='white', fontsize=10)
            ax.set_ylabel("Power", color='white')
            ax.tick_params(colors='white')
            fig_diff.tight_layout()
            diff_path = os.path.join(diff_folder, f"{ch}_Difference_{band_name}.png")
            fig_diff.savefig(diff_path, facecolor='black')
            plt.close(fig_diff)
            
            print(f"Saved detailed plots for channel {ch}, band {band_name} in {ch_folder}")
