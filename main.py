#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main.py - The Squiggle Interpreter: Comprehensive EEG Analysis & Report Generation

This script performs the full EEG processing pipeline:
  • Discovers EDF files (using pathlib so filenames with spaces are supported) and groups them by subject.
  • For each subject, loads EEG data (removing "-LE", applying the standard 10–20 montage, average referencing,
    with optional current source density transform).
  • Computes band powers, topomaps, ERP, coherence, TFR, ICA, source localization, etc.
  • Generates detailed clinical reports (text, CSV, and interactive HTML) including refined clinical and connectivity mappings.
  • Optionally runs extension scripts and displays a live EEG simulation.
  • Optionally exports EDF data metrics to CSV.

Dependencies: mne, numpy, pandas, matplotlib, argparse, pathlib, rich, etc.
Ensure that modules/clinical_report.py, modules/pyramid_model.py, and modules/data_to_csv.py are present.
"""

import os
import sys
import threading
import time
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import mne
import argparse
import signal
from pathlib import Path

# New module imports
from modules import clinical_report    # Clinical report module integrating pyramid mappings
from modules import pyramid_model      # Pyramid mapping module
from modules import data_to_csv        # Module for EDF-to-CSV conversion
from modules import phenotype
from modules.phenotype import classify_eeg_profile

from modules.vigilance import plot_vigilance_hypnogram
from modules import io_utils, processing, plotting, report, clinical, vigilance
from mne.io.constants import FIFF
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from scipy.stats import zscore, pearsonr
import pandas as pd

# Global stop event for live display
stop_event = threading.Event()

# Try to import psd_welch; if unavailable, fall back to psd_array_welch.
try:
    from mne.time_frequency import psd_welch
except ImportError:
    from mne.time_frequency import psd_array_welch as psd_welch

# ---------------- Robust Z-Score Functions ----------------
def robust_mad(x, constant=1.4826, max_iter=10, tol=1e-3):
    """
    Compute a robust MAD (Median Absolute Deviation) with iterative outlier rejection.
    Returns a scaled MAD (to be comparable to std) and the median.
    """
    x = np.asarray(x)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    for _ in range(max_iter):
        mask = np.abs(x - med) <= 3 * mad
        new_med = np.median(x[mask])
        new_mad = np.median(np.abs(x[mask] - new_med))
        if np.abs(new_med - med) < tol and np.abs(new_mad - mad) < tol:
            break
        med, mad = new_med, new_mad
    return mad * constant, med

def robust_zscore(x, use_iqr=False):
    """
    Compute robust z-scores using the median and either MAD or IQR.
    """
    x = np.asarray(x)
    med = np.median(x)
    if use_iqr:
        q75, q25 = np.percentile(x, [75, 25])
        iqr = q75 - q25
        scale = iqr if iqr != 0 else 1.0
    else:
        scale, med = robust_mad(x)
        if scale == 0:
            scale = 1.0
    return (x - med) / scale

def compute_bandpower_robust_zscores(raw, bands=None, fmin=1, fmax=40, n_fft=2048, use_iqr=False):
    """
    Compute robust z-scores of log bandpower for each channel.
    """
    if bands is None:
        bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'SMR': (13, 15),
            'beta': (16, 28),
            'gamma': (29, 30)
        }
    psds, freqs = psd_welch(raw.get_data(), raw.info['sfreq'], fmin=fmin, fmax=fmax, n_fft=n_fft, verbose=False)
    psds_db = 10 * np.log10(psds)
    robust_features = {}
    for band, (low, high) in bands.items():
        band_mask = (freqs >= low) & (freqs <= high)
        band_power = psds_db[:, band_mask].mean(axis=1)
        robust_features[band] = robust_zscore(band_power, use_iqr=use_iqr)
    return robust_features

def load_clinical_outcomes(csv_file, n_channels):
    """
    Load clinical outcomes from a CSV file.
    Expects a column named 'outcome'. If the file is not found or errors occur,
    returns a dummy vector.
    """
    try:
        df = pd.read_csv(csv_file)
        outcomes = df['outcome'].values
        if len(outcomes) < n_channels:
            outcomes = np.pad(outcomes, (0, n_channels - len(outcomes)), mode='constant')
        return outcomes[:n_channels]
    except Exception as e:
        print("Could not load clinical outcomes from CSV:", e)
        return np.random.rand(n_channels)

def compare_zscores(standard_z, robust_z, clinical_outcomes):
    """
    Compare standard z-scores (using mean/std) with robust z-scores via Pearson correlation
    against clinical outcome data.
    """
    for band in standard_z.keys():
        r_std, p_std = pearsonr(standard_z[band], clinical_outcomes)
        r_rob, p_rob = pearsonr(robust_z[band], clinical_outcomes)
        print(f"Band {band}:")
        print(f"  Standard z-score: r = {r_std:.3f}, p = {p_std:.3f}")
        print(f"  Robust z-score  : r = {r_rob:.3f}, p = {p_rob:.3f}")

# ---------------- End Robust Z-Score Functions ----------------

# --- Utility: Group EDF Files by Subject ---
def find_subject_edf_files(directory):
    """
    Find and group EDF files by subject.
    Assumes filenames are in the format: <subjectID>eo.edf and <subjectID>ec.edf.
    """
    edf_files = [f for f in os.listdir(directory) if f.lower().endswith('.edf')]
    subjects = {}
    for f in edf_files:
        f_lower = f.lower()
        subject_id = f_lower[:2]
        if subject_id not in subjects:
            subjects[subject_id] = {"EO": None, "EC": None}
        if "eo" in f_lower:
            subjects[subject_id]["EO"] = f
        elif "ec" in f_lower:
            subjects[subject_id]["EC"] = f
    return subjects

def live_eeg_display(stop_event, update_interval=1.2):
    """
    Display a simulated live EEG waveform in the terminal using rich.
    """
    def generate_eeg_wave(num_points=80):
        x = np.linspace(0, 4 * np.pi, num_points)
        wave = np.sin(x) + np.random.normal(0, 0.3, size=num_points)
        gradient = " .:-=+*#%@"
        norm_wave = (wave - wave.min()) / (wave.max() - wave.min() + 1e-10)
        indices = (norm_wave * (len(gradient) - 1)).astype(int)
        return "".join(gradient[i] for i in indices)
    def get_random_quote():
        quotes = [
            "The Dude abides.",
            "That rug really tied the room together.",
            "Yeah, well, you know, that's just, like, your opinion, man.",
            "Sometimes you eat the bear, and sometimes, well, the bear eats you.",
            "Watch the squiggles, man. They're pure EEG poetry.",
            "This aggression will not stand... not even in beta spindles.",
            "Don’t cross the streams, Walter. I’m seeing delta in my alpha, man.",
            "Calmer than you are? My frontal lobes are lighting up like a bowling alley.",
            "Smokey, this is not 'Nam. This is neurofeedback. There are protocols.",
            "Obviously you’re not a golfer, or you’d know theta doesn’t spike like that."
        ]
        return np.random.choice(quotes)
    console = Console()
    with Live(refresh_per_second=10, console=console) as live:
        while not stop_event.is_set():
            line = generate_eeg_wave(os.get_terminal_size().columns - 4)
            quote = get_random_quote()
            text = f"[bold green]{line}[/bold green]"
            if quote:
                text += f"\n[bold red]{quote}[/bold red]\n"
            text += f"[bold blue]{line[::-1]}[/bold blue]"
            panel = Panel(text, title="Live EEG Display", subtitle="Simulated Waveform", style="white on black")
            live.update(panel)
            time.sleep(update_interval)

# Signal handler for graceful exit
def sigint_handler(signum, frame):
    print("SIGINT received, stopping gracefully...")
    stop_event.set()
    sys.exit(0)

# ---------------- Main Processing Pipeline ----------------
def main():
    parser = argparse.ArgumentParser(
        prog='The Squiggle Interpreter',
        description='Comprehensive EEG Analysis & Clinical Report Generation'
    )
    parser.add_argument('--csd', help="Use current source density (CSD) for graphs? (y/n)")
    parser.add_argument('--zscore', help="Z-score normalization method: 1: Standard, 2: Robust (MAD), 3: Robust (IQR), 4: Published Norms")
    parser.add_argument('--report', help="Generate full clinical report? (y/n)")
    # Options for CSV export:
    parser.add_argument('--csv', action='store_true', help="Export EDF data metrics to CSV")
    parser.add_argument('--edf', help="Path to an EDF file for CSV export")
    parser.add_argument('--epoch_length', type=float, default=2.0, help="Epoch length (in seconds) for CSV export (default: 2.0)")
    parser.add_argument('--output_csv', help="Output CSV file path for CSV export")
    
    args = parser.parse_args()
    
    # Prompt for missing parameters
    if args.csd is None:
        args.csd = input("Use current source density (CSD) for graphs? (y/n, default: n): ") or "n"
    if args.zscore is None:
        print("Choose z-score normalization method:")
        print("  1: Standard (mean/std)")
        print("  2: Robust (MAD-based)")
        print("  3: Robust (IQR-based)")
        print("  4: Published Norms (adult norms)")
        args.zscore = input("Enter choice (default: 1): ") or "1"
    if args.report is None:
        rep = input("Generate full clinical report? (y/n, default: y): ") or "y"
        args.report = rep.lower()
    
    # Process CSV export if requested
    if args.csv:
        if not args.edf or not args.output_csv:
            print("For CSV export, please provide both --edf and --output_csv arguments.")
            sys.exit(1)
        data_to_csv.process_edf_to_csv(args.edf, args.epoch_length, args.output_csv)
        sys.exit(0)
    
    project_dir = os.getcwd()
    overall_output_dir = os.path.join(project_dir, "outputs")
    os.makedirs(overall_output_dir, exist_ok=True)
    
    use_csd_for_graphs = True if args.csd.lower() == "y" else False
    print(f"Using CSD for graphs: {use_csd_for_graphs}")
    
    method_choice = args.zscore
    if method_choice == "4":
        published_norm_stats = {
            "Alpha": {"median": 20.0, "mad": 4.0},
            "Theta": {"median": 16.0, "mad": 3.5},
            "Delta": {"median": 22.0, "mad": 5.0},
            "SMR": {"median": 7.0, "mad": 1.5},
            "Beta": {"median": 6.0, "mad": 1.8},
            "HighBeta": {"median": 4.0, "mad": 1.2}
        }
        print("Using published normative values for adult EEG (published_norm_stats).")
    else:
        published_norm_stats = None
    
    subject_edf_groups = find_subject_edf_files(project_dir)
    print("Found subject EDF files:", subject_edf_groups)
    
    for subject, files in subject_edf_groups.items():
        subject_folder = os.path.join(overall_output_dir, subject)
        os.makedirs(subject_folder, exist_ok=True)
        
        # Define output subfolders
        folders = {
            "topomaps_eo": os.path.join(subject_folder, "topomaps", "EO"),
            "topomaps_ec": os.path.join(subject_folder, "topomaps", "EC"),
            "waveforms_eo": os.path.join(subject_folder, "waveforms", "EO"),
            "erp": os.path.join(subject_folder, "erp"),
            "coherence_eo": os.path.join(subject_folder, "coherence", "EO"),
            "coherence_ec": os.path.join(subject_folder, "coherence", "EC"),
            "zscore_eo": os.path.join(subject_folder, "zscore", "EO"),
            "zscore_ec": os.path.join(subject_folder, "zscore", "EC"),
            "tfr_eo": os.path.join(subject_folder, "tfr", "EO"),
            "tfr_ec": os.path.join(subject_folder, "tfr", "EC"),
            "ica_eo": os.path.join(subject_folder, "ica", "EO"),
            "source": os.path.join(subject_folder, "source_localization"),
            "detailed": os.path.join(subject_folder, "detailed_site_plots"),
            "vigilance": os.path.join(subject_folder, "vigilance")
        }
        for folder in folders.values():
            os.makedirs(folder, exist_ok=True)
        
        # Start live EEG display (optional)
        live_thread = threading.Thread(target=live_eeg_display, args=(stop_event,))
        live_thread.start()
        signal.signal(signal.SIGINT, sigint_handler)
        
        # Select EDF files for EO and EC (if one is missing, use the other)
        eo_file = files["EO"] if files["EO"] else files["EC"]
        ec_file = files["EC"] if files["EC"] else files["EO"]
        print(f"Subject {subject}: EO file: {eo_file}, EC file: {ec_file}")
        
        raw_eo = io_utils.load_eeg_data(os.path.join(project_dir, eo_file), use_csd=False)
        raw_ec = io_utils.load_eeg_data(os.path.join(project_dir, ec_file), use_csd=False)
        print("Loaded data for subject", subject)
        
        # Generate full clinical report if requested
        if args.report.lower() == "y":
            clinical_report.generate_full_clinical_report()
        
        # Compute band power metrics for clinical reports
        bp_eo = processing.compute_all_band_powers(raw_eo)
        bp_ec = processing.compute_all_band_powers(raw_ec)
        print(f"Subject {subject} - Computed band powers for EO channels:", list(bp_eo.keys()))
        print(f"Subject {subject} - Computed band powers for EC channels:", list(bp_ec.keys()))
        
        # Generate clinical site reports (text and CSV)
        clinical.generate_site_reports(bp_eo, bp_ec, subject_folder)

        # --- Phenotype Classification ---
        from modules import phenotype
        from modules.feature_extraction import extract_classification_features


        features = extract_classification_features(raw_eo, [])
        phenotype_results = classify_eeg_profile(features)

        phenotype_report_path = os.path.join(subject_folder, f"{subject}_phenotype.txt")
        with open(phenotype_report_path, "w", encoding="utf-8") as f:
            f.write("Phenotype Classification Results\n")
            f.write("===============================\n")
            for k, v in phenotype_results.items():
                f.write(f"{k}: {v}\n")

        # Append phenotype to main clinical report .txt file
        clinical_txt_path = os.path.join(subject_folder, f"{subject}_clinical_report.txt")
        with open(clinical_txt_path, "a", encoding="utf-8") as f:
            f.write("\n\nPhenotype Classification Results\n")
            f.write("===============================\n")
            for k, v in phenotype_results.items():
                f.write(f"{k}: {v}\n")
        
        
        # Compute standard bandpower features for z-score comparisons
        psds, freqs = psd_welch(raw_eo.get_data(), raw_eo.info['sfreq'], fmin=1, fmax=40, n_fft=2048, verbose=False)
        psds_db = 10 * np.log10(psds)
        default_bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'beta': (12, 30),
            'gamma': (30, 40)
        }
        standard_features = {}
        for band, (low, high) in default_bands.items():
            band_mask = (freqs >= low) & (freqs <= high)
            band_power = psds_db[:, band_mask].mean(axis=1)
            standard_features[band] = zscore(band_power)
        
        if method_choice == "1":
            chosen_features = standard_features
            print("Using standard z-scores (mean/std) for bandpower features.")
        elif method_choice == "2":
            chosen_features = compute_bandpower_robust_zscores(raw_eo, bands=default_bands, use_iqr=False)
            print("Using robust z-scores (MAD-based) for bandpower features.")
        elif method_choice == "3":
            chosen_features = compute_bandpower_robust_zscores(raw_eo, bands=default_bands, use_iqr=True)
            print("Using robust z-scores (IQR-based) for bandpower features.")
        elif method_choice == "4":
            chosen_features = {}
            zscore_maps_eo = processing.compute_all_zscore_maps(raw_eo, published_norm_stats, epoch_len_sec=2.0)
            for band in default_bands.keys():
                chosen_features[band] = zscore_maps_eo.get(band, np.array([]))
            print("Using published norms for z-score normalization.")
        else:
            print("Invalid choice. Defaulting to standard z-scores.")
            chosen_features = standard_features
        
        clinical_csv = os.path.join(project_dir, "clinical_outcomes.csv")
        clinical_outcomes = load_clinical_outcomes(clinical_csv, raw_eo.info['nchan'])
        print("Comparing standard vs. chosen z-score method:")
        compare_zscores(standard_features, chosen_features, clinical_outcomes)
        
        # Optionally apply CSD for graphs
        if use_csd_for_graphs:
            raw_eo_csd = raw_eo.copy().load_data()
            raw_ec_csd = raw_ec.copy().load_data()
            try:
                raw_eo_csd = mne.preprocessing.compute_current_source_density(raw_eo_csd)
                print("CSD applied for graphs (EO).")
            except Exception as e:
                print("CSD for graphs (EO) failed:", e)
                raw_eo_csd = raw_eo
            try:
                raw_ec_csd = mne.preprocessing.compute_current_source_density(raw_ec_csd)
                print("CSD applied for graphs (EC).")
            except Exception as e:
                print("CSD for graphs (EC) failed:", e)
                raw_ec_csd = raw_ec
        else:
            raw_eo_csd = raw_eo
            raw_ec_csd = raw_ec
        
        # Process vigilance: compute states and save hypnogram
        vigilance_states = vigilance.compute_vigilance_states(raw_eo, epoch_length=2.0)
        fig = vigilance.plot_vigilance_hypnogram(vigilance_states, epoch_length=2.0)
        hypno_path = os.path.join(folders["vigilance"], "vigilance_hypnogram.png")
        fig.savefig(hypno_path, facecolor='black')
        plt.close(fig)
        print(f"Saved vigilance hypnogram to {hypno_path}")
        for t, stage in vigilance_states:
            print(f"{t:5.1f}s: {stage}")
        try:
            fig_hypno = vigilance.plot_vigilance_hypnogram(vigilance_states, epoch_length=2.0)
            vigilance_hypno_path = os.path.join(folders["vigilance"], "vigilance_hypnogram.png")
            fig_hypno.savefig(vigilance_hypno_path, facecolor='black')
            plt.close(fig_hypno)
            print(f"Saved vigilance hypnogram to {vigilance_hypno_path}")
        except AttributeError:
            print("Error: 'plot_vigilance_hypnogram' not found in modules.vigilance.")
        
        # Process topomaps for EO
        band_list = list(processing.BANDS.keys())
        for band in band_list:
            abs_eo = [bp_eo[ch][band] for ch in raw_eo.ch_names]
            rel_eo = []
            for ch in raw_eo.ch_names:
                total_power = sum(bp_eo[ch][b] for b in band_list)
                rel_eo.append(bp_eo[ch][band] / total_power if total_power else 0)
            fig_topo_eo = plotting.plot_topomap_abs_rel(abs_eo, rel_eo, raw_eo.info, band, "EO")
            eo_topo_path = os.path.join(folders["topomaps_eo"], f"topomap_{band}.png")
            fig_topo_eo.savefig(eo_topo_path, facecolor='black')
            plt.close(fig_topo_eo)
            print(f"Subject {subject}: Saved EO topomap for {band} to {eo_topo_path}")
        
        # Process waveform grids for EO
        global_waveforms = {}
        data_eo = raw_eo.get_data() * 1e6
        sfreq = raw_eo.info['sfreq']
        for band, band_range in processing.BANDS.items():
            wf_fig = plotting.plot_waveform_grid(data_eo, raw_eo.ch_names, sfreq, band=band_range, epoch_length=10)
            wf_path = os.path.join(folders["waveforms_eo"], f"waveforms_{band}.png")
            wf_fig.savefig(wf_path, facecolor='black')
            plt.close(wf_fig)
            global_waveforms[band] = os.path.basename(wf_path)
            print(f"Subject {subject}: Saved EO waveform grid for {band} to {wf_path}")
        
        # Process ERP for EO and EC
        erp_eo_fig = processing.compute_pseudo_erp(raw_eo)
        erp_eo_path = os.path.join(folders["erp"], "erp_EO.png")
        erp_eo_fig.savefig(erp_eo_path, facecolor='black')
        plt.close(erp_eo_fig)
        print(f"Subject {subject}: Saved ERP EO to {erp_eo_path}")
        
        erp_ec_fig = processing.compute_pseudo_erp(raw_ec)
        erp_ec_path = os.path.join(folders["erp"], "erp_EC.png")
        erp_ec_fig.savefig(erp_ec_path, facecolor='black')
        plt.close(erp_ec_fig)
        print(f"Subject {subject}: Saved ERP EC to {erp_ec_path}")
        
        # Process coherence matrices for EO and EC
        coherence_maps = {"EO": {}, "EC": {}}
        for band in band_list:
            band_range = processing.BANDS[band]
            coh_matrix_eo = processing.compute_coherence_matrix(raw_eo.get_data() * 1e6, sfreq, band_range, nperseg=int(sfreq*2))
            fig_coh_eo = plotting.plot_coherence_matrix(coh_matrix_eo, raw_eo.ch_names)
            coh_path_eo = os.path.join(folders["coherence_eo"], f"coherence_{band}.png")
            fig_coh_eo.savefig(coh_path_eo, facecolor='black')
            plt.close(fig_coh_eo)
            coherence_maps["EO"][band] = os.path.basename(coh_path_eo)
            print(f"Subject {subject}: Saved EO coherence ({band}) to {coh_path_eo}")
            
            coh_matrix_ec = processing.compute_coherence_matrix(raw_ec.get_data() * 1e6, sfreq, band_range, nperseg=int(sfreq*2))
            fig_coh_ec = plotting.plot_coherence_matrix(coh_matrix_ec, raw_ec.ch_names)
            coh_path_ec = os.path.join(folders["coherence_ec"], f"coherence_{band}.png")
            fig_coh_ec.savefig(coh_path_ec, facecolor='black')
            plt.close(fig_coh_ec)
            coherence_maps["EC"][band] = os.path.basename(coh_path_ec)
            print(f"Subject {subject}: Saved EC coherence ({band}) to {coh_path_ec}")
        
        # Process robust z-score topomaps for EO and EC
        if published_norm_stats is not None:
            norm_stats = published_norm_stats
        else:
            norm_stats = {
                "Alpha": {"median": 18.0, "mad": 6.0},
                "Theta": {"median": 15.0, "mad": 5.0},
                "Delta": {"median": 20.0, "mad": 7.0},
                "SMR": {"median": 6.0, "mad": 2.0},
                "Beta": {"median": 5.0, "mad": 2.0},
                "HighBeta": {"median": 3.5, "mad": 1.5}
            }
        zscore_maps_eo = processing.compute_all_zscore_maps(raw_eo, norm_stats, epoch_len_sec=2.0)
        zscore_maps_ec = processing.compute_all_zscore_maps(raw_ec, norm_stats, epoch_len_sec=2.0)
        zscore_images_eo = {}
        zscore_images_ec = {}
        for band in band_list:
            if band in zscore_maps_eo and zscore_maps_eo[band] is not None:
                fig_zscore_eo = plotting.plot_zscore_topomap(zscore_maps_eo[band], raw_eo.info, band, "EO")
                zscore_path_eo = os.path.join(folders["zscore_eo"], f"zscore_{band}.png")
                fig_zscore_eo.savefig(zscore_path_eo, facecolor='black')
                plt.close(fig_zscore_eo)
                zscore_images_eo[band] = os.path.basename(zscore_path_eo)
                print(f"Subject {subject}: Saved EO z-score ({band}) to {zscore_path_eo}")
            if band in zscore_maps_ec and zscore_maps_ec[band] is not None:
                fig_zscore_ec = plotting.plot_zscore_topomap(zscore_maps_ec[band], raw_ec.info, band, "EC")
                zscore_path_ec = os.path.join(folders["zscore_ec"], f"zscore_{band}.png")
                fig_zscore_ec.savefig(zscore_path_ec, facecolor='black')
                plt.close(fig_zscore_ec)
                zscore_images_ec[band] = os.path.basename(zscore_path_ec)
                print(f"Subject {subject}: Saved EC z-score ({band}) to {zscore_path_ec}")
        
        # Process TFR maps for EO and EC
        n_cycles = 2.0
        tfr_maps_eo = processing.compute_all_tfr_maps(raw_eo, n_cycles, tmin=0.0, tmax=4.0)
        tfr_maps_ec = processing.compute_all_tfr_maps(raw_ec, n_cycles, tmin=0.0, tmax=4.0)
        tfr_images_eo = {}
        tfr_images_ec = {}
        for band in band_list:
            if band in tfr_maps_eo and tfr_maps_eo[band] is not None:
                fig_tfr_eo = plotting.plot_tfr(tfr_maps_eo[band], picks=0)
                tfr_path_eo = os.path.join(folders["tfr_eo"], f"tfr_{band}.png")
                fig_tfr_eo.savefig(tfr_path_eo, facecolor='black')
                plt.close(fig_tfr_eo)
                tfr_images_eo[band] = os.path.basename(tfr_path_eo)
                print(f"Subject {subject}: Saved TFR EO ({band}) to {tfr_path_eo}")
            else:
                print(f"Subject {subject}: TFR EO for {band} was not computed.")
            if band in tfr_maps_ec and tfr_maps_ec[band] is not None:
                fig_tfr_ec = plotting.plot_tfr(tfr_maps_ec[band], picks=0)
                tfr_path_ec = os.path.join(folders["tfr_ec"], f"tfr_{band}.png")
                fig_tfr_ec.savefig(tfr_path_ec, facecolor='black')
                plt.close(fig_tfr_ec)
                tfr_images_ec[band] = os.path.basename(tfr_path_ec)
                print(f"Subject {subject}: Saved TFR EC ({band}) to {tfr_path_ec}")
            else:
                print(f"Subject {subject}: TFR EC for {band} was not computed.")
        
        # Process ICA for EO
        ica = processing.compute_ica(raw_eo)
        fig_ica = plotting.plot_ica_components(ica, raw_eo)
        ica_path = os.path.join(folders["ica_eo"], "ica_EO.png")
        fig_ica.savefig(ica_path, facecolor='black')
        plt.close(fig_ica)
        print(f"Subject {subject}: Saved ICA EO to {ica_path}")
        
        # Process source localization for EO and EC
        raw_source_eo = raw_eo.copy()
        raw_source_ec = raw_ec.copy()
        raw_source_eo.set_eeg_reference("average", projection=False)
        raw_source_ec.set_eeg_reference("average", projection=False)
        print("EEG channels for source localization (EO):", mne.pick_types(raw_source_eo.info, meg=False, eeg=True))
        
        fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
        subjects_dir = os.path.dirname(fs_dir)
        subject_fs = "fsaverage"
        src = mne.setup_source_space(subject_fs, spacing="oct6", subjects_dir=subjects_dir, add_dist=False)
        conductivity = (0.3, 0.006, 0.3)
        bem_model = mne.make_bem_model(subject=subject_fs, ico=4, conductivity=conductivity, subjects_dir=subjects_dir)
        bem_solution = mne.make_bem_solution(bem_model)
        
        fwd_eo = mne.make_forward_solution(raw_source_eo.info, trans="fsaverage", src=src,
                                           bem=bem_solution, eeg=True, meg=False, verbose=False)
        fwd_ec = mne.make_forward_solution(raw_source_ec.info, trans="fsaverage", src=src,
                                           bem=bem_solution, eeg=True, meg=False, verbose=False)
        
        events_eo = mne.make_fixed_length_events(raw_eo, duration=2.0)
        epochs_eo = mne.Epochs(raw_eo, events_eo, tmin=-0.1, tmax=0.4, baseline=(None, 0),
                                preload=True, verbose=False)
        cov_eo = mne.compute_covariance(epochs_eo, tmax=0., method="empirical", verbose=False)
        
        inv_op_eo = processing.compute_inverse_operator(raw_source_eo, fwd_eo, cov_eo)
        inv_op_ec = processing.compute_inverse_operator(raw_source_ec, fwd_ec, cov_eo)
        
        source_methods = {"LORETA": "MNE", "sLORETA": "sLORETA", "eLORETA": "eLORETA"}
        source_localization = {"EO": {}, "EC": {}}
        for cond, raw_data, inv_op in [("EO", raw_eo, inv_op_eo), ("EC", raw_ec, inv_op_ec)]:
            for band in band_list:
                band_range = processing.BANDS[band]
                raw_band = raw_data.copy().filter(band_range[0], band_range[1], verbose=False)
                events = mne.make_fixed_length_events(raw_band, duration=2.0)
                epochs = mne.Epochs(raw_band, events, tmin=-0.1, tmax=0.4, baseline=(None, 0),
                                    preload=True, verbose=False)
                evoked = epochs.average()
                cond_folder = os.path.join(folders["source"], cond)
                os.makedirs(cond_folder, exist_ok=True)
                for method, method_label in source_methods.items():
                    try:
                        stc = processing.compute_source_localization(evoked, inv_op, lambda2=1.0/9.0, method=method_label)
                        fig_src = plotting.plot_source_estimate(stc, view="lateral", time_point=0.1, subjects_dir=subjects_dir)
                        src_filename = f"source_{cond}_{method}_{band}.png"
                        src_path = os.path.join(cond_folder, src_filename)
                        fig_src.savefig(src_path, dpi=150, bbox_inches="tight", facecolor="black")
                        plt.close(fig_src)
                        source_localization[cond].setdefault(band, {})[method] = cond + "/" + src_filename
                        print(f"Subject {subject}: Saved {method} source localization for {cond} {band} to {src_path}")
                    except Exception as e:
                        print(f"Error computing source localization for {cond} {band} with {method}: {e}")
        
        print("Source Localization dictionary:")
        print(source_localization)
        
        # Generate detailed per-site reports
        from modules.clinical import generate_full_site_reports
        generate_full_site_reports(raw_eo, raw_ec, folders["detailed"])
        
        # Build site dictionary for report
        site_list = raw_eo.ch_names
        site_dict = {}
        for site in site_list:
            site_dict[site] = {}
            for b in band_list:
                psd_path = os.path.join("detailed_site_plots", site, "PSD_Overlay", f"{site}_PSD_{b}.png")
                wave_path = os.path.join("detailed_site_plots", site, "Waveform_Overlay", f"{site}_Waveform_{b}.png")
                site_dict[site][b] = {"psd": psd_path, "wave": wave_path}
        
        # Build final report data dictionary for HTML report
        report_data = {
            "global_topomaps": {
                "EO": {b: os.path.basename(os.path.join(folders["topomaps_eo"], f"topomap_{b}.png")) for b in band_list},
                "EC": {b: os.path.basename(os.path.join(folders["topomaps_ec"], f"topomap_{b}.png")) for b in band_list}
            },
            "global_waveforms": global_waveforms,
            "coherence": {
                "EO": coherence_maps["EO"],
                "EC": coherence_maps["EC"]
            },
            "global_erp": {
                "EO": os.path.basename(erp_eo_path),
                "EC": os.path.basename(erp_ec_path)
            },
            "zscore": {
                "EO": {b: os.path.basename(os.path.join(folders["zscore_eo"], f"zscore_{b}.png")) for b in band_list},
                "EC": {b: os.path.basename(os.path.join(folders["zscore_ec"], f"zscore_{b}.png")) for b in band_list}
            },
            "tfr": {
                "EO": {b: os.path.basename(os.path.join(folders["tfr_eo"], f"tfr_{b}.png")) for b in band_list},
                "EC": {b: os.path.basename(os.path.join(folders["tfr_ec"], f"tfr_{b}.png")) for b in band_list}
            },
            "ica": {
                "EO": os.path.basename(ica_path),
                "EC": ""
            },
            "source_localization": source_localization,
            "site_list": site_list,
            "band_list": band_list,
            "site_dict": site_dict,
            "global_topomaps_path": "topomaps",
            "global_waveforms_path": "waveforms",
            "coherence_path": "coherence",
            "global_erp_path": "erp",
            "tfr_path": "tfr",
            "ica_path": "ica",
            "sites_path": "",
            "source_path": "./source_localization"
        }
        
        subject_report_path = os.path.join(subject_folder, "eeg_report.html")
        report.build_html_report(report_data, subject_report_path)
        print(f"Subject {subject}: Generated interactive HTML report at {subject_report_path}")
        
        extension_script = os.path.join(project_dir, "extensions", "EEG_Extension.py")
        if os.path.exists(extension_script):
            print(f"Subject {subject}: Running extension script: {extension_script}")
            subprocess.run([sys.executable, extension_script])
        else:
            print(f"Subject {subject}: No extension script found in 'extensions' folder.")
        
        stop_event.set()
        live_thread.join()
        print("Live EEG display stopped for subject", subject)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, sigint_handler)
    main()
