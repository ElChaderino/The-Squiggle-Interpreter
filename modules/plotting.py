#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plotting.py

This module provides plotting functions for EEG data visualization in The Squiggle Interpreter.
It generates topomaps, waveform grids, coherence matrices, PSD overlays, waveform overlays,
z-score topomaps, TFR plots, ICA components, source estimates, and difference plots.

Key Features:
- Generates detailed site plots (PSD and waveform overlays) for EO vs. EC comparisons.
- Supports topomaps for absolute, relative, and instability power.
- Handles parallel plotting tasks with robust error handling.
- Ensures dark mode for all plots to match the HTML report theme.
- Provides fallback mechanisms to prevent pipeline crashes.
"""

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy import signal
from .config import BANDS, PLOT_CONFIG, CRITICAL_SITES, DETAILED_SITES
from .processing import compute_all_band_powers, compute_all_zscore_maps, compute_tfr, compute_pseudo_erp, compute_coherence_matrix, compute_ica
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def remove_overlapping_channels(info, tol=0.05):
    """
    Remove channels with overlapping positions to improve topomap clarity.
    
    Args:
        info (mne.Info): MNE Info object.
        tol (float): Tolerance for position comparison.
    
    Returns:
        tuple: (cleaned info, list of unique indices)
    """
    if not info or not info["chs"]:
        logger.warning("Cannot remove overlapping channels: Invalid info.")
        return info, []
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

def plot_topomap(values, info, title, cmap=None, clim=None, axes=None, show_colorbar=True):
    """
    Generic function to plot a topomap with customizable parameters.
    
    Args:
        values (array-like): Values to plot per channel.
        info (mne.Info): MNE Info with channel locations.
        title (str): Plot title.
        cmap (str, optional): Colormap name.
        clim (tuple, optional): Color limits (min, max).
        axes (matplotlib.axes.Axes, optional): Axes to plot on.
        show_colorbar (bool): Whether to show a colorbar.
    
    Returns:
        matplotlib.figure.Figure: The topomap figure, or None if invalid.
    """
    if not values or not info or not info["chs"]:
        logger.warning(f"Cannot plot topomap for {title}: Invalid input data.")
        return None
    values = np.array(values)
    if len(values) != len(info["chs"]):
        logger.warning(f"Data length mismatch for {title}: {len(values)} vs {len(info['chs'])}.")
        return None

    info_clean, sel_idx = remove_overlapping_channels(info)
    values_subset = values[sel_idx]
    cmap = cmap or PLOT_CONFIG["topomap_cmap"]
    standalone = axes is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='black')
    else:
        fig, ax = plt.gcf(), axes

    ax.set_facecolor('black')
    try:
        im, _ = mne.viz.plot_topomap(values_subset, info_clean, axes=ax, cmap=cmap, show=False)
        if clim:
            im.set_clim(*clim)
        elif "Difference" in title:
            max_abs = np.max(np.abs(values_subset))
            im.set_clim(-max_abs, max_abs)
        ax.set_title(title, color='white', fontsize=10)
        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', fraction=0.05, pad=0.08)
            cbar.ax.tick_params(colors='white')
        if standalone:
            fig.tight_layout()
        logger.debug(f"Generated topomap for {title}.")
        return fig
    except Exception as e:
        logger.error(f"Failed to plot topomap for {title}: {e}")
        return None

def plot_topomap_abs_rel(abs_vals, rel_vals, raw, band_name, cond_name, instability_vals=None):
    """
    Generate side-by-side topomaps for absolute, relative, and instability power.
    
    Args:
        abs_vals (array-like): Absolute power values per channel.
        rel_vals (array-like): Relative power values per channel.
        raw (mne.io.Raw): Raw EEG data.
        band_name (str): Name of the frequency band.
        cond_name (str): Condition label (e.g., "EO" or "EC").
        instability_vals (array-like, optional): Instability values per channel.
    
    Returns:
        matplotlib.figure.Figure: The generated topomap figure, or None if invalid.
    """
    if not abs_vals or not rel_vals or not raw or not raw.info["chs"]:
        logger.warning(f"Cannot plot topomap for {band_name} ({cond_name}): Invalid input data.")
        return None
    if instability_vals is not None and len(instability_vals) != len(raw.ch_names):
        logger.warning(f"Instability values length mismatch for {band_name} ({cond_name}). Skipping instability topomap.")
        instability_vals = None
    info = raw.info
    n_subplots = 3 if instability_vals is not None else 2
    fig, axes = plt.subplots(1, n_subplots, figsize=(PLOT_CONFIG["topomap_figsize"][0] * n_subplots, PLOT_CONFIG["topomap_figsize"][1]), facecolor='black')
    fig.patch.set_facecolor('black')
    axes = np.atleast_1d(axes)

    plot_topomap(abs_vals, info, f"{band_name} Abs Power ({cond_name})", cmap=PLOT_CONFIG["topomap_cmap"], axes=axes[0])
    plot_topomap(rel_vals, info, f"{band_name} Rel Power ({cond_name})", cmap=PLOT_CONFIG["topomap_cmap"], axes=axes[1])
    if instability_vals is not None:
        plot_topomap(instability_vals, info, f"{band_name} Instability ({cond_name})", cmap=PLOT_CONFIG["topomap_cmap"], axes=axes[2])
    
    fig.suptitle(f"Topomaps (Abs, Rel, Instability) - {band_name} ({cond_name})", color='white', fontsize=12)
    fig.tight_layout()
    logger.info(f"Generated topomap for {band_name} ({cond_name}).")
    return fig

def plot_waveform_grid(data, ch_names, sfreq, band=(8, 12), epoch_length=10):
    """
    Generate a grid of waveform plots for all channels for a given frequency band.
    
    Args:
        data (np.array): 2D array (n_channels x n_samples).
        ch_names (list): List of channel names.
        sfreq (float): Sampling frequency.
        band (tuple): Frequency band (fmin, fmax) for filtering.
        epoch_length (float): Duration (in seconds) of the plotted segment.
    
    Returns:
        matplotlib.figure.Figure: The generated waveform grid figure, or None if invalid.
    """
    if data is None or not ch_names or not sfreq or not np.isfinite(data).all():
        logger.warning("Cannot plot waveform grid: Invalid input data.")
        return None
    n_channels = data.shape[0]
    n_samples = int(sfreq * epoch_length)
    if n_samples > data.shape[1]:
        n_samples = data.shape[1]
    t = np.linspace(0, epoch_length, n_samples, endpoint=False)
    n_cols = PLOT_CONFIG["n_cols_waveform"]
    n_rows = int(np.ceil(n_channels / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(PLOT_CONFIG["waveform_figsize"][0], n_rows * 3.5), facecolor='black')
    fig.patch.set_facecolor('black')
    axes = np.atleast_2d(axes)

    for i in range(n_channels):
        ax = axes[i // n_cols, i % n_cols]
        ax.set_facecolor('black')
        try:
            filt = mne.filter.filter_data(data[i], sfreq, band[0], band[1], verbose=False)
            ax.plot(t, filt[:n_samples], color=PLOT_CONFIG["waveform_colors"][0], lw=1)
            ax.set_title(ch_names[i], color='white', fontsize=12, pad=15)
            ax.set_xlim([0, epoch_length])
            ax.tick_params(colors='white', labelsize=10)
            ax.set_xlabel("Time (s)", color='white', fontsize=10)
            ax.set_ylabel("Amplitude (µV)", color='white', fontsize=10)
        except Exception as e:
            logger.warning(f"Failed to plot waveform for channel {ch_names[i]}: {e}")
            ax.axis('off')

    for j in range(n_channels, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis('off')

    fig.suptitle(f"Waveform Grid ({band[0]}-{band[1]}Hz)", color='white', fontsize=16, y=1.02)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    logger.info(f"Generated waveform grid for band {band}.")
    return fig

def plot_coherence_matrix(coh_matrix, ch_names=None):
    """
    Generate a heatmap of the global coherence matrix with keys and axis labels.
    
    Args:
        coh_matrix (np.array): Coherence matrix.
        ch_names (list, optional): List of channel names.
    
    Returns:
        matplotlib.figure.Figure: The coherence heatmap figure, or None if invalid.
    """
    if coh_matrix is None or not coh_matrix.size:
        logger.warning("Cannot plot coherence matrix: Invalid input data.")
        return None
    fig, ax = plt.subplots(figsize=PLOT_CONFIG["coherence_figsize"], facecolor='black')
    ax.set_facecolor('black')
    try:
        cax = ax.imshow(coh_matrix, cmap=PLOT_CONFIG["topomap_cmap"], aspect='auto')
        ax.set_title("Global Coherence Matrix", color='white', fontsize=12)
        if ch_names and len(ch_names) == coh_matrix.shape[0]:
            ax.set_xticks(range(len(ch_names)))
            ax.set_xticklabels(ch_names, rotation=90, fontsize=8, color='white')
            ax.set_yticks(range(len(ch_names)))
            ax.set_yticklabels(ch_names, fontsize=8, color='white')
        else:
            ax.set_xlabel("Channels", color='white')
            ax.set_ylabel("Channels", color='white')
        cbar = plt.colorbar(cax, ax=ax)
        cbar.set_label("Coherence", color='white')
        cbar.ax.tick_params(colors='white')
        fig.tight_layout()
        logger.info("Generated coherence matrix heatmap.")
        return fig
    except Exception as e:
        logger.warning(f"Failed to plot coherence matrix: {e}")
        return None

def plot_band_psd_overlay(eo_sig, ec_sig, sfreq, band, ch_name, band_name, colors=None):
    """
    Plot PSD overlay for a given channel comparing EO vs. EC.
    
    Args:
        eo_sig, ec_sig (np.array): 1D signals for EO and EC.
        sfreq (float): Sampling frequency.
        band (tuple): Frequency band (fmin, fmax).
        ch_name (str): Channel name.
        band_name (str): Frequency band name.
        colors (tuple, optional): Colors for EO and EC.
    
    Returns:
        matplotlib.figure.Figure: The PSD overlay figure, or None if invalid.
    """
    if eo_sig is None or ec_sig is None or not sfreq:
        logger.warning(f"Cannot plot PSD overlay for {ch_name} {band_name}: Missing input data.")
        return None
    if len(eo_sig) < 256 or len(ec_sig) < 256:
        logger.warning(f"Insufficient samples for {ch_name} {band_name}: EO={len(eo_sig)}, EC={len(ec_sig)}.")
        return None

    # Replace non-finite values with zeros
    eo_sig = np.nan_to_num(eo_sig, nan=0.0, posinf=0.0, neginf=0.0)
    ec_sig = np.nan_to_num(ec_sig, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.isfinite(eo_sig).all() or not np.isfinite(ec_sig).all():
        logger.warning(f"Non-finite values detected in signals for {ch_name} {band_name} after cleaning.")
        return None

    colors = colors or PLOT_CONFIG["waveform_colors"]
    fmin, fmax = band
    try:
        # Increase nperseg for better frequency resolution (e.g., 2-second window)
        nperseg = min(int(sfreq * 2.0), len(eo_sig), len(ec_sig))  # Use 2-second window
        if nperseg < 256:
            nperseg = 256  # Minimum for welch
            eo_sig = np.pad(eo_sig, (0, max(0, nperseg - len(eo_sig))), mode='constant')
            ec_sig = np.pad(ec_sig, (0, max(0, nperseg - len(ec_sig))), mode='constant')
        freqs_eo, psd_eo = signal.welch(eo_sig, fs=sfreq, nperseg=nperseg)
        freqs_ec, psd_ec = signal.welch(ec_sig, fs=sfreq, nperseg=nperseg)

        # Log frequency bins for debugging
        logger.debug(f"Frequency bins for {ch_name} {band_name}: freqs_eo={freqs_eo[:10]}... (total {len(freqs_eo)} bins)")
        # Adjust frequency mask to include the nearest bins
        mask_eo = (freqs_eo >= fmin) & (freqs_eo <= fmax)
        mask_ec = (freqs_ec >= fmin) & (freqs_ec <= fmax)

        # Validate frequency mask
        if not np.any(mask_eo) or not np.any(mask_ec):
            logger.warning(f"No frequencies in range {band} for {ch_name} {band_name} (freqs_eo={freqs_eo}, freqs_ec={freqs_ec}).")
            # Fallback: Include the nearest frequency bins
            freqs_eo_indices = np.where((freqs_eo >= fmin - 0.5) & (freqs_eo <= fmax + 0.5))[0]
            freqs_ec_indices = np.where((freqs_ec >= fmin - 0.5) & (freqs_ec <= fmax + 0.5))[0]
            if len(freqs_eo_indices) == 0 or len(freqs_ec_indices) == 0:
                logger.warning(f"Fallback frequency mask also empty for {ch_name} {band_name}.")
                raise ValueError("No frequencies available even with fallback.")
            mask_eo = np.zeros_like(freqs_eo, dtype=bool)
            mask_ec = np.zeros_like(freqs_ec, dtype=bool)
            mask_eo[freqs_eo_indices] = True
            mask_ec[freqs_ec_indices] = True

        # Normalize PSD to total power in the band to reduce scaling issues
        psd_eo_band = psd_eo[mask_eo]
        psd_ec_band = psd_ec[mask_ec]
        total_power_eo = np.sum(psd_eo_band)
        total_power_ec = np.sum(psd_ec_band)
        if total_power_eo > 0:
            psd_eo_normalized = psd_eo_band / total_power_eo
        else:
            psd_eo_normalized = np.zeros_like(psd_eo_band)
        if total_power_ec > 0:
            psd_ec_normalized = psd_ec_band / total_power_ec
        else:
            psd_ec_normalized = np.zeros_like(psd_ec_band)

        # Convert to dB, avoiding log(0) issues
        psd_eo_db = 10 * np.log10(psd_eo_normalized + 1e-6)  # Use a larger constant to avoid negative infinities
        psd_ec_db = 10 * np.log10(psd_ec_normalized + 1e-6)

        # Log mean power for debugging
        logger.debug(f"Mean power for {ch_name} {band_name}: EO={total_power_eo:.2e} µV²/Hz, EC={total_power_ec:.2e} µV²/Hz")

        # Plot the PSD
        fig, ax = plt.subplots(figsize=PLOT_CONFIG["psd_figsize"], facecolor='black')
        ax.set_facecolor('black')
        ax.plot(freqs_eo[mask_eo], psd_eo_db, linestyle='-', color=colors[0], label='EO')
        ax.plot(freqs_ec[mask_ec], psd_ec_db, linestyle='--', color=colors[1], label='EC')
        ax.set_title(f"{ch_name} {band_name} PSD (Normalized)", color='white', fontsize=10)
        ax.set_xlabel("Frequency (Hz)", color='white')
        ax.set_ylabel("Relative Power (dB)", color='white')
        ax.tick_params(colors='white')

        # Dynamically set y-axis limits
        y_min = min(psd_eo_db.min(), psd_ec_db.min())
        y_max = max(psd_eo_db.max(), psd_ec_db.max())
        y_range = y_max - y_min
        if y_range > 0:
            ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
        else:
            ax.set_ylim(-10, 10)  # Default range if data is flat

        ax.legend(fontsize=8)
        fig.tight_layout()
        logger.info(f"Generated PSD overlay for {ch_name} {band_name}.")
        return fig
    except Exception as e:
        logger.warning(f"Failed to plot PSD overlay for {ch_name} {band_name}: {e}")
        # Fallback: Plot raw signal if PSD fails
        try:
            n_samples = min(len(eo_sig), len(ec_sig))
            t = np.linspace(0, n_samples / sfreq, n_samples, endpoint=False)
            fig, ax = plt.subplots(figsize=PLOT_CONFIG["psd_figsize"], facecolor='black')
            ax.set_facecolor('black')
            ax.plot(t, eo_sig[:n_samples], linestyle='-', color=colors[0], label='EO (Raw)')
            ax.plot(t, ec_sig[:n_samples], linestyle='--', color=colors[1], label='EC (Raw)')
            ax.set_title(f"{ch_name} {band_name} Raw Signal (Fallback)", color='white', fontsize=10)
            ax.set_xlabel("Time (s)", color='white')
            ax.set_ylabel("Amplitude (µV)", color='white')
            ax.tick_params(colors='white')
            ax.legend(fontsize=8)
            fig.tight_layout()
            logger.info(f"Generated fallback raw signal plot for {ch_name} {band_name}.")
            return fig
        except Exception as e2:
            logger.error(f"Failed to plot fallback raw signal for {ch_name} {band_name}: {e2}")
            return None
def plot_band_waveform_overlay(eo_sig, ec_sig, sfreq, band, ch_name, band_name, colors=None, epoch_length=10):
    """
    Plot waveform overlays for a given channel comparing EO vs. EC after band-pass filtering.
    
    Args:
        eo_sig, ec_sig (np.array): 1D signals for EO and EC.
        sfreq (float): Sampling frequency.
        band (tuple): Frequency band (fmin, fmax).
        ch_name (str): Channel name.
        band_name (str): Frequency band name.
        colors (tuple, optional): Colors for EO and EC.
        epoch_length (float): Duration (in seconds) for the plot.
    
    Returns:
        matplotlib.figure.Figure: The waveform overlay figure, or None if invalid.
    """
    if eo_sig is None or ec_sig is None or not sfreq:
        logger.warning(f"Cannot plot waveform overlay for {ch_name} {band_name}: Missing input data.")
        return None
    if len(eo_sig) < 32 or len(ec_sig) < 32:
        logger.warning(f"Insufficient samples for {ch_name} {band_name}: EO={len(eo_sig)}, EC={len(ec_sig)}.")
        return None
    # Replace non-finite values with zeros
    eo_sig = np.nan_to_num(eo_sig, nan=0.0, posinf=0.0, neginf=0.0)
    ec_sig = np.nan_to_num(ec_sig, nan=0.0, posinf=0.0, neginf=0.0)
    # Check for all-zero or invalid data
    if np.all(eo_sig == 0) or np.all(ec_sig == 0):
        logger.warning(f"Skipping waveform overlay for {ch_name} {band_name}: Data is all zeros.")
        return None
    logger.debug(f"EO data stats for {ch_name} {band_name}: mean={np.mean(eo_sig):.2e}, std={np.std(eo_sig):.2e}")
    logger.debug(f"EC data stats for {ch_name} {band_name}: mean={np.mean(ec_sig):.2e}, std={np.std(ec_sig):.2e}")
    colors = colors or PLOT_CONFIG["waveform_colors"]
    fmin, fmax = band
    try:
        # Validate frequency range
        if fmax > sfreq / 2:
            logger.warning(f"Frequency band {band_name} ({fmin}-{fmax} Hz) exceeds Nyquist frequency ({sfreq/2} Hz). Skipping.")
            return None
        filt_eo = mne.filter.filter_data(eo_sig, sfreq, fmin, fmax, verbose=False)
        filt_ec = mne.filter.filter_data(ec_sig, sfreq, fmin, fmax, verbose=False)
        n_samples = int(sfreq * epoch_length)
        if n_samples > len(filt_eo):
            n_samples = len(filt_eo)
        if n_samples > len(filt_ec):
            n_samples = len(filt_ec)
        epoch_eo = filt_eo[:n_samples]
        epoch_ec = filt_ec[:n_samples]
        t = np.linspace(0, epoch_length, n_samples, endpoint=False)

        fig, ax = plt.subplots(figsize=PLOT_CONFIG["psd_figsize"], facecolor='black')
        ax.set_facecolor('black')
        ax.plot(t, epoch_eo, linestyle='-', color=colors[0], label='EO')
        ax.plot(t, epoch_ec, linestyle='--', color=colors[1], label='EC')
        ax.set_title(f"{ch_name} {band_name} Waveform", color='white', fontsize=10)
        ax.set_xlabel("Time (s)", color='white')
        ax.set_ylabel("Amplitude (µV)", color='white')
        ax.tick_params(colors='white')
        ax.legend(fontsize=8)
        fig.tight_layout()
        logger.info(f"Generated waveform overlay for {ch_name} {band_name}.")
        return fig
    except Exception as e:
        logger.error(f"Failed to plot waveform overlay for {ch_name} {band_name}: {e}")
        # Fallback: Plot raw signal if filtering fails
        try:
            n_samples = min(len(eo_sig), len(ec_sig))
            t = np.linspace(0, n_samples / sfreq, n_samples, endpoint=False)
            fig, ax = plt.subplots(figsize=PLOT_CONFIG["psd_figsize"], facecolor='black')
            ax.set_facecolor('black')
            ax.plot(t, eo_sig[:n_samples], linestyle='-', color=colors[0], label='EO (Raw)')
            ax.plot(t, ec_sig[:n_samples], linestyle='--', color=colors[1], label='EC (Raw)')
            ax.set_title(f"{ch_name} {band_name} Raw Signal (Fallback)", color='white', fontsize=10)
            ax.set_xlabel("Time (s)", color='white')
            ax.set_ylabel("Amplitude (µV)", color='white')
            ax.tick_params(colors='white')
            ax.legend(fontsize=8)
            fig.tight_layout()
            logger.info(f"Generated fallback raw signal plot for {ch_name} {band_name}.")
            return fig
        except Exception as e2:
            logger.error(f"Failed to plot fallback raw signal for {ch_name} {band_name}: {e2}")
            return None

def generate_full_site_plots(raw_eo, raw_ec, output_dir):
    """
    Generate PSD, waveform, and difference bar plots for all available channels and frequency bands.
    
    Args:
        raw_eo (mne.io.Raw): Eyes-open raw EEG data.
        raw_ec (mne.io.Raw): Eyes-closed raw EEG data.
        output_dir (str or Path): Directory to save the plots (detailed_site_plots).
    
    Returns:
        dict: Mapping of sites to plot paths for use in report generation.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting site plot generation in {output_dir}")

    # Validate inputs
    if raw_eo is None or raw_ec is None:
        logger.error("Skipping site plots: Missing EO or EC data.")
        with open(output_dir / "missing_channels_log.txt", "a", encoding="utf-8") as f:
            f.write("Skipping site plots: Missing EO or EC data.\n")
        return {}
    
    if raw_eo.get_data().size == 0 or raw_ec.get_data().size == 0:
        logger.error("Skipping site plots: Empty EO or EC data.")
        with open(output_dir / "missing_channels_log.txt", "a", encoding="utf-8") as f:
            f.write("Skipping site plots: Empty EO or EC data.\n")
        return {}
    
    logger.info(f"EO channels: {raw_eo.ch_names}")
    logger.info(f"EC channels: {raw_ec.ch_names}")
    logger.info(f"EO data shape: {raw_eo.get_data().shape}")
    logger.info(f"EC data shape: {raw_ec.get_data().shape}")

    # Use all common channels, handling case sensitivity
    common_channels = list(set(ch.lower() for ch in raw_eo.ch_names).intersection(ch.lower() for ch in raw_ec.ch_names))
    common_channels = [ch for ch in raw_eo.ch_names if ch.lower() in common_channels]  # Preserve original case
    if not common_channels:
        logger.error("No common channels between EO and EC data.")
        with open(output_dir / "missing_channels_log.txt", "a", encoding="utf-8") as f:
            f.write(f"No common channels between EO ({raw_eo.ch_names}) and EC ({raw_ec.ch_names})\n")
        return {}
    logger.info(f"Common channels for site plots: {common_channels}")

    # Initialize site_dict
    site_dict = {ch: {} for ch in common_channels}
    
    # Verify sampling frequency
    sfreq = raw_eo.info['sfreq']
    if sfreq != raw_ec.info['sfreq']:
        logger.warning(f"Sampling frequencies differ: EO={sfreq}, EC={raw_ec.info['sfreq']}. Using EO sampling frequency.")
    
    # Compute band powers for difference plots
    bp_eo = compute_all_band_powers(raw_eo) if raw_eo else {}
    bp_ec = compute_all_band_powers(raw_ec) if raw_ec else {}
    logger.debug(f"Band powers computed for EO: {list(bp_eo.keys())}")
    logger.debug(f"Band powers computed for EC: {list(bp_ec.keys())}")

    # Process each channel
    for site in common_channels:
        site_folder = output_dir / site
        psd_folder = site_folder / "PSD_Overlay"
        wave_folder = site_folder / "Waveform_Overlay"
        diff_folder = site_folder / "Difference"
        try:
            psd_folder.mkdir(parents=True, exist_ok=True)
            wave_folder.mkdir(parents=True, exist_ok=True)
            diff_folder.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directories for {site}: {psd_folder}, {wave_folder}, {diff_folder}")
        except Exception as e:
            logger.error(f"Failed to create directories for {site}: {e}")
            with open(output_dir / "missing_channels_log.txt", "a", encoding="utf-8") as f:
                f.write(f"Failed to create directories for {site}: {e}\n")
            continue

        logger.debug(f"Processing site: {site}")
        try:
            eo_data = raw_eo.get_data(picks=[site])[0] * 1e6  # Convert to microvolts
            ec_data = raw_ec.get_data(picks=[site])[0] * 1e6
            # Replace non-finite values with zeros
            eo_data = np.nan_to_num(eo_data, nan=0.0, posinf=0.0, neginf=0.0)
            ec_data = np.nan_to_num(ec_data, nan=0.0, posinf=0.0, neginf=0.0)
            logger.info(f"Site {site} - EO samples: {len(eo_data)}, EC samples: {len(ec_data)}")
            if len(eo_data) < 32 or len(ec_data) < 32:
                logger.warning(f"Skipping site {site}: Insufficient samples (EO: {len(eo_data)}, EC: {len(ec_data)}).")
                with open(output_dir / "missing_channels_log.txt", "a", encoding="utf-8") as f:
                    f.write(f"Site {site}: Insufficient samples (EO: {len(eo_data)}, EC: {len(ec_data)})\n")
                continue

            for band_name in BANDS.keys():
                band = BANDS[band_name]
                logger.debug(f"Generating plots for {site} {band_name}")
                site_dict[site][band_name] = {}
                try:
                    # Generate PSD overlay
                    logger.debug(f"Attempting PSD overlay for {site} {band_name}")
                    fig_psd = plot_band_psd_overlay(eo_data, ec_data, sfreq, band, site, band_name)
                    if fig_psd:
                        psd_path = psd_folder / f"{site}_PSD_{band_name}.png"
                        try:
                            fig_psd.savefig(psd_path, dpi=PLOT_CONFIG["dpi"], facecolor='black')
                            logger.info(f"Saved PSD overlay for {site} {band_name} to {psd_path}")
                            site_dict[site][band_name]["psd"] = str(os.path.relpath(psd_path, start=output_dir))
                        except Exception as e:
                            logger.error(f"Failed to save PSD plot for {site} {band_name}: {e}")
                        finally:
                            plt.close(fig_psd)
                            gc.collect()  # Free memory
                    else:
                        logger.warning(f"No PSD plot generated for {site} {band_name}.")
                        site_dict[site][band_name]["psd"] = ""

                    # Generate waveform overlay
                    logger.debug(f"Attempting waveform overlay for {site} {band_name}")
                    fig_wave = None
                    try:
                        fig_wave = plot_band_waveform_overlay(eo_data, ec_data, sfreq, band, site, band_name)
                        if fig_wave:
                            wave_path = wave_folder / f"{site}_Waveform_{band_name}.png"
                            fig_wave.savefig(wave_path, dpi=PLOT_CONFIG["dpi"], facecolor='black')
                            logger.info(f"Saved waveform overlay for {site} {band_name} to {wave_path}")
                            site_dict[site][band_name]["wave"] = str(os.path.relpath(wave_path, start=output_dir))
                        else:
                            logger.warning(f"No waveform plot generated for {site} {band_name}.")
                            site_dict[site][band_name]["wave"] = ""
                    except Exception as e:
                        logger.error(f"Failed to generate waveform for {site} {band_name}: {e}")
                        site_dict[site][band_name]["wave"] = ""
                    finally:
                        if fig_wave:
                            plt.close(fig_wave)
                            gc.collect()  # Free memory

                    # Generate difference bar plot
                    logger.debug(f"Attempting difference bar plot for {site} {band_name}")
                    diff_vals = [bp_eo.get(site, {}).get(band_name, 0) - bp_ec.get(site, {}).get(band_name, 0)]
                    fig_diff = plot_difference_bar(diff_vals, [site], band_name)
                    if fig_diff:
                        diff_path = diff_folder / f"{site}_Difference_{band_name}.png"
                        try:
                            fig_diff.savefig(diff_path, dpi=PLOT_CONFIG["dpi"], facecolor='black')
                            logger.info(f"Saved difference bar plot for {site} {band_name} to {diff_path}")
                            site_dict[site][band_name]["diff_bar"] = str(os.path.relpath(diff_path, start=output_dir))
                        except Exception as e:
                            logger.error(f"Failed to save difference bar plot for {site} {band_name}: {e}")
                        finally:
                            plt.close(fig_diff)
                            gc.collect()  # Free memory
                    else:
                        logger.warning(f"No difference bar plot generated for {site} {band_name}.")
                        site_dict[site][band_name]["diff_bar"] = ""
                except Exception as e:
                    logger.warning(f"Failed to generate plots for {site} {band_name}: {e}")
                    with open(output_dir / "missing_channels_log.txt", "a", encoding="utf-8") as f:
                        f.write(f"Site {site} {band_name}: Failed to generate plots - {e}\n")
                    site_dict[site][band_name]["psd"] = ""
                    site_dict[site][band_name]["wave"] = ""
                    site_dict[site][band_name]["diff_bar"] = ""
                    continue
        except Exception as e:
            logger.error(f"Error processing site {site}: {e}")
            with open(output_dir / "missing_channels_log.txt", "a", encoding="utf-8") as f:
                f.write(f"Site {site}: Error - {e}\n")
            continue

    logger.info(f"Completed site plot generation. Generated plots for {len(site_dict)} sites.")
    logger.debug(f"site_dict: {site_dict}")
    return site_dict

def plot_zscore_topomap(z_scores, info, band_name, cond_name):
    """
    Generate a topomap of robust z-scores for a specified frequency band and condition.
    
    Args:
        z_scores (array-like): Z-scores per channel.
        info (mne.Info): MNE info with channel locations.
        band_name (str): Frequency band name.
        cond_name (str): Condition label.
    
    Returns:
        matplotlib.figure.Figure: The z-score topomap figure, or None if invalid.
    """
    return plot_topomap(z_scores, info, f"{band_name} Robust Z-Score ({cond_name})",
                       cmap=PLOT_CONFIG["zscore_cmap"], clim=PLOT_CONFIG["zscore_clim"])

def plot_tfr(tfr, picks=0):
    """
    Plot Time-Frequency Representation (TFR) for a given channel using imshow.
    
    Args:
        tfr (mne.time_frequency.AverageTFR): Computed TFR object.
        picks (int): Channel index to plot.
    
    Returns:
        matplotlib.figure.Figure: Figure displaying the TFR, or None if invalid.
    """
    if tfr is None or not tfr.data.size:
        logger.warning("Cannot plot TFR: Invalid TFR data.")
        return None
    try:
        data = tfr.data[picks]  # shape: (n_freqs, n_times)
        times = tfr.times
        freqs = tfr.freqs
        fig, ax = plt.subplots(figsize=PLOT_CONFIG["tfr_figsize"], facecolor='black')
        ax.set_facecolor('black')
        im = ax.imshow(data, aspect='auto', origin='lower',
                       extent=[times[0], times[-1], freqs[0], freqs[-1]],
                       cmap=PLOT_CONFIG["topomap_cmap"])
        ax.set_xlabel("Time (s)", color='white')
        ax.set_ylabel("Frequency (Hz)", color='white')
        ax.set_title("Time-Frequency Representation", color='white', fontsize=12)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Power", color='white')
        cbar.ax.tick_params(colors='white')
        fig.tight_layout()
        logger.info("Generated TFR plot.")
        return fig
    except Exception as e:
        logger.warning(f"Failed to plot TFR: {e}")
        return None

def plot_ica_components(ica, raw):
    """
    Plot ICA components with a unified title.
    
    Args:
        ica (mne.preprocessing.ICA): Fitted ICA object.
        raw (mne.io.Raw): Raw EEG data.
    
    Returns:
        matplotlib.figure.Figure or list: Figure(s) displaying the ICA components, or None if invalid.
    """
    if ica is None or raw is None:
        logger.warning("Cannot plot ICA components: Invalid ICA or raw data.")
        return None
    try:
        fig = ica.plot_components(inst=raw, show=False)
        if isinstance(fig, list):
            for f in fig:
                f.suptitle("ICA Components", color='white')
        else:
            fig.suptitle("ICA Components", color='white')
        logger.info("Generated ICA components plot.")
        return fig
    except Exception as e:
        logger.warning(f"Failed to plot ICA components: {e}")
        return None

def plot_source_estimate(stc, view="lateral", time_point=0.1, subjects_dir=None):
    """
    Plot the source estimate for a given view and time point.
    
    Args:
        stc: Source estimate object (e.g., from MNE's inverse methods).
        view (str): Desired brain view (e.g., "lateral", "medial").
        time_point (float): Time (in seconds) at which to capture the view.
        subjects_dir (str): Directory where subject MRI data is stored.
    
    Returns:
        matplotlib.figure.Figure: Figure containing the source estimate visualization, or None if invalid.
    """
    if stc is None or subjects_dir is None:
        logger.warning("Cannot plot source estimate: Invalid stc or subjects_dir.")
        return None
    try:
        brain = stc.plot(hemi="both", subjects_dir=subjects_dir, time_viewer=False, smoothing_steps=5, colormap="hot")
        brain.set_time(time_point)
        brain.show_view(view)
        img = brain.screenshot()
        brain.close()

        fig, ax = plt.subplots(figsize=PLOT_CONFIG["source_figsize"], facecolor='black')
        ax.set_facecolor('black')
        im = ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"Source Localization ({view}, t={time_point:.2f}s)", color='white', fontsize=12)
        vmin = stc.data.min()
        vmax = stc.data.max()
        from matplotlib.cm import get_cmap
        cmap = get_cmap("hot")
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label("Source Activity", color='white')
        cbar.ax.tick_params(colors='white')
        fig.tight_layout()
        logger.info(f"Generated source estimate plot for view {view} at t={time_point:.2f}s.")
        return fig
    except Exception as e:
        logger.warning(f"Failed to plot source estimate: {e}")
        return None

def plot_difference_topomap(diff_vals, info, band_name):
    """
    Generate a topomap of difference values for a specified frequency band.
    
    Args:
        diff_vals (array-like): Difference values per channel.
        info (mne.Info): MNE info with channel locations.
        band_name (str): Frequency band name.
    
    Returns:
        matplotlib.figure.Figure: The difference topomap figure, or None if invalid.
    """
    return plot_topomap(diff_vals, info, f"{band_name} Difference Topomap",
                       cmap=PLOT_CONFIG["difference_cmap"])

def plot_difference_bar(diff_vals, ch_names, band_name):
    """
    Generate a bar graph of difference values for a specified frequency band.
    
    Args:
        diff_vals (array-like): Difference values for each channel.
        ch_names (list): List of channel names.
        band_name (str): Frequency band name.
    
    Returns:
        matplotlib.figure.Figure: The difference bar graph figure, or None if invalid.
    """
    if not diff_vals or not ch_names:
        logger.warning(f"Cannot plot difference bar for {band_name}: Invalid input data.")
        return None
    diff_vals = np.array(diff_vals)
    if len(diff_vals) != len(ch_names):
        logger.warning(f"Data length mismatch for {band_name}: {len(diff_vals)} vs {len(ch_names)}.")
        return None
    fig, ax = plt.subplots(figsize=PLOT_CONFIG["bar_figsize"], facecolor='black')
    ax.set_facecolor('black')
    indices = np.arange(len(ch_names))
    bars = ax.bar(indices, diff_vals, color=['green' if v >= 0 else 'red' for v in diff_vals])
    ax.set_xticks(indices)
    ax.set_xticklabels(ch_names, rotation=90, fontsize=8, color='white')
    ax.set_ylabel("Difference", color='white')
    ax.set_title(f"{band_name} Difference Bar Graph", color='white', fontsize=12)
    ax.tick_params(colors='white')
    fig.tight_layout()
    logger.info(f"Generated difference bar graph for {band_name}.")
    return fig

def process_topomaps(raw, condition, folders, band_list):
    """
    Process and save topomaps for all specified frequency bands.
    
    Args:
        raw (mne.io.Raw): Raw EEG data.
        condition (str): Condition label (e.g., "EO" or "EC").
        folders (dict): Dictionary of output folders.
        band_list (list): List of frequency band names.
    
    Returns:
        dict: Mapping of band names to saved topomap file paths, or empty dict if failed.
    """
    if raw is None or not raw.ch_names:
        logger.warning(f"Skipping topomap processing for {condition}: No data available.")
        return {}
    try:
        bp = compute_all_band_powers(raw)
        if not bp:
            logger.warning(f"No band powers computed for {condition}.")
            return {}
        topomaps = {}
        logger.debug(f"Band list for {condition}: {band_list}")
        logger.debug(f"Sample band powers: {list(bp.items())[:1]}")  # Log sample of bp
        for band in band_list:
            if not isinstance(band, str):
                logger.error(f"Invalid band key type for {band} ({condition}): {type(band)}")
                band_str = str(band)  # Convert to string as fallback
            else:
                band_str = band
            try:
                abs_power = [bp.get(ch, {}).get(band, 0) for ch in raw.ch_names]
                rel_power = []
                instability_vals = []
                for ch in raw.ch_names:
                    total_power = sum(bp.get(ch, {}).get(b, 0) for b in band_list)
                    rel_power.append(bp.get(ch, {}).get(band, 0) / total_power if total_power else 0)
                    instability = bp.get(ch, {}).get('instability', {}).get(band, 0)
                    instability_vals.append(instability if np.isfinite(instability) else 0)
                fig = plot_topomap_abs_rel(abs_power, rel_power, raw, band_str, condition, instability_vals if any(v != 0 for v in instability_vals) else None)
                if fig:
                    topo_path = Path(folders[f"topomaps_{condition.lower()}"]) / f"topomaps_Abs_Rel_INSTABILITY_{band_str}_{condition}.png"
                    topo_path.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(topo_path, facecolor='black')
                    plt.close(fig)
                    topomaps[band_str] = topo_path.name
                    logger.info(f"Saved {condition} topomap for {band_str} to {topo_path}")
            except Exception as e:
                logger.warning(f"Failed to process topomap for {band_str} ({condition}): {e}")
        logger.debug(f"Returning topomaps dictionary for {condition}: {topomaps}")
        return topomaps
    except Exception as e:
        logger.error(f"Failed to process topomaps for {condition}: {e}")
        return {}

def process_waveforms(raw, condition, folders, band_list):
    """
    Process and save waveform grids for all specified frequency bands.
    
    Args:
        raw (mne.io.Raw): Raw EEG data.
        condition (str): Condition label (e.g., "EO" or "EC").
        folders (dict): Dictionary of output folders.
        band_list (list): List of frequency band names.
    
    Returns:
        dict: Mapping of band names to saved waveform file paths, or empty dict if failed.
    """
    if raw is None:
        logger.warning(f"Skipping waveform processing for {condition}: No data available.")
        return {}
    try:
        global_waveforms = {}
        data = raw.get_data() * 1e6
        sfreq = raw.info['sfreq']
        for band in band_list:
            try:
                wf_fig = plot_waveform_grid(data, raw.ch_names, sfreq, band=BANDS[band], epoch_length=PLOT_CONFIG["epoch_length"])
                if wf_fig:
                    wf_path = Path(folders[f"waveforms_{condition.lower()}"]) / f"waveforms_{band}.png"
                    wf_path.parent.mkdir(parents=True, exist_ok=True)
                    wf_fig.savefig(wf_path, facecolor='black')
                    plt.close(wf_fig)
                    global_waveforms[band] = wf_path.name
                    logger.info(f"Saved {condition} waveform grid for {band} to {wf_path}")
            except Exception as e:
                logger.warning(f"Failed to process waveform for {band} ({condition}): {e}")
        return global_waveforms
    except Exception as e:
        logger.error(f"Failed to process waveforms for {condition}: {e}")
        return {}

def process_erp(raw, condition, folders):
    """
    Process and save ERP visualization.
    
    Args:
        raw (mne.io.Raw): Raw EEG data.
        condition (str): Condition label (e.g., "EO" or "EC").
        folders (dict): Dictionary of output folders.
    
    Returns:
        str: Saved ERP file path, or empty string if failed.
    """
    if raw is None:
        logger.warning(f"Skipping ERP processing for {condition}: No data available.")
        return ""
    try:
        erp_fig = compute_pseudo_erp(raw)
        if erp_fig:
            erp_path = Path(folders["erp"]) / f"erp_{condition}.png"
            erp_path.parent.mkdir(parents=True, exist_ok=True)
            erp_fig.savefig(erp_path, facecolor='black')
            plt.close(erp_fig)
            logger.info(f"Saved ERP {condition} to {erp_path}")
            return erp_path.name
        return ""
    except Exception as e:
        logger.warning(f"Failed to process ERP for {condition}: {e}")
        return ""

def process_coherence(raw, condition, folders, band_list):
    """
    Process and save coherence matrices for all specified frequency bands.
    
    Args:
        raw (mne.io.Raw): Raw EEG data.
        condition (str): Condition label (e.g., "EO" or "EC").
        folders (dict): Dictionary of output folders.
        band_list (list): List of frequency band names.
    
    Returns:
        dict: Mapping of band names to saved coherence file paths, or empty dict if failed.
    """
    if raw is None or not raw.ch_names:
        logger.warning(f"Skipping coherence processing for {condition}: No data available.")
        return {}
    try:
        coherence_maps = {}
        sfreq = raw.info['sfreq']
        data = raw.get_data() * 1e6  # Convert to microvolts
        # Validate data
        if not np.isfinite(data).all():
            logger.warning(f"Non-finite values detected in {condition} data for coherence processing.")
            return {}
        if data.shape[1] < int(sfreq * 2):
            logger.warning(f"Insufficient data length for {condition} coherence: {data.shape[1]} samples.")
            return {}
        for band in band_list:
            try:
                logger.debug(f"Computing coherence matrix for {band} ({condition})")
                coh_matrix = compute_coherence_matrix(data, sfreq, BANDS[band], nperseg=int(sfreq * 2))
                if coh_matrix is None or not np.isfinite(coh_matrix).all():
                    logger.warning(f"Invalid coherence matrix for {band} ({condition}): Skipping.")
                    continue
                fig_coh = plot_coherence_matrix(coh_matrix, raw.ch_names)
                if fig_coh:
                    coh_path = Path(folders[f"coherence_{condition.lower()}"]) / f"coherence_{band}.png"
                    coh_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        fig_coh.savefig(coh_path, facecolor='black')
                        logger.info(f"Saved {condition} coherence ({band}) to {coh_path}")
                        coherence_maps[band] = coh_path.name
                    except Exception as e:
                        logger.warning(f"Failed to save coherence plot for {band} ({condition}): {e}")
                    finally:
                        plt.close(fig_coh)
                        gc.collect()  # Free memory
                else:
                    logger.warning(f"No coherence plot generated for {band} ({condition}).")
            except Exception as e:
                logger.warning(f"Failed to process coherence for {band} ({condition}): {e}")
        return coherence_maps
    except Exception as e:
        logger.error(f"Failed to process coherence for {condition}: {e}")
        return {}

def process_zscores(raw, condition, folders, band_list, norm_stats):
    """
    Process and save z-score topomaps for all specified frequency bands.
    
    Args:
        raw (mne.io.Raw): Raw EEG data.
        condition (str): Condition label (e.g., "EO" or "EC").
        folders (dict): Dictionary of output folders.
        band_list (list): List of frequency band names.
        norm_stats (dict): Normative statistics for z-score computation.
    
    Returns:
        dict: Mapping of band names to saved z-score file paths, or empty dict if failed.
    """
    if raw is None:
        logger.warning(f"Skipping z-score processing for {condition}: No data available.")
        return {}
    try:
        zscore_maps = compute_all_zscore_maps(raw, norm_stats, epoch_len_sec=2.0)
        zscore_images = {}
        for band in band_list:
            try:
                if band in zscore_maps and zscore_maps[band] is not None:
                    fig_zscore = plot_zscore_topomap(zscore_maps[band], raw.info, band, condition)
                    if fig_zscore:
                        zscore_path = Path(folders[f"zscore_{condition.lower()}"]) / f"zscore_{band}.png"
                        zscore_path.parent.mkdir(parents=True, exist_ok=True)
                        fig_zscore.savefig(zscore_path, facecolor='black')
                        plt.close(fig_zscore)
                        zscore_images[band] = zscore_path.name
                        logger.info(f"Saved {condition} z-score ({band}) to {zscore_path}")
            except Exception as e:
                logger.warning(f"Failed to process z-score for {band} ({condition}): {e}")
        return zscore_images
    except Exception as e:
        logger.error(f"Failed to process z-scores for {condition}: {e}")
        return {}

def process_variance_topomaps(raw, condition, folders, band_list):
    """
    Process and save variance topomaps for all specified frequency bands.
    
    Args:
        raw (mne.io.Raw): Raw EEG data.
        condition (str): Condition label (e.g., "EO" or "EC").
        folders (dict): Dictionary of output folders.
        band_list (list): List of frequency band names.
    
    Returns:
        dict: Mapping of band names to saved variance file paths, or empty dict if failed.
    """
    if raw is None or not raw.ch_names:
        logger.warning(f"Skipping variance topomap processing for {condition}: No data available.")
        return {}
    try:
        from mne.time_frequency import psd_array_welch
        data = raw.get_data() * 1e6  # Convert to microvolts
        sfreq = raw.info['sfreq']
        min_samples = 512
        if data.shape[1] < min_samples:
            logger.warning(f"Insufficient data length for {condition} variance topomaps: {data.shape[1]} samples.")
            return {}
        if not np.isfinite(data).all():
            logger.warning(f"Non-finite values detected in {condition} data for variance topomaps.")
            return {}
        n_fft = min(data.shape[1], 2048)  # Dynamic FFT length
        psds, freqs = psd_array_welch(data, sfreq, fmin=1, fmax=40, n_fft=n_fft, verbose=False)
        psds_db = 10 * np.log10(psds)
        variance_maps = {}
        for band in band_list:
            try:
                band_range = BANDS[band]
                band_mask = (freqs >= band_range[0]) & (freqs <= band_range[1])
                if not np.any(band_mask):
                    logger.warning(f"No frequencies found for {band} ({condition}) in range {band_range}.")
                    continue
                band_power = psds_db[:, band_mask].mean(axis=1)
                variance = np.var(band_power)
                fig_variance = plot_topomap([variance] * len(band_power), raw.info,
                                           f"{band} Variance ({condition})", cmap=PLOT_CONFIG["topomap_cmap"])
                if fig_variance:
                    variance_path = Path(folders[f"variance_{condition.lower()}"]) / f"variance_{band}.png"
                    variance_path.parent.mkdir(parents=True, exist_ok=True)
                    fig_variance.savefig(variance_path, facecolor='black')
                    plt.close(fig_variance)
                    variance_maps[band] = variance_path.name
                    logger.info(f"Saved {condition} variance topomap for {band} to {variance_path}")
            except Exception as e:
                logger.warning(f"Failed to process variance topomap for {band} ({condition}): {e}")
        return variance_maps
    except Exception as e:
        logger.error(f"Failed to process variance topomaps for {condition}: {e}")
        return {}

def process_tfr(raw, condition, folders, band_list):
    """
    Process and save TFR visualizations for all specified frequency bands.
    
    Args:
        raw (mne.io.Raw): Raw EEG data.
        condition (str): Condition label (e.g., "EO" or "EC").
        folders (dict): Dictionary of output folders.
        band_list (list): List of frequency band names.
    
    Returns:
        dict: Mapping of band names to saved TFR file paths, or empty dict if failed.
    """
    if raw is None or not raw.ch_names:
        logger.warning(f"Skipping TFR processing for {condition}: No data available.")
        return {}
    try:
        tfr_images = {}
        n_cycles = 2.0
        for band in band_list:
            try:
                freqs = np.linspace(BANDS[band][0], BANDS[band][1], num=10)
                tfr = compute_tfr(raw, freqs=freqs, n_cycles=n_cycles, tmin=0.0, tmax=4.0)
                if tfr and tfr.data.size:
                    fig_tfr = plot_tfr(tfr, picks=0)
                    if fig_tfr:
                        tfr_path = Path(folders[f"tfr_{condition.lower()}"]) / f"tfr_{band}.png"
                        tfr_path.parent.mkdir(parents=True, exist_ok=True)
                        fig_tfr.savefig(tfr_path, facecolor='black')
                        plt.close(fig_tfr)
                        tfr_images[band] = tfr_path.name
                        logger.info(f"Saved TFR {condition} ({band}) to {tfr_path}")
                else:
                    logger.warning(f"TFR {condition} for {band} was not computed.")
            except Exception as e:
                logger.warning(f"Failed to process TFR for {band} ({condition}): {e}")
        return tfr_images
    except Exception as e:
        logger.error(f"Failed to process TFR for {condition}: {e}")
        return {}

def process_ica(raw, condition, folders):
    """
    Process and save ICA visualization.
    
    Args:
        raw (mne.io.Raw): Raw EEG data.
        condition (str): Condition label (e.g., "EO" or "EC").
        folders (dict): Dictionary of output folders.
    
    Returns:
        str: Saved ICA file path, or empty string if failed.
    """
    if raw is None:
        logger.warning(f"Skipping ICA processing for {condition}: No data available.")
        return ""
    try:
        ica = compute_ica(raw)
        fig_ica = plot_ica_components(ica, raw)
        if fig_ica:
            ica_path = Path(folders[f"ica_{condition.lower()}"]) / f"ica_{condition}.png"
            ica_path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(fig_ica, list):
                for i, f in enumerate(fig_ica):
                    f.savefig(ica_path.with_stem(f"{ica_path.stem}_{i}"), facecolor='black')
                    plt.close(f)
            else:
                fig_ica.savefig(ica_path, facecolor='black')
                plt.close(fig_ica)
            logger.info(f"Saved ICA {condition} to {ica_path}")
            return ica_path.name
        return ""
    except Exception as e:
        logger.warning(f"Failed to process ICA for {condition}: {e}")
        return ""
