"""
Visualization routines for EEG analysis.

This module provides functions for:
  - Global topomaps (absolute & relative power)
  - Global waveform grid
  - PSD overlay plots comparing EO vs. EC for a single channel
  - Waveform overlay plots comparing EO vs. EC for a single channel
  - Global coherence matrix heatmap with channel tick labels
  - Robust z-score topomap
  - Time-Frequency Representation (TFR) plotting for a selected channel
  - ICA component plotting
  - Source localization visualization (e.g., LORETA, sLORETA, eLORETA) with a colorbar
  - **New: Difference Topomap and Difference Bar Graph** plotting routines

All plots are styled in dark mode.
"""

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import mne
import numpy as np
import scipy.signal as sig
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional, Union
import scipy.stats as stats
from scipy.stats import zscore, median_abs_deviation
from scipy.stats import entropy
import warnings
import matplotlib.colors as mpl


# ------------------ Utility ------------------

def remove_overlapping_channels(info, tol=0.05):
    """
    Remove channels with overlapping positions to improve topomap clarity.

    Returns:
      tuple: (cleaned info, list of unique indices)
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


# ------------------ Topomap ------------------

def plot_topomap_abs_rel(abs_vals, rel_vals, info, band_name, cond_name):
    """
    Generate side-by-side topomaps for absolute and relative power with enhanced visuals.
    """
    info_clean, sel_idx = remove_overlapping_channels(info)
    abs_vals_subset = np.array(abs_vals)[sel_idx]
    rel_vals_subset = np.array(rel_vals)[sel_idx]

    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='#000000')
    fig.patch.set_facecolor('#000000')

    # Enhanced absolute power plot
    ax_abs = axes[0]
    ax_abs.set_facecolor('#000000')
    vmin_abs, vmax_abs = np.percentile(abs_vals_subset, [5, 95])
    im_abs, _ = mne.viz.plot_topomap(
        abs_vals_subset, info_clean, axes=ax_abs,
        cmap='plasma', show=False,
        vlim=(vmin_abs, vmax_abs),
        outlines='head',
        contours=6
    )
    ax_abs.set_title(f"{band_name} Absolute Power\n({cond_name})", 
                     color='#00ffff', fontsize=12, pad=20)
    cbar_abs = plt.colorbar(im_abs, ax=ax_abs, orientation='horizontal', 
                           fraction=0.05, pad=0.08)
    cbar_abs.ax.tick_params(colors='#00ffff')
    cbar_abs.set_label('μV²', color='#00ffff')

    # Enhanced relative power plot
    ax_rel = axes[1]
    ax_rel.set_facecolor('#000000')
    vmin_rel, vmax_rel = np.percentile(rel_vals_subset, [5, 95])
    im_rel, _ = mne.viz.plot_topomap(
        rel_vals_subset, info_clean, axes=ax_rel,
        cmap='magma', show=False,
        vlim=(vmin_rel, vmax_rel),
        outlines='head',
        contours=6
    )
    ax_rel.set_title(f"{band_name} Relative Power\n({cond_name})", 
                     color='#ff00ff', fontsize=12, pad=20)
    cbar_rel = plt.colorbar(im_rel, ax=ax_rel, orientation='horizontal', 
                           fraction=0.05, pad=0.08)
    cbar_rel.ax.tick_params(colors='#ff00ff')
    cbar_rel.set_label('%', color='#ff00ff')

    fig.suptitle(f"Brain Activity Analysis - {band_name}", 
                 color='#ffffff', fontsize=14, y=1.05)
    
    # Add subtle grid for better spatial reference
    for ax in axes:
        ax.grid(True, color='#333333', linestyle='--', alpha=0.3)
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ------------------ Waveform Grid ------------------

def plot_waveform_grid(data, ch_names, sfreq, band=(8, 12), epoch_length=10):
    """
    Generate an enhanced grid of waveform plots with candy-colored gradients.
    """
    n_channels = data.shape[0]
    n_samples = int(sfreq * epoch_length)
    t = np.linspace(0, epoch_length, n_samples, endpoint=False)
    n_cols = 4
    n_rows = int(np.ceil(n_channels / n_cols))

    plt.style.use('dark_background')
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3), 
                            facecolor='#000000')
    fig.patch.set_facecolor('#000000')
    axes = np.atleast_2d(axes)

    colors = ['#00ffff', '#ff00ff', '#00ff00', '#ff3366', '#ffff00']
    
    for i in range(n_channels):
        ax = axes[i // n_cols, i % n_cols]
        ax.set_facecolor('#000000')
        
        # Filter and plot with enhanced styling
        filt = mne.filter.filter_data(data[i], sfreq, band[0], band[1], verbose=False)
        line = ax.plot(t, filt[:n_samples], color=colors[i % len(colors)], 
                      lw=1.5, alpha=0.8)[0]
        
        # Add gradient fill
        ax.fill_between(t, filt[:n_samples], alpha=0.1, 
                       color=colors[i % len(colors)])
        
        # Add glow effect
        line.set_path_effects([path_effects.withSimplePatchShadow(
            offset=(0, 0), 
            shadow_rgbFace=colors[i % len(colors)],
            alpha=0.3,
            rho=0.3
        )])
        
        ax.set_title(ch_names[i], color='#ffffff', fontsize=10)
        ax.set_xlim([0, epoch_length])
        ax.tick_params(colors='#ffffff', labelsize=8)
        ax.grid(True, color='#333333', linestyle='--', alpha=0.3)

    for j in range(n_channels, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis('off')

    fig.suptitle(f"Neural Oscillations ({band[0]}-{band[1]} Hz)", 
                 color='#ffffff', fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


# ------------------ Coherence Matrix ------------------

def plot_coherence_matrix(coh_matrix, ch_names=None):
    """
    Generate an enhanced heatmap of the coherence matrix with candy-colored styling.
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#000000')
    ax.set_facecolor('#000000')
    
    # Use custom colormap for better visualization
    cmap = plt.cm.get_cmap('plasma')
    im = ax.imshow(coh_matrix, cmap=cmap, aspect='auto')
    
    # Enhanced styling
    ax.set_title("Neural Network Coherence", color='#00ffff', 
                 fontsize=14, pad=20)
    
    if ch_names is not None and len(ch_names) == coh_matrix.shape[0]:
        ax.set_xticks(range(len(ch_names)))
        ax.set_xticklabels(ch_names, rotation=45, ha='right', 
                          fontsize=8, color='#ffffff')
        ax.set_yticks(range(len(ch_names)))
        ax.set_yticklabels(ch_names, fontsize=8, color='#ffffff')
    
    ax.set_xlabel("Channels", color='#ffffff', fontsize=10)
    ax.set_ylabel("Channels", color='#ffffff', fontsize=10)
    
    # Enhanced colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Coherence Strength", color='#00ffff', fontsize=10)
    cbar.ax.tick_params(colors='#ffffff')
    
    # Add grid for better readability
    ax.grid(True, color='#333333', linestyle='--', alpha=0.3)
    
    fig.tight_layout()
    return fig


# ------------------ PSD & Waveform Overlays ------------------

def plot_band_psd_overlay(eo_sig, ec_sig, sfreq, band, ch_name, band_name, 
                         colors=('#00ffff', '#ff00ff')):
    """
    Enhanced PSD overlay plot with candy-colored gradients and effects.
    """
    fmin, fmax = band
    freqs_eo, psd_eo = sig.welch(eo_sig, fs=sfreq, nperseg=int(sfreq * 2))
    freqs_ec, psd_ec = sig.welch(ec_sig, fs=sfreq, nperseg=int(sfreq * 2))

    mask_eo = (freqs_eo >= fmin) & (freqs_eo <= fmax)
    mask_ec = (freqs_ec >= fmin) & (freqs_ec <= fmax)
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='#000000')
    ax.set_facecolor('#000000')
    
    # Create gradient fills
    ax.fill_between(freqs_eo[mask_eo], psd_eo[mask_eo], alpha=0.1, 
                    color=colors[0])
    ax.fill_between(freqs_ec[mask_ec], psd_ec[mask_ec], alpha=0.1, 
                    color=colors[1])
    
    # Plot main lines with glow effect
    for freqs, psd, color, label in [
        (freqs_eo[mask_eo], psd_eo[mask_eo], colors[0], 'Eyes Open'),
        (freqs_ec[mask_ec], psd_ec[mask_ec], colors[1], 'Eyes Closed')
    ]:
        line = ax.plot(freqs, psd, color=color, label=label, 
                      linewidth=2.5, alpha=0.8)[0]
        line.set_path_effects([path_effects.withSimplePatchShadow(
            offset=(0, 0),
            shadow_rgbFace=color,
            alpha=0.3,
            rho=0.3
        )])
    
    ax.set_title(f"{ch_name} {band_name} Spectral Power", 
                 color='#ffffff', fontsize=12, pad=20)
    ax.set_xlabel("Frequency (Hz)", color='#ffffff', fontsize=10)
    ax.set_ylabel("Power (μV²/Hz)", color='#ffffff', fontsize=10)
    ax.tick_params(colors='#ffffff')
    
    # Enhanced legend
    ax.legend(facecolor='#000000', edgecolor='#ffffff', 
             labelcolor='#ffffff', fontsize=9)
    
    # Add subtle grid
    ax.grid(True, color='#333333', linestyle='--', alpha=0.3)
    
    fig.tight_layout()
    return fig


def plot_waveform_overlay(data_eo, data_ec, ch_name, sfreq, epoch_length=10):
    """
    Create an enhanced waveform overlay plot comparing EO vs. EC conditions.
    """
    n_samples = int(sfreq * epoch_length)
    t = np.linspace(0, epoch_length, n_samples, endpoint=False)
    
    # Ensure data is not empty and has sufficient samples
    if data_eo is None or data_ec is None:
        raise ValueError("Both EO and EC data must be provided")
    if len(data_eo) < n_samples or len(data_ec) < n_samples:
        raise ValueError("Insufficient data samples for the specified epoch length")
        
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#000000')
    ax.set_facecolor('#000000')
    
    # Plot EO condition with enhanced styling
    line_eo = ax.plot(t, data_eo[:n_samples], color='#00ffff', 
                      label='Eyes Open', lw=2, alpha=0.8)[0]
    line_eo.set_path_effects([path_effects.withSimplePatchShadow(
        offset=(0, 0),
        shadow_rgbFace='#00ffff',
        alpha=0.3,
        rho=0.3
    )])
    
    # Plot EC condition with enhanced styling
    line_ec = ax.plot(t, data_ec[:n_samples], color='#ff00ff', 
                      label='Eyes Closed', lw=2, alpha=0.8)[0]
    line_ec.set_path_effects([path_effects.withSimplePatchShadow(
        offset=(0, 0),
        shadow_rgbFace='#ff00ff',
        alpha=0.3,
        rho=0.3
    )])
    
    # Add gradient fills
    ax.fill_between(t, data_eo[:n_samples], alpha=0.1, color='#00ffff')
    ax.fill_between(t, data_ec[:n_samples], alpha=0.1, color='#ff00ff')
    
    ax.set_title(f"Waveform Comparison - {ch_name}", color='#ffffff', 
                 fontsize=14, pad=20)
    ax.set_xlabel("Time (s)", color='#ffffff', fontsize=12)
    ax.set_ylabel("Amplitude (μV)", color='#ffffff', fontsize=12)
    ax.tick_params(colors='#ffffff')
    ax.grid(True, color='#333333', linestyle='--', alpha=0.3)
    
    # Enhanced legend
    legend = ax.legend(facecolor='#000000', edgecolor='#ffffff', 
                      fontsize=10, framealpha=0.8)
    for text in legend.get_texts():
        text.set_color('#ffffff')
    
    fig.tight_layout()
    return fig


def plot_band_waveform_overlay(
    sig1: np.ndarray,
    sig2: np.ndarray,
    sfreq: float,
    band: Tuple[float, float],
    ch_name: str,
    band_name: str,
    colors: Tuple[str, str] = ('#00ffff', '#ff00ff'),  # Candy cyan and magenta
    epoch_length: float = 10.0,
    figsize: Tuple[float, float] = (10, 4),
) -> plt.Figure:
    """
    Enhanced waveform overlay plot with candy-colored gradients and effects.
    """
    fmin, fmax = band
    sig1_filt = mne.filter.filter_data(sig1, sfreq, fmin, fmax, verbose=False)
    sig2_filt = mne.filter.filter_data(sig2, sfreq, fmin, fmax, verbose=False)
    
    n_samples = int(epoch_length * sfreq)
    if n_samples > len(sig1_filt):
        n_samples = len(sig1_filt)
    if n_samples > len(sig2_filt):
        n_samples = len(sig2_filt)
    
    t = np.arange(n_samples) / sfreq
    sig1_epoch = sig1_filt[:n_samples]
    sig2_epoch = sig2_filt[:n_samples]
        
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=figsize, facecolor='#000000')
    ax.set_facecolor('#000000')
    
    # Create gradient fills
    ax.fill_between(t, sig1_epoch, alpha=0.1, color=colors[0])
    ax.fill_between(t, sig2_epoch, alpha=0.1, color=colors[1])
    
    # Plot main lines with glow effect
    for sig, color, label in [
        (sig1_epoch, colors[0], 'Eyes Open'),
        (sig2_epoch, colors[1], 'Eyes Closed')
    ]:
        line = ax.plot(t, sig, color=color, label=label, 
                      linewidth=2.5, alpha=0.8)[0]
        line.set_path_effects([path_effects.withSimplePatchShadow(
            offset=(0, 0),
            shadow_rgbFace=color,
            alpha=0.3,
            rho=0.3
        )])
    
    ax.set_title(f"{ch_name} {band_name} Neural Activity", 
                 color='#ffffff', fontsize=12, pad=20)
    ax.set_xlabel("Time (s)", color='#ffffff', fontsize=10)
    ax.set_ylabel("Amplitude (μV)", color='#ffffff', fontsize=10)
    
    # Enhanced legend
    ax.legend(facecolor='#000000', edgecolor='#ffffff', 
             labelcolor='#ffffff', fontsize=9)
    
    ax.tick_params(colors='#ffffff')
    
    # Add subtle grid
    ax.grid(True, color='#333333', linestyle='--', alpha=0.3)
    
    # Add time markers
    time_markers = np.arange(0, epoch_length + 1, 2)
    ax.set_xticks(time_markers)
    
    # Add amplitude scale - Fixed calculation
    percentiles1 = np.percentile(sig1_epoch, [2, 98])
    percentiles2 = np.percentile(sig2_epoch, [2, 98])
    y_max = max(abs(percentiles1[0]), abs(percentiles1[1]),
                abs(percentiles2[0]), abs(percentiles2[1]))
    ax.set_ylim(-y_max * 1.2, y_max * 1.2)
    
    fig.tight_layout()
    return fig


# ------------------ Z-Score Topomap ------------------

def plot_zscore_topomap(z_scores, info, band_name, cond_name):
    """
    Enhanced z-score topomap with candy-colored styling.
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='#000000')
    ax.set_facecolor('#000000')
    
    # Use custom colormap for better visualization
    im, _ = mne.viz.plot_topomap(
        z_scores, info, axes=ax,
        cmap='plasma',
        show=False,
        outlines='head',
        contours=6
    )
    
    # Set symmetric color limits
    max_abs = max(abs(np.percentile(z_scores, [2, 98])))
    im.set_clim(-max_abs, max_abs)
    
    ax.set_title(f"{band_name} Activity Pattern\n({cond_name})", 
                 color='#00ffff', fontsize=12, pad=20)
    
    # Enhanced colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal')
    cbar.set_label("Z-Score", color='#00ffff', fontsize=10)
    cbar.ax.tick_params(colors='#ffffff')
    
    # Add subtle grid
    ax.grid(True, color='#333333', linestyle='--', alpha=0.3)
    
    fig.tight_layout()
    return fig


# ------------------ Time-Frequency Representation ------------------

def plot_tfr(tfr, picks=0):
    """
    Plot Time-Frequency Representation (TFR) for a given channel using imshow,
    adding labels and a colorbar.

    Parameters:
      tfr (mne.time_frequency.AverageTFR): Computed TFR object.
      picks (int): Channel index to plot.

    Returns:
      matplotlib.figure.Figure: Figure displaying the TFR.
    """
    data = tfr.data[picks]  # shape: (n_freqs, n_times)
    times = tfr.times
    freqs = tfr.freqs
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='black')
    im = ax.imshow(data, aspect='auto', origin='lower',
                   extent=[times[0], times[-1], freqs[0], freqs[-1]],
                   cmap='viridis')
    ax.set_xlabel("Time (s)", color='white')
    ax.set_ylabel("Frequency (Hz)", color='white')
    ax.set_title("Time-Frequency Representation", color='white')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Power", color='white')
    cbar.ax.tick_params(colors='white')
    return fig


# ------------------ ICA Components ------------------

def plot_ica_components(ica, raw):
    """
    Plot ICA components with a unified title.

    Parameters:
      ica (mne.preprocessing.ICA): Fitted ICA object.
      raw (mne.io.Raw): Raw EEG data.

    Returns:
      matplotlib.figure.Figure or list: Figure(s) displaying the ICA components.
    """
    fig = ica.plot_components(inst=raw, show=False)
    if isinstance(fig, list):
        for f in fig:
            f.suptitle("ICA Components", color='white')
    else:
        fig.suptitle("ICA Components", color='white')
    return fig


# ------------------ Source Localization ------------------

def plot_source_estimate(stc, view="lateral", time_point=0.1, subjects_dir=None):
    """
    Plot the source estimate for a given view and time point, adding a title and a colorbar.

    Parameters:
      stc: Source estimate object (e.g., from MNE's inverse methods).
      view (str): Desired brain view (e.g., "lateral", "medial").
      time_point (float): Time (in seconds) at which to capture the view.
      subjects_dir (str): Directory where subject MRI data is stored.

    Returns:
      matplotlib.figure.Figure: Figure containing the source estimate visualization.
    """
    if subjects_dir is None:
        raise ValueError("A valid 'subjects_dir' must be provided to plot_source_estimate()")
    brain = stc.plot(hemi="both", subjects_dir=subjects_dir, time_viewer=False, smoothing_steps=5, colormap="hot")
    brain.set_time(time_point)
    brain.show_view(view)
    img = brain.screenshot()  # Returns a numpy array.
    brain.close()

    fig, ax = plt.subplots(figsize=(8, 6), facecolor='black')
    im = ax.imshow(img)
    ax.axis('off')
    ax.set_title(f"Source Localization ({view}, t={time_point:.2f}s)", color='white')
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
    return fig


# ------------------ NEW: Difference Topomap & Bar Graph ------------------

def plot_difference_topomap(diff_vals, info, band_name):
    """
    Generate a topomap of difference values for a specified frequency band.

    Parameters:
      diff_vals (array-like): Difference values per channel.
      info (mne.Info): MNE info with channel locations.
      band_name (str): Frequency band name.

    Returns:
      matplotlib.figure.Figure: The difference topomap figure.
    """
    info_clean, sel_idx = remove_overlapping_channels(info)
    diff_vals_subset = np.array(diff_vals)[sel_idx]

    fig, ax = plt.subplots(figsize=(6, 4), facecolor='black')
    ax.set_facecolor('black')
    im, _ = mne.viz.plot_topomap(diff_vals_subset, info_clean, axes=ax, cmap='RdBu_r', show=False)
    # Set symmetric color limits centered at zero
    max_abs = np.max(np.abs(diff_vals_subset))
    im.set_clim(-max_abs, max_abs)
    ax.set_title(f"{band_name} Difference Topomap", color='white', fontsize=10)
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal')
    cbar.ax.tick_params(colors='white')
    fig.tight_layout()
    return fig


def plot_difference_bar(diff_vals, ch_names, band_name):
    """
    Generate a bar graph of difference values for a specified frequency band.

    Parameters:
      diff_vals (array-like): Difference values for each channel.
      ch_names (list): List of channel names.
      band_name (str): Frequency band name.

    Returns:
      matplotlib.figure.Figure: The difference bar graph figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='black')
    ax.set_facecolor('black')
    indices = np.arange(len(ch_names))
    # Color bars based on sign (green for positive, red for negative)
    bars = ax.bar(indices, diff_vals, color=['green' if v >= 0 else 'red' for v in diff_vals])
    ax.set_xticks(indices)
    ax.set_xticklabels(ch_names, rotation=90, fontsize=8, color='white')
    ax.set_ylabel("Difference", color='white')
    ax.set_title(f"{band_name} Difference Bar Graph", color='white', fontsize=12)
    ax.tick_params(colors='white')
    fig.tight_layout()
    return fig


class BrainVisualizer:
    def __init__(self):
        self.channel_positions = {
            'FP1': (-0.2, 0.7), 'FP2': (0.2, 0.7),
            'F7': (-0.5, 0.5), 'F3': (-0.3, 0.4), 'FZ': (0, 0.4), 'F4': (0.3, 0.4), 'F8': (0.5, 0.5),
            'T3': (-0.7, 0), 'C3': (-0.4, 0), 'CZ': (0, 0), 'C4': (0.4, 0), 'T4': (0.7, 0),
            'T5': (-0.5, -0.5), 'P3': (-0.3, -0.4), 'PZ': (0, -0.4), 'P4': (0.3, -0.4), 'T6': (0.5, -0.5),
            'O1': (-0.2, -0.7), 'OZ': (0, -0.7), 'O2': (0.2, -0.7)
        }
        
    def plot_3d_brain_activity(self, activity_data: Dict[str, float], 
                             title: str = "3D Brain Activity Map",
                             save_path: Optional[str] = None) -> None:
        """
        Create an interactive 3D visualization of brain activity.
        
        Args:
            activity_data: Dictionary mapping channel names to activity values
            title: Plot title
            save_path: Optional path to save the plot
        """
        # Convert 2D positions to 3D using spherical projection
        positions_3d = {}
        for ch, pos in self.channel_positions.items():
            x, y = pos
            r = np.sqrt(x**2 + y**2)
            z = np.sqrt(1 - r**2) if r < 1 else 0
            positions_3d[ch] = (x, y, z)
        
        # Create 3D plot
        fig = go.Figure()
        
        # Add electrode points
        x, y, z = [], [], []
        values = []
        labels = []
        for ch in activity_data:
            if ch in positions_3d:
                pos = positions_3d[ch]
                x.append(pos[0])
                y.append(pos[1])
                z.append(pos[2])
                values.append(activity_data[ch])
                labels.append(ch)
        
        # Normalize values for color scaling
        norm_values = stats.zscore(values)
        
        # Add points for electrodes
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers+text',
            marker=dict(
                size=10,
                color=norm_values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Z-Score")
            ),
            text=labels,
            hoverinfo='text+x+y+z'
        ))
        
        # Add surface interpolation
        theta = np.linspace(0, 2*np.pi, 100)
        phi = np.linspace(0, np.pi, 100)
        theta, phi = np.meshgrid(theta, phi)
        
        r = 1
        x_surf = r * np.sin(phi) * np.cos(theta)
        y_surf = r * np.sin(phi) * np.sin(theta)
        z_surf = r * np.cos(phi)
        
        fig.add_trace(go.Surface(
            x=x_surf, y=y_surf, z=z_surf,
            opacity=0.3,
            showscale=False
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            )
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()
    
    def plot_brain_connectivity(self, connectivity_matrix: np.ndarray,
                              channel_names: List[str],
                              threshold: float = 0.5,
                              title: str = "Brain Connectivity Network",
                              save_path: Optional[str] = None) -> None:
        """
        Create a network visualization of brain connectivity.
        
        Args:
            connectivity_matrix: Square matrix of connectivity values between channels
            channel_names: List of channel names
            threshold: Minimum connectivity value to show connection
            title: Plot title
            save_path: Optional path to save the plot
        """
        # Create graph
        G = nx.Graph()
        
        # Add nodes
        for ch in channel_names:
            if ch in self.channel_positions:
                G.add_node(ch, pos=self.channel_positions[ch])
        
        # Add edges above threshold
        n = len(channel_names)
        for i in range(n):
            for j in range(i+1, n):
                if connectivity_matrix[i,j] >= threshold:
                    ch1, ch2 = channel_names[i], channel_names[j]
                    if ch1 in self.channel_positions and ch2 in self.channel_positions:
                        G.add_edge(ch1, ch2, weight=connectivity_matrix[i,j])
        
        # Create plot
        plt.figure(figsize=(12, 8))
        pos = nx.get_node_attributes(G, 'pos')
        
        # Draw edges with varying thickness based on weight
        weights = [G[u][v]['weight'] for u,v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=[w*3 for w in weights],
                             edge_color='gray', alpha=0.5)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=500,
                             node_color='lightblue',
                             alpha=0.8)
        
        # Add labels
        nx.draw_networkx_labels(G, pos)
        
        plt.title(title)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()

    def validate_and_visualize(self,
                             data: Dict[str, np.ndarray],
                             sfreq: float,
                             output_dir: str,
                             subject_id: str) -> None:
        """
        Combines Gunkelman validation with visualization.
        """
        validator = GunkelmanValidator()
        
        # Validate each channel
        validation_results = {}
        for ch_name, ch_data in data.items():
            validation_results[ch_name] = validator.validate_signal(ch_data, sfreq)
        
        # Create visualization with validation overlay
        fig = self.plot_3d_brain_activity(data)
        
        # Add validation indicators
        for ch_name, results in validation_results.items():
            if not results['passed_all_checks']:
                # Add warning indicators to the plot
                pos = self.channel_positions.get(ch_name)
                if pos:
                    plt.plot(pos[0], pos[1], 'rx', markersize=10)
        
        plt.savefig(f"{output_dir}/validated_brain_activity_{subject_id}.png")
        plt.close()
        
        return validation_results

def create_advanced_visualizations(data: Dict[str, Dict],
                                 output_dir: str,
                                 subject_id: str) -> None:
    """
    Create advanced visualizations including 3D brain activity and connectivity.
    
    Args:
        data: Dictionary containing EEG data and metrics
        output_dir: Directory to save visualizations
        subject_id: Subject identifier
    """
    visualizer = BrainVisualizer()
    
    # Create 3D brain activity visualization
    activity_data = {ch: data[ch].get('Alpha_Power', 0) for ch in data}
    visualizer.plot_3d_brain_activity(
        activity_data,
        title=f"3D Brain Activity Map - {subject_id}",
        save_path=f"{output_dir}/3d_brain_activity_{subject_id}.html"
    )
    
    # Create connectivity visualization
    # Example: using alpha power correlations between channels
    n_channels = len(data)
    connectivity_matrix = np.zeros((n_channels, n_channels))
    channel_names = list(data.keys())
    
    for i, ch1 in enumerate(channel_names):
        for j, ch2 in enumerate(channel_names):
            if i != j:
                # Example: compute correlation between channel metrics
                metrics1 = np.array([data[ch1].get(k, 0) for k in data[ch1]])
                metrics2 = np.array([data[ch2].get(k, 0) for k in data[ch2]])
                if len(metrics1) > 0 and len(metrics2) > 0:
                    correlation = np.corrcoef(metrics1, metrics2)[0,1]
                    connectivity_matrix[i,j] = abs(correlation)
    
    visualizer.plot_brain_connectivity(
        connectivity_matrix,
        channel_names,
        threshold=0.7,
        title=f"Brain Connectivity Network - {subject_id}",
        save_path=f"{output_dir}/brain_connectivity_{subject_id}.png"
    )

class GunkelmanValidator:
    """
    Implements Gunkelman-style validation traps and cross-checks for EEG data quality.
    Focuses on detecting subtle EMG contamination and other artifacts.
    """
    
    def __init__(self):
        self.frequency_bands = {
            'Delta': (1, 4),
            'Theta': (4, 8),
            'Alpha': (8, 13),
            'SMR': (12, 15),
            'Beta': (13, 30),
            'High_Beta': (20, 30),
            'Gamma': (30, 45)
        }
        
        # Thresholds based on Gunkelman's recommendations
        self.thresholds = {
            'emg_ratio': 2.5,  # Ratio of high-freq to mid-freq power
            'alpha_reactivity': 0.5,  # Minimum expected alpha reduction
            'ref_stability': 0.8,  # Minimum correlation between references
            'entropy_threshold': 0.7  # Maximum expected sample entropy
        }
    
    def check_emg_contamination(self, data: np.ndarray, sfreq: float) -> Dict[str, float]:
        """
        Implements multiple EMG detection traps:
        1. High-frequency power ratio check
        2. Temporal stability analysis
        3. Spectral edge frequency validation
        """
        results = {}
        
        # 1. High-frequency power ratio trap
        freqs, psd = sig.welch(data, fs=sfreq, nperseg=int(sfreq * 2))
        beta_mask = (freqs >= 13) & (freqs <= 30)
        high_freq_mask = (freqs > 30) & (freqs <= 100)
        
        beta_power = np.mean(psd[beta_mask])
        high_freq_power = np.mean(psd[high_freq_mask])
        emg_ratio = high_freq_power / beta_power
        
        results['emg_ratio'] = emg_ratio
        results['emg_detected'] = emg_ratio > self.thresholds['emg_ratio']
        
        # 2. Temporal stability analysis
        epoch_len = int(sfreq * 2)  # 2-second epochs
        n_epochs = len(data) // epoch_len
        epoch_vars = []
        
        for i in range(n_epochs):
            epoch = data[i*epoch_len:(i+1)*epoch_len]
            # Calculate high-frequency variance
            filtered = mne.filter.filter_data(
                epoch[None, :], sfreq, 30, 100, verbose=False)[0]
            epoch_vars.append(np.var(filtered))
        
        stability_score = np.std(epoch_vars) / np.mean(epoch_vars)
        results['temporal_stability'] = stability_score
        
        # 3. Spectral edge frequency check
        def spectral_edge_freq(psd, freqs, percent=0.95):
            cumsum = np.cumsum(psd)
            normalized_cumsum = cumsum / cumsum[-1]
            return freqs[np.where(normalized_cumsum >= percent)[0][0]]
        
        sef95 = spectral_edge_freq(psd, freqs)
        results['sef95'] = sef95
        results['sef_warning'] = sef95 > 30  # Warning if SEF95 > 30 Hz
        
        return results
    
    def validate_references(self, 
                          data_car: np.ndarray, 
                          data_mastoid: np.ndarray,
                          data_bipolar: np.ndarray) -> Dict[str, float]:
        """
        Cross-validates different reference montages to detect artifacts.
        Uses Gunkelman's reference stability criteria.
        """
        results = {}
        
        # 1. Compare CAR vs Mastoid
        car_mastoid_corr = np.corrcoef(data_car.ravel(), data_mastoid.ravel())[0,1]
        results['car_mastoid_correlation'] = car_mastoid_corr
        
        # 2. Bipolar consistency check
        # Compare variance ratios between montages
        var_ratio = np.var(data_bipolar) / np.var(data_car)
        results['bipolar_variance_ratio'] = var_ratio
        
        # 3. Reference stability warning
        results['reference_warning'] = (
            car_mastoid_corr < self.thresholds['ref_stability'] or
            var_ratio > 2.0 or var_ratio < 0.5
        )
        
        return results
    
    def check_alpha_reactivity(self, 
                             eyes_closed: np.ndarray, 
                             eyes_open: np.ndarray,
                             sfreq: float) -> Dict[str, float]:
        """
        Implements Gunkelman's alpha reactivity traps.
        """
        results = {}
        
        def get_alpha_power(data):
            freqs, psd = sig.welch(data, fs=sfreq, nperseg=int(sfreq * 2))
            alpha_mask = (freqs >= 8) & (freqs <= 13)
            return np.mean(psd[alpha_mask])
        
        alpha_ec = get_alpha_power(eyes_closed)
        alpha_eo = get_alpha_power(eyes_open)
        
        reactivity = (alpha_ec - alpha_eo) / alpha_ec
        results['alpha_reactivity'] = reactivity
        
        # Gunkelman's reactivity criteria
        results['reactivity_warning'] = reactivity < self.thresholds['alpha_reactivity']
        
        # Check alpha stability (shouldn't be too stable - could indicate artifact)
        alpha_stability = np.std(eyes_closed) / np.mean(np.abs(eyes_closed))
        results['alpha_stability'] = alpha_stability
        results['stability_warning'] = alpha_stability < 0.1  # Too stable is suspicious
        
        return results
    
    def robust_artifact_detection(self, data: np.ndarray) -> Dict[str, Union[float, bool]]:
        """
        Implements Gunkelman's robust statistical approaches for artifact detection.
        """
        results = {}
        
        # 1. Compare robust vs standard z-scores
        standard_z = zscore(data)
        robust_z = (data - np.median(data)) / median_abs_deviation(data)
        
        z_diff = np.mean(np.abs(standard_z - robust_z))
        results['z_score_difference'] = z_diff
        results['z_score_warning'] = z_diff > 1.5
        
        # 2. Sample Entropy (complexity measure)
        def sample_entropy(signal, m=2, r=0.2):
            """Simplified sample entropy calculation"""
            r = r * np.std(signal)
            N = len(signal)
            B = 0.0
            A = 0.0
            
            for i in range(N - m):
                for j in range(i + 1, N - m):
                    matches = 0
                    for k in range(m):
                        if abs(signal[i+k] - signal[j+k]) < r:
                            matches += 1
                    if matches == m:
                        B += 1
                        if abs(signal[i+m] - signal[j+m]) < r:
                            A += 1
                            
            return -np.log(A/B) if B > 0 and A > 0 else np.inf
        
        entropy_val = sample_entropy(data)
        results['sample_entropy'] = entropy_val
        results['entropy_warning'] = entropy_val > self.thresholds['entropy_threshold']
        
        return results
    
    def check_phase_relationships(self, 
                                data: np.ndarray, 
                                sfreq: float,
                                band_of_interest: Tuple[float, float] = (8, 13)) -> Dict[str, float]:
        """
        Implements Gunkelman's phase relationship checks to detect artifacts and validate signal quality.
        Particularly useful for detecting EMG contamination masquerading as neural activity.
        """
        results = {}
        
        # 1. Phase-Amplitude Coupling (PAC) Analysis
        def compute_pac(signal, phase_band, amp_band):
            # Extract phase of lower frequency band
            phase_filt = mne.filter.filter_data(
                signal[None, :], sfreq, phase_band[0], phase_band[1], verbose=False)[0]
            phase = np.angle(sig.hilbert(phase_filt))
            
            # Extract amplitude of higher frequency band
            amp_filt = mne.filter.filter_data(
                signal[None, :], sfreq, amp_band[0], amp_band[1], verbose=False)[0]
            amplitude = np.abs(sig.hilbert(amp_filt))
            
            # Compute modulation index
            n_bins = 18
            phase_bins = np.linspace(-np.pi, np.pi, n_bins+1)
            mean_amp = np.zeros(n_bins)
            
            for i in range(n_bins):
                mask = (phase >= phase_bins[i]) & (phase < phase_bins[i+1])
                mean_amp[i] = np.mean(amplitude[mask]) if np.any(mask) else 0
                
            # Normalize
            mean_amp = mean_amp / np.sum(mean_amp)
            
            # Compute modulation index using Kullback-Leibler divergence
            uniform = np.ones(n_bins) / n_bins
            return entropy(mean_amp, uniform)
        
        # Check PAC between different frequency bands
        pac_pairs = [
            ((4, 8), (30, 45)),   # Theta-Gamma
            ((8, 13), (30, 45)),  # Alpha-Gamma
            ((13, 30), (30, 45))  # Beta-Gamma
        ]
        
        pac_values = {}
        for phase_band, amp_band in pac_pairs:
            key = f"PAC_{phase_band[0]}-{phase_band[1]}_{amp_band[0]}-{amp_band[1]}"
            pac_values[key] = compute_pac(data, phase_band, amp_band)
        
        results['pac_values'] = pac_values
        
        # 2. Phase Locking Value (PLV) Analysis
        def compute_plv(signal1, signal2, freq_band):
            # Filter signals
            sig1_filt = mne.filter.filter_data(
                signal1[None, :], sfreq, freq_band[0], freq_band[1], verbose=False)[0]
            sig2_filt = mne.filter.filter_data(
                signal2[None, :], sfreq, freq_band[0], freq_band[1], verbose=False)[0]
            
            # Extract instantaneous phase
            phase1 = np.angle(sig.hilbert(sig1_filt))
            phase2 = np.angle(sig.hilbert(sig2_filt))
            
            # Compute PLV
            phase_diff = phase1 - phase2
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            return plv
        
        # 3. Phase Continuity Check
        def check_phase_continuity(signal, freq_band):
            # Filter signal
            sig_filt = mne.filter.filter_data(
                signal[None, :], sfreq, freq_band[0], freq_band[1], verbose=False)[0]
            
            # Extract instantaneous phase
            phase = np.angle(sig.hilbert(sig_filt))
            
            # Compute phase differences
            phase_diff = np.diff(phase)
            
            # Correct for phase wrapping
            phase_diff = np.mod(phase_diff + np.pi, 2 * np.pi) - np.pi
            
            # Calculate phase continuity score
            continuity_score = 1 - (np.std(phase_diff) / np.pi)
            return continuity_score
        
        # Compute phase continuity for band of interest
        phase_cont = check_phase_continuity(data, band_of_interest)
        results['phase_continuity'] = phase_cont
        
        # Set warnings based on Gunkelman's criteria
        results['warnings'] = []
        
        # Check for suspiciously high PAC values
        max_pac = max(pac_values.values())
        if max_pac > 0.3:  # Threshold based on Gunkelman's observations
            results['warnings'].append(
                f"Unusually high PAC detected ({max_pac:.2f}), possible EMG contamination"
            )
        
        # Check for suspicious phase continuity
        if phase_cont > 0.95:  # Too continuous might indicate artifact
            results['warnings'].append(
                "Suspiciously high phase continuity, possible filtering artifact"
            )
        elif phase_cont < 0.3:  # Too discontinuous might indicate EMG
            results['warnings'].append(
                "Low phase continuity, possible EMG contamination"
            )
        
        return results

    def check_coherence_validity(self,
                               data1: np.ndarray,
                               data2: np.ndarray,
                               sfreq: float) -> Dict[str, float]:
        """
        Implements Gunkelman's coherence validation checks.
        Helps identify spurious coherence from common reference or EMG.
        """
        results = {}
        
        # 1. Compute coherence across different frequency bands
        def compute_band_coherence(sig1, sig2, freq_band):
            freqs, coh = sig.coherence(sig1, sig2, fs=sfreq, 
                                     nperseg=int(sfreq * 2))
            mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
            return np.mean(coh[mask])
        
        # Compute coherence for each frequency band
        band_coherence = {}
        for band_name, (fmin, fmax) in self.frequency_bands.items():
            band_coherence[band_name] = compute_band_coherence(
                data1, data2, (fmin, fmax)
            )
        
        results['band_coherence'] = band_coherence
        
        # 2. Check for spurious coherence patterns
        
        # High frequency coherence check (EMG indicator)
        high_freq_coh = compute_band_coherence(data1, data2, (30, 100))
        results['high_freq_coherence'] = high_freq_coh
        
        # Compare high vs low frequency coherence
        low_freq_coh = np.mean([
            band_coherence['Delta'],
            band_coherence['Theta'],
            band_coherence['Alpha']
        ])
        
        coh_ratio = high_freq_coh / low_freq_coh if low_freq_coh > 0 else np.inf
        results['coherence_ratio'] = coh_ratio
        
        # 3. Instantaneous phase synchrony analysis
        def compute_windowed_sync(sig1, sig2, freq_band, window_size=1000):
            # Filter signals
            sig1_filt = mne.filter.filter_data(
                sig1[None, :], sfreq, freq_band[0], freq_band[1], verbose=False)[0]
            sig2_filt = mne.filter.filter_data(
                sig2[None, :], sfreq, freq_band[0], freq_band[1], verbose=False)[0]
            
            # Get instantaneous phases
            phase1 = np.angle(sig.hilbert(sig1_filt))
            phase2 = np.angle(sig.hilbert(sig2_filt))
            
            # Compute windowed synchrony
            n_windows = len(phase1) // window_size
            sync_values = []
            
            for i in range(n_windows):
                start = i * window_size
                end = start + window_size
                phase_diff = phase1[start:end] - phase2[start:end]
                sync = np.abs(np.mean(np.exp(1j * phase_diff)))
                sync_values.append(sync)
            
            return np.array(sync_values)
        
        # Compute alpha band synchrony stability
        alpha_sync = compute_windowed_sync(data1, data2, (8, 13))
        results['alpha_sync_stability'] = np.std(alpha_sync)
        
        # Generate warnings based on Gunkelman's criteria
        results['warnings'] = []
        
        if coh_ratio > 0.7:  # Threshold based on Gunkelman's observations
            results['warnings'].append(
                f"High frequency coherence ratio ({coh_ratio:.2f}) suggests possible EMG contamination"
            )
        
        if results['alpha_sync_stability'] < 0.05:
            results['warnings'].append(
                "Suspiciously stable alpha synchrony, possible reference artifact"
            )
        
        # Check for physiologically implausible coherence patterns
        implausible = False
        for band in ['Gamma', 'High_Beta']:
            if band_coherence[band] > band_coherence['Alpha'] * 1.5:
                implausible = True
                break
        
        if implausible:
            results['warnings'].append(
                "Physiologically implausible coherence pattern detected"
            )
        
        return results

    def validate_signal(self, 
                       data: np.ndarray,
                       sfreq: float,
                       montage_data: Dict[str, np.ndarray] = None) -> Dict[str, Dict]:
        """
        Comprehensive signal validation using all Gunkelman checks.
        """
        validation_results = {}
        
        # Existing checks
        validation_results['emg'] = self.check_emg_contamination(data, sfreq)
        validation_results['artifacts'] = self.robust_artifact_detection(data)
        
        # Add phase relationship checks
        validation_results['phase'] = self.check_phase_relationships(data, sfreq)
        
        # Reference validation if montage data provided
        if montage_data and all(k in montage_data for k in ['car', 'mastoid', 'bipolar']):
            validation_results['reference'] = self.validate_references(
                montage_data['car'], montage_data['mastoid'], montage_data['bipolar']
            )
            
            # Add coherence validation between references
            validation_results['coherence'] = self.check_coherence_validity(
                montage_data['car'], montage_data['mastoid'], sfreq
            )
        
        # Aggregate all warnings
        warnings = []
        for check_type, results in validation_results.items():
            if isinstance(results, dict) and 'warnings' in results:
                warnings.extend(results['warnings'])
        
        validation_results['warnings'] = warnings
        validation_results['passed_all_checks'] = len(warnings) == 0
        
        return validation_results

class DataValidator:
    """
    Implements comprehensive cross-validation methods for EEG data handling and mathematical operations.
    Works alongside GunkelmanValidator to provide additional layers of safety checks.
    """
    
    def __init__(self):
        self.expected_ranges = {
            'raw_eeg': (-500, 500),  # μV
            'power_bands': (0, 1000),  # μV²
            'coherence': (0, 1),
            'phase': (-np.pi, np.pi),
            'correlation': (-1, 1)
        }
        
        self.sampling_rates = [128, 250, 256, 500, 512, 1000, 1024, 2048]  # Common EEG sampling rates
        
    def validate_numerical_operations(self, 
                                   data: np.ndarray,
                                   operation_type: str) -> Dict[str, bool]:
        """
        Cross-validates numerical operations to detect computational artifacts.
        
        Args:
            data: Input data array
            operation_type: Type of operation being validated
        """
        results = {'valid': True, 'warnings': []}
        
        # Check for NaN/Inf values
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            results['valid'] = False
            results['warnings'].append(f"Invalid values detected in {operation_type}")
        
        # Check for numerical stability
        if operation_type == 'fft':
            # Verify FFT symmetry for real data
            fft_data = np.fft.fft(data)
            symmetry_error = np.max(np.abs(fft_data - np.conj(np.flip(fft_data))))
            if symmetry_error > 1e-10:
                results['warnings'].append("FFT symmetry violation detected")
        
        elif operation_type == 'filter':
            # Check for edge effects
            edge_variance = np.var(data[:100]) / np.var(data[100:-100])
            if edge_variance > 2.0:
                results['warnings'].append("Possible filter edge effects detected")
        
        elif operation_type == 'correlation':
            # Verify correlation matrix properties
            if data.shape[0] == data.shape[1]:  # Square matrix
                # Check symmetry
                if not np.allclose(data, data.T, rtol=1e-10):
                    results['warnings'].append("Correlation matrix asymmetry detected")
                # Check diagonal
                if not np.allclose(np.diag(data), 1.0, rtol=1e-10):
                    results['warnings'].append("Invalid correlation matrix diagonal")
        
        return results
    
    def validate_signal_properties(self,
                                 data: np.ndarray,
                                 sfreq: float,
                                 signal_type: str) -> Dict[str, bool]:
        """
        Validates signal properties using multiple cross-checking methods.
        """
        results = {'valid': True, 'warnings': []}
        
        # Validate sampling rate
        closest_standard_rate = min(self.sampling_rates, 
                                  key=lambda x: abs(x - sfreq))
        if abs(sfreq - closest_standard_rate) > 1:
            results['warnings'].append(
                f"Unusual sampling rate: {sfreq}Hz. Nearest standard rate: {closest_standard_rate}Hz"
            )
        
        # Validate amplitude range
        data_range = np.percentile(data, [0.1, 99.9])
        expected_min, expected_max = self.expected_ranges[signal_type]
        
        if data_range[0] < expected_min or data_range[1] > expected_max:
            results['warnings'].append(
                f"Signal amplitude outside expected range for {signal_type}"
            )
        
        # Check for DC offset
        dc_offset = np.mean(data)
        if abs(dc_offset) > 100:  # Arbitrary threshold for DC offset
            results['warnings'].append(f"Large DC offset detected: {dc_offset:.2f}")
        
        # Validate signal continuity
        discontinuities = np.where(np.diff(data) > np.std(data) * 5)[0]
        if len(discontinuities) > 0:
            results['warnings'].append(
                f"Signal discontinuities detected at samples: {discontinuities[:5]}"
            )
        
        return results
    
    def validate_spectral_properties(self,
                                   data: np.ndarray,
                                   sfreq: float) -> Dict[str, bool]:
        """
        Cross-validates spectral properties of the signal.
        """
        results = {'valid': True, 'warnings': []}
        
        # Compute power spectrum
        freqs, psd = sig.welch(data, fs=sfreq, nperseg=int(sfreq * 2))
        
        # Check for line noise
        line_freq = 60  # or 50 depending on region
        line_noise_mask = (freqs >= line_freq - 1) & (freqs <= line_freq + 1)
        line_noise_power = np.mean(psd[line_noise_mask])
        signal_power = np.mean(psd)
        
        if line_noise_power > signal_power * 0.5:
            results['warnings'].append("Significant line noise detected")
        
        # Check for aliasing
        nyquist = sfreq / 2
        high_freq_power = np.mean(psd[freqs > nyquist * 0.8])
        if high_freq_power > signal_power * 0.1:
            results['warnings'].append("Possible aliasing detected")
        
        # Validate spectral slope
        freq_mask = (freqs >= 1) & (freqs <= 40)  # Typical EEG range
        slope, _ = np.polyfit(np.log10(freqs[freq_mask]), 
                            np.log10(psd[freq_mask]), 1)
        
        if slope > -0.5:  # EEG typically has 1/f-like spectrum
            results['warnings'].append("Unusual spectral slope detected")
        
        return results
    
    def validate_preprocessing(self,
                             raw_data: np.ndarray,
                             processed_data: np.ndarray,
                             operation: str) -> Dict[str, bool]:
        """
        Cross-validates preprocessing operations.
        """
        results = {'valid': True, 'warnings': []}
        
        # Check for data loss
        if len(processed_data) < len(raw_data) * 0.9:  # Allow for some data loss
            results['warnings'].append(
                f"Significant data loss after {operation}"
            )
        
        # Check for variance changes
        raw_var = np.var(raw_data)
        processed_var = np.var(processed_data)
        var_ratio = processed_var / raw_var
        
        if var_ratio < 0.1 or var_ratio > 10:
            results['warnings'].append(
                f"Suspicious variance change after {operation}"
            )
        
        # Check for artificial periodicities
        raw_acf = np.correlate(raw_data, raw_data, mode='full')
        proc_acf = np.correlate(processed_data, processed_data, mode='full')
        
        raw_peaks = sig.find_peaks(raw_acf)[0]
        proc_peaks = sig.find_peaks(proc_acf)[0]
        
        if len(proc_peaks) > len(raw_peaks) * 2:
            results['warnings'].append(
                f"Artificial periodicities introduced by {operation}"
            )
        
        return results
    
    def cross_validate_with_gunkelman(self,
                                    data: np.ndarray,
                                    sfreq: float,
                                    gunkelman_results: Dict) -> Dict[str, bool]:
        """
        Cross-validates results with Gunkelman validator output.
        """
        results = {'valid': True, 'warnings': []}
        
        # Validate signal properties
        signal_props = self.validate_signal_properties(data, sfreq, 'raw_eeg')
        results['warnings'].extend(signal_props['warnings'])
        
        # Validate spectral properties
        spectral_props = self.validate_spectral_properties(data, sfreq)
        results['warnings'].extend(spectral_props['warnings'])
        
        # Cross-check with Gunkelman's EMG detection
        if gunkelman_results.get('emg', {}).get('emg_detected'):
            # Verify with spectral ratio
            freqs, psd = sig.welch(data, fs=sfreq, nperseg=int(sfreq * 2))
            high_freq_mask = (freqs >= 30) & (freqs <= 100)
            low_freq_mask = (freqs >= 1) & (freqs <= 30)
            
            spectral_ratio = (
                np.mean(psd[high_freq_mask]) / np.mean(psd[low_freq_mask])
            )
            
            if spectral_ratio < 0.1:  # Contradicts Gunkelman's EMG detection
                results['warnings'].append(
                    "Conflicting EMG detection results between methods"
                )
        
        # Cross-check phase continuity
        if 'phase' in gunkelman_results:
            phase_cont = gunkelman_results['phase'].get('phase_continuity', 0)
            hilbert_phase = np.angle(sig.hilbert(data))
            computed_cont = 1 - (np.std(np.diff(hilbert_phase)) / np.pi)
            
            if abs(phase_cont - computed_cont) > 0.2:
                results['warnings'].append(
                    "Inconsistent phase continuity measures"
                )
        
        return results

def validate_and_cross_check(data: np.ndarray,
                           sfreq: float,
                           montage_data: Dict[str, np.ndarray] = None) -> Dict:
    """
    Comprehensive validation combining both validators.
    """
    gunkelman = GunkelmanValidator()
    data_validator = DataValidator()
    
    # Run Gunkelman's validation
    gunkelman_results = gunkelman.validate_signal(data, sfreq, montage_data)
    
    # Run data validation
    data_validation = {
        'numerical': data_validator.validate_numerical_operations(
            data, 'raw_signal'
        ),
        'signal_props': data_validator.validate_signal_properties(
            data, sfreq, 'raw_eeg'
        ),
        'spectral': data_validator.validate_spectral_properties(
            data, sfreq
        ),
        'cross_check': data_validator.cross_validate_with_gunkelman(
            data, sfreq, gunkelman_results
        )
    }
    
    # Combine warnings
    all_warnings = []
    all_warnings.extend(gunkelman_results.get('warnings', []))
    
    for validation_type, results in data_validation.items():
        all_warnings.extend(results.get('warnings', []))
    
    # Final validation result
    return {
        'gunkelman_validation': gunkelman_results,
        'data_validation': data_validation,
        'all_warnings': all_warnings,
        'passed_all_checks': len(all_warnings) == 0
    }

def plot_band_topomap(raw: mne.io.Raw, band_name: str, instability_values: Dict[str, float] = None, 
                    validations: Dict[str, str] = None) -> plt.Figure:
    """
    Plot a topomap for a specific frequency band with instability markers.
    
    Args:
        raw (mne.io.Raw): Raw EEG data
        band_name (str): Name of the frequency band
        instability_values (Dict[str, float]): Dictionary of channel instability values
        validations (Dict[str, str]): Dictionary of channel instability validations
        
    Returns:
        plt.Figure: The generated matplotlib figure
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Get band power data
    band_power = compute_band_power(raw, band_name)
    
    # Plot band power topomap
    mne.viz.plot_topomap(band_power, raw.info, axes=ax1, show=False)
    ax1.set_title(f'{band_name} Band Power')
    
    # Create instability visualization
    if instability_values and validations:
        # Convert instability values to array matching channel order
        inst_data = np.array([instability_values.get(ch, 0) for ch in raw.ch_names])
        
        # Create custom colormap for instability levels
        colors = ['green', 'yellow', 'red']
        n_bins = 256
        cmap = mpl.colors.LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
        
        # Plot instability topomap
        im = mne.viz.plot_topomap(inst_data, raw.info, axes=ax2, 
                                 cmap=cmap, vmin=0, vmax=1, show=False)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Instability Index')
        
        # Add markers for validation levels
        for ch_idx, ch_name in enumerate(raw.ch_names):
            validation = validations.get(ch_name, 'normal')
            if validation != 'normal':
                pos = raw.info['chs'][ch_idx]['loc'][:2]
                marker = 'x' if validation == 'critical' else 'o'
                color = 'red' if validation == 'critical' else 'orange'
                ax2.plot(pos[0], pos[1], marker, color=color, markersize=10, markeredgewidth=2)
        
        ax2.set_title(f'{band_name} Instability Map')
    
    plt.tight_layout()
    return fig
