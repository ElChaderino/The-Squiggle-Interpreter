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
import mne
import numpy as np
import scipy.signal as sig


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
    Generate side-by-side topomaps for absolute and relative power.

    Parameters:
      abs_vals (array-like): Absolute power values per channel.
      rel_vals (array-like): Relative power values per channel.
      info (mne.Info): MNE Info with channel locations.
      band_name (str): Name of the frequency band.
      cond_name (str): Condition label (e.g., "EO" or "EC").

    Returns:
      matplotlib.figure.Figure: The generated topomap figure.
    """
    info_clean, sel_idx = remove_overlapping_channels(info)
    abs_vals_subset = np.array(abs_vals)[sel_idx]
    rel_vals_subset = np.array(rel_vals)[sel_idx]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor='black')
    fig.patch.set_facecolor('black')

    ax_abs = axes[0]
    ax_abs.set_facecolor('black')
    im_abs, _ = mne.viz.plot_topomap(abs_vals_subset, info_clean, axes=ax_abs, cmap='viridis', show=False)
    ax_abs.set_title(f"{band_name} Abs Power ({cond_name})", color='white', fontsize=10)
    cbar_abs = plt.colorbar(im_abs, ax=ax_abs, orientation='horizontal', fraction=0.05, pad=0.08)
    cbar_abs.ax.tick_params(colors='white')

    ax_rel = axes[1]
    ax_rel.set_facecolor('black')
    im_rel, _ = mne.viz.plot_topomap(rel_vals_subset, info_clean, axes=ax_rel, cmap='viridis', show=False)
    ax_rel.set_title(f"{band_name} Rel Power ({cond_name})", color='white', fontsize=10)
    cbar_rel = plt.colorbar(im_rel, ax=ax_rel, orientation='horizontal', fraction=0.05, pad=0.08)
    cbar_rel.ax.tick_params(colors='white')

    fig.suptitle(f"Global Topomap (Abs & Rel) - {band_name} ({cond_name})", color='white')
    fig.tight_layout()
    return fig


# ------------------ Waveform Grid ------------------

def plot_waveform_grid(data, ch_names, sfreq, band=(8, 12), epoch_length=10):
    """
    Generate a grid of waveform plots for all channels for a given frequency band.

    Parameters:
      data (np.array): 2D array (n_channels x n_samples).
      ch_names (list): List of channel names.
      sfreq (float): Sampling frequency.
      band (tuple): Frequency band (fmin, fmax) for filtering.
      epoch_length (float): Duration (in seconds) of the plotted segment.

    Returns:
      matplotlib.figure.Figure: The generated waveform grid figure.
    """
    n_channels = data.shape[0]
    n_samples = int(sfreq * epoch_length)
    t = np.linspace(0, epoch_length, n_samples, endpoint=False)
    n_cols = 4
    n_rows = int(np.ceil(n_channels / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3), facecolor='black')
    fig.patch.set_facecolor('black')
    axes = np.atleast_2d(axes)

    for i in range(n_channels):
        ax = axes[i // n_cols, i % n_cols]
        ax.set_facecolor('black')
        filt = mne.filter.filter_data(data[i], sfreq, band[0], band[1], verbose=False)
        ax.plot(t, filt[:n_samples], color="cyan", lw=1)
        ax.set_title(ch_names[i], color='white', fontsize=8)
        ax.set_xlim([0, epoch_length])
        ax.tick_params(colors='white', labelsize=7)

    for j in range(n_channels, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis('off')

    fig.suptitle(f"Global Waveform Grid (EO, {band[0]}-{band[1]}Hz)", color='white', fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


# ------------------ Coherence Matrix ------------------

def plot_coherence_matrix(coh_matrix, ch_names=None):
    """
    Generate a heatmap of the global coherence matrix with keys and axis labels.

    Parameters:
      coh_matrix (np.array): Coherence matrix.
      ch_names (list, optional): List of channel names. If provided (and its length matches the
                                 matrix dimensions), these are used for the x and y tick labels.

    Returns:
      matplotlib.figure.Figure: The coherence heatmap figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='black')
    cax = ax.imshow(coh_matrix, cmap='viridis', aspect='auto')
    ax.set_title("Global Coherence Matrix", color='white')
    if ch_names is not None and len(ch_names) == coh_matrix.shape[0]:
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
    return fig


# ------------------ PSD & Waveform Overlays ------------------

def plot_band_psd_overlay(eo_sig, ec_sig, sfreq, band, ch_name, band_name, colors):
    """
    Plot PSD overlay for a given channel comparing EO vs. EC.

    Parameters:
      eo_sig, ec_sig (np.array): 1D signals for EO and EC.
      sfreq (float): Sampling frequency.
      band (tuple): Frequency band (fmin, fmax).
      ch_name (str): Channel name.
      band_name (str): Frequency band name.
      colors (tuple): Colors for EO and EC.

    Returns:
      matplotlib.figure.Figure: The PSD overlay figure.
    """
    fmin, fmax = band
    freqs_eo, psd_eo = sig.welch(eo_sig, fs=sfreq, nperseg=int(sfreq * 2))
    freqs_ec, psd_ec = sig.welch(ec_sig, fs=sfreq, nperseg=int(sfreq * 2))

    mask_eo = (freqs_eo >= fmin) & (freqs_eo <= fmax)
    mask_ec = (freqs_ec >= fmin) & (freqs_ec <= fmax)
    psd_eo_db = 10 * np.log10(psd_eo[mask_eo])
    psd_ec_db = 10 * np.log10(psd_ec[mask_ec])

    fig, ax = plt.subplots(figsize=(6, 4), facecolor='black')
    ax.set_facecolor('black')
    ax.plot(freqs_eo[mask_eo], psd_eo_db, linestyle='-', color=colors[0], label='EO')
    ax.plot(freqs_ec[mask_ec], psd_ec_db, linestyle='--', color=colors[1], label='EC')
    ax.set_title(f"{ch_name} {band_name} PSD", color='white', fontsize=10)
    ax.set_xlabel("Frequency (Hz)", color='white')
    ax.set_ylabel("Power (dB)", color='white')
    ax.tick_params(colors='white')
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_band_waveform_overlay(eo_sig, ec_sig, sfreq, band, ch_name, band_name, colors, epoch_length=10):
    """
    Plot waveform overlays for a given channel comparing EO vs. EC after band-pass filtering.

    Parameters:
      eo_sig, ec_sig (np.array): 1D signals for EO and EC.
      sfreq (float): Sampling frequency.
      band (tuple): Frequency band (fmin, fmax).
      ch_name (str): Channel name.
      band_name (str): Frequency band name.
      colors (tuple): Colors for EO and EC.
      epoch_length (float): Duration (in seconds) for the plot.

    Returns:
      matplotlib.figure.Figure: The waveform overlay figure.
    """
    fmin, fmax = band
    filt_eo = mne.filter.filter_data(eo_sig, sfreq, fmin, fmax, verbose=False)
    filt_ec = mne.filter.filter_data(ec_sig, sfreq, fmin, fmax, verbose=False)

    n_samples = int(sfreq * epoch_length)
    epoch_eo = filt_eo[:n_samples]
    epoch_ec = filt_ec[:n_samples]
    t = np.linspace(0, epoch_length, n_samples, endpoint=False)

    fig, ax = plt.subplots(figsize=(6, 4), facecolor='black')
    ax.set_facecolor('black')
    ax.plot(t, epoch_eo, linestyle='-', color=colors[0], label='EO')
    ax.plot(t, epoch_ec, linestyle='--', color=colors[1], label='EC')
    ax.set_title(f"{ch_name} {band_name} Waveform", color='white', fontsize=10)
    ax.set_xlabel("Time (s)", color='white')
    ax.set_ylabel("Amplitude (µV)", color='white')
    ax.tick_params(colors='white')
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ------------------ Z-Score Topomap ------------------

def plot_zscore_topomap(z_scores, info, band_name, cond_name):
    """
    Generate a topomap of robust z-scores for a specified frequency band and condition.

    Parameters:
      z_scores (array-like): Z-scores per channel.
      info (mne.Info): MNE info with channel locations.
      band_name (str): Frequency band name.
      cond_name (str): Condition label.

    Returns:
      matplotlib.figure.Figure: The z-score topomap figure.
    """
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='black')
    ax.set_facecolor('black')
    im, _ = mne.viz.plot_topomap(z_scores, info, axes=ax, cmap='coolwarm', show=False)
    im.set_clim(-3, 3)
    ax.set_title(f"{band_name} Robust Z-Score ({cond_name})", color='white', fontsize=10)
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal')
    cbar.ax.tick_params(colors='white')
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
