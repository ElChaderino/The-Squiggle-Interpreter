"""
Advanced Signal Processing and Analysis Module

This module provides functions for:
  - Basic power analysis for predefined frequency bands.
  - Derived metrics (percentage change, theta–beta ratio).
  - Advanced metrics: entropy (sample, spectral, LZiv), fractal dimension (Higuchi),
    and detrended fluctuation analysis (DFA) on epochs.
  - Robust z-score computation using median and MAD.
  - Advanced analysis functions: Phase–Amplitude Coupling (PAC) analysis, Brain Symmetry Index (BSI),
    and connectivity (coherence) analysis.
  - Pseudo-ERP computation.
  - Time-Frequency Representation (TFR) computation using Morlet wavelets.
  - ICA computation.
  - Additionally, compute_all_zscore_maps() and compute_all_tfr_maps() are added to compute robust
    z-score topomaps and TFR maps for each frequency band.
  - **New: Source Localization Functions** – compute_inverse_operator() and compute_source_localization().

Dependencies:
  - numpy
  - mne
  - scipy (for coherence, welch, and hilbert)
  - matplotlib.pyplot
  - antropy (for sample_entropy, spectral_entropy, lziv_complexity)
  - nolds (for DFA)
"""

import numpy as np
import mne
from scipy.signal import coherence, welch, hilbert
import matplotlib.pyplot as plt
import nolds
from antropy import sample_entropy, spectral_entropy, lziv_complexity
import logging
from pathlib import Path
from . import plotting
import time
import multiprocessing as mp
import sys
import os

# Simple local implementation for now if utils not created
def execute_task(func, args):
    try:
        return func(*args)
    except Exception as e:
        logging.error(f"Error in task {func.__name__}: {e}", exc_info=True)
        return None

def execute_task_with_queue(func, args, queue):
    try:
        result = func(*args)
        queue.put(result)
    except Exception as e:
        logging.error(f"Error executing {func.__name__} in queue task: {e}", exc_info=True)
        queue.put(f"ERROR: {e}")

logger = logging.getLogger(__name__)

# Define standard frequency bands.
BANDS = {
    "Delta": (1, 4),
    "Theta": (4, 8),
    "Alpha": (8, 12),
    "SMR": (12, 15),
    "Beta": (15, 27),
    "HighBeta": (28, 38),
}

# ------------------ Basic Metrics ------------------

def compute_band_power(data, sfreq, band):
    """
    Filter the signal to a specified frequency band and compute mean power.
    
    Parameters:
      data (np.array): 1D signal array.
      sfreq (float): Sampling frequency in Hz.
      band (tuple): Frequency band (fmin, fmax).
    
    Returns:
      float: Mean power in the band.
    """
    fmin, fmax = band
    data_filt = mne.filter.filter_data(data, sfreq, fmin, fmax, verbose=False)
    return np.mean(data_filt ** 2)

def compute_all_band_powers(raw):
    """
    Compute mean power for each channel and for all defined frequency bands.
    
    Parameters:
      raw (mne.io.Raw): MNE Raw object.
      
    Returns:
      dict: {channel: {band: power, ...}, ...}
    """
    sfreq = raw.info["sfreq"]
    data = raw.get_data() * 1e6  # Convert to microvolts.
    results = {}
    for i, ch in enumerate(raw.ch_names):
        results[ch] = {}
        for band_name, band_range in BANDS.items():
            results[ch][band_name] = compute_band_power(data[i], sfreq, band_range)
    return results

def compute_percentage_change(power_EO, power_EC):
    """
    Compute percentage change from EO to EC.
    
    Parameters:
      power_EO (float): Power in EO condition.
      power_EC (float): Power in EC condition.
    
    Returns:
      float: Percentage change.
    """
    if power_EO == 0:
        return np.nan
    return ((power_EC - power_EO) / power_EO) * 100

def compute_theta_beta_ratio(raw, ch):
    """
    Compute the Theta/Beta power ratio for a specific channel.
    
    Parameters:
      raw (mne.io.Raw): MNE Raw object.
      ch (str): Channel name.
    
    Returns:
      float: Theta/Beta ratio.
    """
    sfreq = raw.info["sfreq"]
    data = raw.get_data(picks=[ch])[0] * 1e6
    theta = compute_band_power(data, sfreq, BANDS["Theta"])
    beta = compute_band_power(data, sfreq, BANDS["Beta"])
    if beta == 0:
        return np.nan
    return theta / beta

# ------------------ Advanced Metrics ------------------

def higuchi_fd(X, kmax):
    """
    Compute the Higuchi Fractal Dimension of a signal.
    
    Parameters:
      X (np.array): 1D signal.
      kmax (int): Maximum interval.
      
    Returns:
      float: Estimated fractal dimension.
    """
    N = len(X)
    L = []
    x_vals = []
    for k in range(1, kmax + 1):
        Lk = []
        for m in range(k):
            idxs = np.arange(1, int(np.floor((N - m) / k)), dtype=int)
            if len(idxs) == 0:
                continue
            Lm = np.sum(np.abs(X[m + idxs * k] - X[m + k * (idxs - 1)]))
            norm = (N - 1) / (len(idxs) * k)
            Lk.append(Lm * norm)
        if Lk:
            L.append(np.mean(Lk))
            x_vals.append(np.log(1.0 / k))
    L = np.log(np.array(L))
    slope = np.polyfit(x_vals, L, 1)[0]
    return slope

def compute_dfa_on_epochs(signal, sfreq, epoch_len_sec=2.0):
    """
    Compute Detrended Fluctuation Analysis (DFA) on epochs of the signal.
    
    Parameters:
      signal (np.array): 1D signal.
      sfreq (float): Sampling frequency.
      epoch_len_sec (float): Epoch duration in seconds.
      
    Returns:
      float or None: Average DFA value across epochs.
    """
    epoch_len = int(epoch_len_sec * sfreq)
    n_epochs = len(signal) // epoch_len
    if n_epochs == 0:
        return None
    dfa_vals = []
    for i in range(n_epochs):
        epoch = signal[i * epoch_len : (i + 1) * epoch_len]
        try:
            dfa_vals.append(nolds.dfa(epoch))
        except Exception:
            continue
    return np.mean(dfa_vals) if dfa_vals else None

def compute_entropy_on_epochs(signal, sfreq, epoch_len_sec=2.0):
    """
    Compute epoch-averaged entropy metrics:
      - Sample Entropy
      - Spectral Entropy
      - Lempel-Ziv Complexity
      
    Parameters:
      signal (np.array): 1D signal.
      sfreq (float): Sampling frequency.
      epoch_len_sec (float): Epoch duration in seconds.
      
    Returns:
      tuple: (average sample entropy, average spectral entropy, average LZ complexity)
    """
    epoch_len = int(epoch_len_sec * sfreq)
    n_epochs = len(signal) // epoch_len
    if n_epochs == 0:
        return None, None, None
    sampents, specents, lzivs = [], [], []
    for i in range(n_epochs):
        epoch = signal[i * epoch_len : (i + 1) * epoch_len]
        try:
            sampents.append(sample_entropy(epoch))
            specents.append(spectral_entropy(epoch, sfreq, method="welch", normalize=True))
            lzivs.append(lziv_complexity(epoch))
        except Exception:
            continue
    return (np.mean(sampents) if sampents else None,
            np.mean(specents) if specents else None,
            np.mean(lzivs) if lzivs else None)

def compute_robust_zscore(power, norm_median, norm_mad):
    """
    Compute robust z-score using median and median absolute deviation (MAD).
    
    Parameters:
      power (float): Observed power.
      norm_median (float): Normative median.
      norm_mad (float): Normative MAD.
      
    Returns:
      float: Robust z-score.
    """
    if norm_mad == 0:
        norm_mad = 1
    return (power - norm_median) / norm_mad

def compute_all_zscore_maps(raw, norm_stats_dict, epoch_len_sec=2.0):
    """Compute robust z-score maps for each frequency band.
    
    If norm_stats_dict is None, computes z-scores relative to the current data's median/MAD.

    Parameters:
      raw (mne.io.Raw): Raw EEG data.
      norm_stats_dict (dict): Normative statistics for each band.
      epoch_len_sec (float): Epoch duration in seconds.
      
    Returns:
      dict: {band: list of z-scores for each channel}
    """
    sfreq = raw.info["sfreq"]
    data = raw.get_data() * 1e6
    zscore_maps = {}
    for band, band_range in BANDS.items():
        band_powers = [compute_band_power(data[i], sfreq, band_range) for i in range(data.shape[0])]
        
        # Handle the case where pre-computed norms are not provided
        if norm_stats_dict is None:
            # Calculate median and MAD from the current data's band powers
            current_median = np.median(band_powers)
            current_mad = np.median(np.abs(band_powers - current_median))
            stats = {"median": current_median, "mad": current_mad}
        else:
            # Use pre-computed norms, falling back to current data if band missing
            stats = norm_stats_dict.get(band, {
                "median": np.median(band_powers),
                "mad": np.median(np.abs(band_powers - np.median(band_powers)))
            })
            
        # Ensure MAD is not zero to avoid division by zero
        if stats["mad"] == 0:
            stats["mad"] = 1e-6 # Use a small value instead of zero
            
        z_scores = [(p - stats["median"]) / stats["mad"] for p in band_powers]
        zscore_maps[band] = z_scores
    return zscore_maps

# ------------------ Advanced Analysis Functions ------------------

def compute_modulation_index(phase_data, amp_data, n_bins=18):
    """
    Compute a modulation index (MI) using KL divergence.
    
    Parameters:
      phase_data (np.array): Phase signal.
      amp_data (np.array): Amplitude envelope signal.
      n_bins (int): Number of bins for phase.
      
    Returns:
      float: Modulation index.
    """
    phase = np.angle(hilbert(phase_data))
    amp_env = np.abs(hilbert(amp_data))
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    digitized = np.digitize(phase, bins) - 1
    bin_means = np.array([amp_env[digitized == i].mean() for i in range(n_bins)])
    bin_prob = bin_means / (bin_means.sum() + 1e-10)
    uniform_prob = 1.0 / n_bins
    kl_div = np.sum(bin_prob * np.log((bin_prob + 1e-10) / uniform_prob))
    max_kl = np.log(n_bins)
    return kl_div / max_kl

def run_pac_analysis(data_arr, channels, sfreq, low_freqs, high_freqs, pac_scale_factor, output_dir, condition_name):
    """
    Compute phase-amplitude coupling (PAC) for each channel.
    
    Parameters:
      data_arr (np.array): 2D array (n_channels x n_samples).
      channels (list): List of channel names.
      sfreq (float): Sampling frequency.
      low_freqs (list): List of low-frequency bands (tuples).
      high_freqs (list): List of high-frequency bands (tuples).
      pac_scale_factor (float): Scaling factor for PAC values.
      output_dir (str): Directory to save PAC plots.
      condition_name (str): Condition label ("EO" or "EC").
      
    Returns:
      np.array: PAC matrix of shape (n_low, n_high, n_channels).
    """
    n_lf = len(low_freqs)
    n_hf = len(high_freqs)
    n_ch = len(channels)
    pac_matrix = np.zeros((n_lf, n_hf, n_ch))
    for i, (lf1, lf2) in enumerate(low_freqs):
        for j, (hf1, hf2) in enumerate(high_freqs):
            for k in range(n_ch):
                phase_sig = mne.filter.filter_data(data_arr[k], sfreq, lf1, lf2, verbose=False)
                amp_sig = mne.filter.filter_data(data_arr[k], sfreq, hf1, hf2, verbose=False)
                pac_matrix[i, j, k] = compute_modulation_index(phase_sig, amp_sig)
    pac_matrix *= pac_scale_factor
    for k, ch in enumerate(channels):
        vals = pac_matrix[:, :, k]
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='black')
        ax.imshow(vals, cmap='plasma', aspect='auto', interpolation='nearest')
        ax.set_title(f"PAC Heatmap - {ch} ({condition_name})", color='white', fontsize=10)
        ax.set_xticks(range(len(high_freqs)))
        ax.set_xticklabels([f"{hf[0]}-{hf[1]}Hz" for hf in high_freqs], color='white', rotation=45)
        ax.set_yticks(range(len(low_freqs)))
        ax.set_yticklabels([f"{lf[0]}-{lf[1]}Hz" for lf in low_freqs], color='white')
        fig.tight_layout()
        pac_path = os.path.join(output_dir, f"PAC_{ch}_{condition_name}.png")
        fig.savefig(pac_path, facecolor='black')
        plt.close(fig)
    return pac_matrix

def compute_coherence_matrix(data, sfreq, band, nperseg):
    """
    Compute a coherence matrix for all channels in a specified frequency band.
    
    Parameters:
      data (np.array): 2D array (n_channels x n_samples).
      sfreq (float): Sampling frequency.
      band (tuple): Frequency band.
      nperseg (int): nperseg parameter for coherence.
      
    Returns:
      np.array: Coherence matrix.
    """
    n_channels = data.shape[0]
    coh_matrix = np.zeros((n_channels, n_channels))
    for i in range(n_channels):
        for j in range(i+1, n_channels):
            f, Cxy = coherence(data[i], data[j], sfreq, nperseg=nperseg)
            mask = (f >= band[0]) & (f <= band[1])
            coh_val = np.mean(Cxy[mask])
            coh_matrix[i, j] = coh_val
            coh_matrix[j, i] = coh_val
    return coh_matrix

def compute_connectivity_coherence(data, sfreq, band, nperseg):
    """
    Compute connectivity coherence matrix (wrapper for compute_coherence_matrix).
    
    Parameters:
      data (np.array): 2D array (n_channels x n_samples).
      sfreq (float): Sampling frequency.
      band (tuple): Frequency band.
      nperseg (int): nperseg parameter for coherence.
      
    Returns:
      np.array: Coherence matrix.
    """
    return compute_coherence_matrix(data, sfreq, band, nperseg)

# ------------------ Pseudo-ERP Computation ------------------

def compute_pseudo_erp(raw):
    """
    Compute a pseudo-ERP by epoching the raw data using fixed-length events and averaging.
    
    Parameters:
      raw (mne.io.Raw): Raw EEG data.
      
    Returns:
      matplotlib.figure.Figure: Figure displaying the evoked response with dark styling.
    """
    events = mne.make_fixed_length_events(raw, duration=2.0)
    epochs = mne.Epochs(raw, events, tmin=-0.1, tmax=0.4, baseline=(None, 0),
                        preload=True, verbose=False)
    evoked = epochs.average()
    fig = evoked.plot(spatial_colors=True, show=False)
    # Apply dark styling to the figure and its axes:
    fig.patch.set_facecolor('black')
    for ax in fig.get_axes():
         ax.set_facecolor('black')
         ax.tick_params(colors='white')
         for label in ax.get_xticklabels() + ax.get_yticklabels():
              label.set_color('white')
         leg = ax.get_legend()
         if leg:
              for text in leg.get_texts():
                   text.set_color('white')
    return fig

# ------------------ Time-Frequency Representation (TFR) ------------------

def compute_tfr(raw, freqs, n_cycles, tmin=0.0, tmax=2.0):
    """
    Compute Time-Frequency Representation (TFR) using Morlet wavelets.
    
    Parameters:
      raw (mne.io.Raw): Raw EEG data.
      freqs (array-like): Frequencies of interest.
      n_cycles (array-like or float): Number of cycles.
      tmin (float): Start time for epochs.
      tmax (float): End time for epochs.
      
    Returns:
      mne.time_frequency.AverageTFR: The computed TFR.
    """
    events = mne.make_fixed_length_events(raw, duration=tmax-tmin)
    epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose=False)
    tfr = epochs.compute_tfr(method="morlet", freqs=freqs, n_cycles=n_cycles,
                             decim=3, n_jobs=1, verbose=False)
    return tfr

def compute_tfr_for_band(raw, band_range, n_cycles, tmin=0.0, tmax=2.0):
    """
    Compute a TFR for a given frequency band.
    Adjusts the epoch window if necessary so that wavelets fit within the signal.
    
    Parameters:
      raw (mne.io.Raw): Raw EEG data.
      band_range (tuple): Frequency band (fmin, fmax).
      n_cycles (float or array): Number of cycles.
      tmin (float): Start time for epochs.
      tmax (float): End time for epochs.
      
    Returns:
      mne.time_frequency.AverageTFR or None: Averaged TFR, or None on error.
    """
    required_duration = n_cycles / band_range[0]
    actual_duration = tmax - tmin
    if actual_duration < required_duration:
        center = (tmax + tmin) / 2.0
        tmin = center - required_duration / 2.0
        tmax = center + required_duration / 2.0
        print(f"Adjusted epoch window to ({tmin:.2f}, {tmax:.2f}) for band {band_range}")
    try:
        events = mne.make_fixed_length_events(raw, duration=tmax-tmin)
        epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose=False)
        freqs = np.linspace(band_range[0], band_range[1], 10)
        tfr = epochs.compute_tfr(method="morlet", freqs=freqs, n_cycles=n_cycles,
                                 decim=3, n_jobs=1, verbose=False)
        return tfr.average()
    except Exception as e:
        print(f"Error computing TFR for band {band_range}: {e}")
        return None

def compute_all_tfr_maps(raw, n_cycles, tmin=0.0, tmax=2.0):
    """
    Compute TFR maps for each frequency band.
    
    Parameters:
      raw (mne.io.Raw): Raw EEG data.
      n_cycles (float): Number of cycles for TFR.
      tmin (float): Start time for epochs.
      tmax (float): End time for epochs.
      
    Returns:
      dict: {band: AverageTFR object or None}
    """
    tfr_maps = {}
    for band, band_range in BANDS.items():
        tfr_maps[band] = compute_tfr_for_band(raw, band_range, n_cycles, tmin, tmax)
    return tfr_maps

# ------------------ ICA Computation ------------------

def compute_ica(raw, n_components=0.95, method='fastica', random_state=42):
    """
    Compute ICA on the raw data after applying a high-pass filter at 1 Hz.
    
    Parameters:
      raw (mne.io.Raw): Raw EEG data.
      n_components (float or int): Number of components.
      method (str): ICA method.
      random_state (int): Random seed.
      
    Returns:
      mne.preprocessing.ICA: Fitted ICA object.
    """
    raw_copy = raw.copy().filter(l_freq=1, h_freq=None, verbose=False)
    ica = mne.preprocessing.ICA(n_components=n_components, method=method, random_state=random_state, verbose=False)
    ica.fit(raw_copy, verbose=False)
    return ica

# ------------------ Source Localization Functions ------------------

def compute_inverse_operator(raw, fwd, cov, loose=0.2, depth=0.8):
    """
    Compute the inverse operator using the forward solution and noise covariance.
    
    Parameters:
      raw (mne.io.Raw): Raw EEG data.
      fwd (mne.Forward): Forward solution.
      cov (mne.Covariance): Noise covariance.
      loose (float): Loose orientation parameter.
      depth (float): Depth weighting parameter.
      
    Returns:
      mne.minimum_norm.InverseOperator: The computed inverse operator.
    """
    inv_op = mne.minimum_norm.make_inverse_operator(raw.info, fwd, cov,
                                                    loose=loose, depth=depth, verbose=False)
    return inv_op

def compute_source_localization(evoked, inv_op, lambda2=1.0/9.0, method="sLORETA"):
    """
    Compute the source localization (source estimate) from the evoked response.
    
    Parameters:
      evoked (mne.Evoked): Evoked response.
      inv_op (mne.minimum_norm.InverseOperator): Inverse operator.
      lambda2 (float): Regularization parameter.
      method (str): Inverse solution method (e.g., "sLORETA", "MNE" for LORETA).
      
    Returns:
      mne.SourceEstimate: The source estimate.
    """
    try:
         stc = mne.minimum_norm.apply_inverse(evoked, inv_op, lambda2=lambda2, method=method, pick_ori=None, verbose=False)
         return stc
    except Exception as e:
         logger.error(f"Error applying inverse solution with method {method}: {e}")
         return None

# --- Moved Helper from main.py ---
def compute_source_for_band(cond, raw_data, inv_op, band, folders, source_methods, subjects_dir):
    """Computes and saves source estimates for a specific band.

    Called in parallel by run_source_localization_analysis.
    Needs access to BANDS, compute_source_localization, plotting.plot_source_estimate.

    Returns:
        list[tuple[str, str, str, str]]: List of (cond, band, method, rel_filename)
    """
    results_for_band = []
    try:
        logger.info(f"    Computing sources for: {cond} - {band}")
        band_range = BANDS[band]
        raw_band = raw_data.copy().filter(band_range[0], band_range[1], verbose=False)
        events = mne.make_fixed_length_events(raw_band, duration=2.0)
        if len(events) < 1:
             logging.warning(f"    Skipping {cond}-{band}: Not enough events after filtering.")
             return []
        epochs = mne.Epochs(raw_band, events, tmin=-0.1, tmax=0.4, baseline=(None, 0), preload=True, verbose=False)
        if len(epochs) < 1:
            logging.warning(f"    Skipping {cond}-{band}: No valid epochs created.")
            return []
        evoked = epochs.average()

        subject_folder = Path(folders["subject"]) 
        cond_folder = Path(folders["plots_src"]) / cond 
        cond_folder.mkdir(parents=True, exist_ok=True)

        for method, method_label in source_methods.items():
            try:
                stc = compute_source_localization(evoked, inv_op, method=method_label) 
                if stc is None: 
                     logger.warning(f"    STC computation failed for {cond}-{band}-{method}. Skipping plot.")
                     continue

                try:
                     fig_src = plotting.plot_source_estimate(stc, view="lateral", time_point=0.1, subjects_dir=subjects_dir)
                     if fig_src:
                          src_filename = f"source_{cond}_{method}_{band}.png"
                          src_path = cond_folder / src_filename
                          fig_src.savefig(str(src_path), dpi=150, facecolor='black') 
                          plt.close(fig_src)
                          rel_path = str(src_path.relative_to(subject_folder))
                          results_for_band.append((cond, band, method, rel_path))
                     else:
                          logger.warning(f"    Plotting failed for {cond}-{band}-{method}, no figure returned.")
                          
                except Exception as plot_e:
                     logger.error(f"    Error plotting source for {cond}-{band}-{method}: {plot_e}", exc_info=True)
                     if 'fig_src' in locals() and isinstance(fig_src, plt.Figure):
                          plt.close(fig_src)

            except Exception as compute_e:
                logger.error(f"    Error computing source localization for {cond}-{band} with {method}: {compute_e}", exc_info=True)

    except Exception as band_e:
         logger.error(f"    Error processing band {band} for {cond} in source localization: {band_e}", exc_info=True)

    return results_for_band

# --- New Orchestration Function --- 
def run_source_localization_analysis(raw_eo, raw_ec, folders, band_list, num_workers=None) -> dict:
    """Orchestrates source localization analysis.

    Sets up necessary MNE components (fsaverage, BEM, forward, inverse)
    and runs source computation for each band in parallel.

    Args:
        raw_eo (mne.io.Raw | None): Eyes open data.
        raw_ec (mne.io.Raw | None): Eyes closed data.
        folders (dict): Dictionary of output folder paths (must include 'subject' and 'plots_src').
        band_list (list[str]): List of frequency band names to analyze.
        num_workers (int, optional): Number of workers for parallel processing. Defaults to cpu count.

    Returns:
        dict: Dictionary containing relative paths to the generated source plot images,
              structured as {condition: {band: {method: rel_path}}}.
    """
    logger.info("--- Starting Source Localization Analysis ---")
    source_loc_start_time = time.time()
    source_localization_results = {"EO": {}, "EC": {}}
    source_methods = {"LORETA": "MNE", "sLORETA": "sLORETA", "eLORETA": "eLORETA"}
    n_workers = num_workers if num_workers else mp.cpu_count()

    if raw_eo is None and raw_ec is None:
        logger.warning("  Skipping source localization: No EO or EC data available.")
        return source_localization_results

    subjects_dir, src, bem_solution = None, None, None
    fwd_eo, fwd_ec = None, None
    inv_op_eo, inv_op_ec = None, None
    cov_common = None
    setup_successful = True

    try:
        logger.info("  Fetching fsaverage data...")
        fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
        subjects_dir = os.path.dirname(fs_dir)
        subject_fs = "fsaverage"

        logger.info("  Setting up source space...")
        src = mne.setup_source_space(subject_fs, spacing="oct6", subjects_dir=subjects_dir, add_dist=False, verbose=False)
        logger.info("  Setting up BEM model...")
        conductivity = (0.3, 0.006, 0.3)
        bem_model = mne.make_bem_model(subject=subject_fs, ico=4, conductivity=conductivity, subjects_dir=subjects_dir, verbose=False)
        bem_solution = mne.make_bem_solution(bem_model, verbose=False)

        if raw_eo:
            logger.info("  Computing covariance from EO data...")
            events_eo = mne.make_fixed_length_events(raw_eo, duration=2.0)
            epochs_eo = mne.Epochs(raw_eo, events_eo, tmin=-0.1, tmax=0.4, baseline=(None, 0), preload=True, verbose=False)
            cov_common = mne.compute_covariance(epochs_eo, tmax=0., method="empirical", verbose=False)

        if raw_eo:
            logger.info("  Preparing EO forward solution...")
            fwd_eo = mne.make_forward_solution(raw_eo.info, trans="fsaverage", src=src, bem=bem_solution, eeg=True, meg=False, verbose=False)
        if raw_ec:
            logger.info("  Preparing EC forward solution...")
            fwd_ec = mne.make_forward_solution(raw_ec.info, trans="fsaverage", src=src, bem=bem_solution, eeg=True, meg=False, verbose=False)

        if raw_eo and fwd_eo and cov_common:
            logger.info("  Preparing EO inverse operator...")
            inv_op_eo = compute_inverse_operator(raw_eo, fwd_eo, cov_common)
        if raw_ec and fwd_ec:
            if cov_common:
                logger.info("  Preparing EC inverse operator (using EO covariance)...")
                inv_op_ec = compute_inverse_operator(raw_ec, fwd_ec, cov_common)
            else:
                logger.warning("  Cannot prepare EC inverse operator: No suitable covariance available.")

    except Exception as e_setup:
        logger.error(f"  ❌ Error during MNE setup for source localization: {e_setup}", exc_info=True)
        setup_successful = False

    if not setup_successful:
         logger.warning("Skipping source localization due to setup errors.")
         return source_localization_results

    tasks = []
    result_keys = []
    logger.info("  Preparing parallel source computation tasks...")
    for cond, raw_data, inv_op in [("EO", raw_eo, inv_op_eo), ("EC", raw_ec, inv_op_ec)]:
        if raw_data is None or inv_op is None:
            logger.info(f"  Skipping source computation for {cond}: Missing data or inverse operator.")
            continue
        for band in band_list:
            args = (cond, raw_data, inv_op, band, folders, source_methods, subjects_dir)
            tasks.append((compute_source_for_band, args, f"source_{cond}_{band}"))

    if not tasks:
         logger.warning("  No source computation tasks defined after setup.")
         return source_localization_results

    logger.info(f"  🚀 Starting {len(tasks)} parallel source computation tasks using {n_workers} workers...")
    parallel_start_time = time.time()
    computed_results_list = []
    try:
        with mp.Pool(processes=n_workers) as pool:
            async_results_src = []
            temp_result_keys_src = []

            for func, args, key in tasks:
                temp_result_keys_src.append(key)
                res = pool.apply_async(execute_task, args=(func, args))
                async_results_src.append(res)
            
            pool.close()
            
            timeout_seconds_src = 600
            for i, res in enumerate(async_results_src):
                key = temp_result_keys_src[i]
                try:
                    result_list = res.get(timeout=timeout_seconds_src)
                    if result_list is not None:
                        computed_results_list.extend(result_list)
                        logger.info(f"    ✅ Task '{key}' completed.")
                    else:
                         logger.warning(f"    ⚠️ Task '{key}' failed (returned None). Check logs.")
                except mp.TimeoutError:
                    logger.error(f"    ❌ Task '{key}' timed out after {timeout_seconds_src}s.")
                except Exception as e:
                    logger.error(f"    ❌ Task '{key}' failed with error: {e}", exc_info=True)
            
            pool.join()
            
    except Exception as pool_e_src:
        logger.error(f"Source localization multiprocessing pool error: {pool_e_src}", exc_info=True)

    parallel_end_time = time.time()
    logger.info(f"  Source computation finished in {parallel_end_time - parallel_start_time:.2f}s")

    for cond, band, method, rel_filename in computed_results_list:
        if cond not in source_localization_results:
             source_localization_results[cond] = {}
        if band not in source_localization_results[cond]:
             source_localization_results[cond][band] = {}
        source_localization_results[cond][band][method] = rel_filename

    source_loc_end_time = time.time()
    logger.info(f"--- Source Localization Analysis finished in {source_loc_end_time - source_loc_start_time:.2f}s ---")

    return source_localization_results
