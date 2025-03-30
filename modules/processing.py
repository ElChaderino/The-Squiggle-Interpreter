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
    """
    Compute robust z-score maps for each frequency band.
    
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
        stats = norm_stats_dict.get(band, {"median": np.median(band_powers),
                                             "mad": np.median(np.abs(band_powers - np.median(band_powers)))})
        z_scores = [compute_robust_zscore(p, stats["median"], stats["mad"]) for p in band_powers]
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
    stc = mne.minimum_norm.apply_inverse(evoked, inv_op, lambda2=lambda2,
                                         method=method, pick_ori=None, verbose=False)
    return stc
