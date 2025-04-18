# modules/processing.py
import numpy as np
import mne
from scipy.signal import coherence, welch, hilbert
import nolds
from antropy import sample_entropy, spectral_entropy, lziv_complexity
from scipy.stats import zscore, pearsonr
import matplotlib.pyplot as plt
import logging
import os
from pathlib import Path
from mne.time_frequency import psd_array_welch, tfr_morlet
from .config import BANDS, load_zscore_stats, PLOT_CONFIG, OUTPUT_FOLDERS

logger = logging.getLogger(__name__)

def compute_band_power(data, sfreq, band):
    """
    Filter the signal to a specified frequency band and compute mean power.
    
    Args:
        data (np.array): 1D signal array.
        sfreq (float): Sampling frequency in Hz.
        band (tuple): Frequency band (fmin, fmax).
    
    Returns:
        float: Mean power in the band.
    """
    if data is None or not data.size:
        logger.warning(f"Invalid data for band {band}.")
        return 0.0
    try:
        fmin, fmax = band
        data_filt = mne.filter.filter_data(data, sfreq, fmin, fmax, verbose=False)
        power = np.mean(data_filt ** 2)
        logger.debug(f"Computed band power for {band}: {power:.2f}")
        return power
    except Exception as e:
        logger.warning(f"Failed to compute band power for {band}: {e}")
        return 0.0

def compute_all_band_powers(raw):
    """
    Compute mean power for each channel and for all defined frequency bands.
    
    Args:
        raw (mne.io.Raw): MNE Raw object.
    
    Returns:
        dict: {channel: {band: power, ...}, ...}
    """
    if raw is None:
        logger.warning("Cannot compute band powers: No data available.")
        return {}
    sfreq = raw.info["sfreq"]
    data = raw.get_data() * 1e6  # Convert to microvolts
    results = {}
    for i, ch in enumerate(raw.ch_names):
        results[ch] = {}
        for band_name, band_range in BANDS.items():
            results[ch][band_name] = compute_band_power(data[i], sfreq, band_range)
    logger.info(f"Computed band powers for {raw.info['description']}: {list(results.keys())}")
    return results

def compute_alpha_peak_frequency(data, sfreq, freq_range):
    """
    Compute the peak frequency in the alpha band.
    
    Args:
        data (np.ndarray): 1D signal array.
        sfreq (float): Sampling frequency.
        freq_range (tuple): Frequency range (fmin, fmax).
    
    Returns:
        float: Peak frequency.
    """
    if data is None or not data.size:
        logger.warning("Cannot compute alpha peak frequency: Invalid data.")
        return np.nan
    fmin, fmax = freq_range
    freqs, psd = welch(data, fs=sfreq, nperseg=int(sfreq * 2), noverlap=int(sfreq))
    alpha_mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(alpha_mask):
        logger.warning(f"No frequencies in alpha range {freq_range}.")
        return np.nan
    alpha_freqs = freqs[alpha_mask]
    alpha_psd = psd[alpha_mask]
    peak_freq = float(alpha_freqs[np.argmax(alpha_psd)])
    logger.debug(f"Computed alpha peak frequency: {peak_freq:.2f} Hz")
    return peak_freq

def compute_frontal_asymmetry(bp_EO, ch_left="F3", ch_right="F4"):
    """
    Compute alpha asymmetry between left and right frontal channels.
    
    Args:
        bp_EO (dict): Band powers for EO condition.
        ch_left, ch_right (str): Left and right channel names.
    
    Returns:
        float: Frontal asymmetry value.
    """
    try:
        alpha_left = bp_EO[ch_left]["Alpha"]
        alpha_right = bp_EO[ch_right]["Alpha"]
        if alpha_left == 0 or alpha_right == 0:
            logger.warning(f"Zero alpha power for {ch_left} or {ch_right}.")
            return np.nan
        asymmetry = float(np.log(alpha_right / alpha_left))
        logger.debug(f"Computed frontal asymmetry: {asymmetry:.2f}")
        return asymmetry
    except KeyError as e:
        logger.warning(f"Cannot compute frontal asymmetry: {e}")
        return np.nan

def compute_instability_index(raw, bands):
    """
    Compute variance-based instability index for each band.
    
    Args:
        raw (mne.io.Raw): Raw EEG data.
        bands (dict): Dictionary of band names and frequency ranges.
    
    Returns:
        dict: {band: {channel: variance, ...}, ...}
    """
    if raw is None:
        logger.warning("Cannot compute instability index: No data available.")
        return {}
    sfreq = raw.info["sfreq"]
    data = raw.get_data() * 1e6
    instability = {}
    for band_name, (fmin, fmax) in bands.items():
        data_filt = mne.filter.filter_data(data, sfreq, fmin, fmax, verbose=False)
        variance = np.var(data_filt, axis=1)
        instability[band_name] = {ch: float(var) for ch, var in zip(raw.ch_names, variance)}
        logger.debug(f"Instability Index (Variance) for {band_name}: {instability[band_name]}")
    return instability

def compute_coherence(raw, ch1, ch2, band, sfreq, log_freqs=False):
    """
    Compute coherence between two channels for a specified band.
    
    Args:
        raw (mne.io.Raw): Raw EEG data.
        ch1, ch2 (str): Channel names.
        band (tuple): Frequency band (fmin, fmax).
        sfreq (float): Sampling frequency.
        log_freqs (bool): Whether to log frequency-specific coherence values.
    
    Returns:
        float: Average coherence value.
    """
    if raw is None or ch1 not in raw.ch_names or ch2 not in raw.ch_names:
        logger.warning(f"Cannot compute coherence for {ch1}-{ch2}: Invalid data or channels.")
        return np.nan
    fmin, fmax = band
    data = raw.get_data(picks=[ch1, ch2]) * 1e6
    duration = 2.0
    events = mne.make_fixed_length_events(raw, duration=duration)
    epochs = mne.Epochs(raw, events, tmin=0, tmax=duration, picks=[ch1, ch2], baseline=None, preload=True, verbose=False)
    epochs_data = epochs.get_data() * 1e6
    csd_obj = mne.time_frequency.csd_array_fourier(
        epochs_data, sfreq=sfreq, fmin=fmin, fmax=fmax, n_fft=int(sfreq * 2)
    )
    freqs = csd_obj.frequencies
    coherence_values = []
    if log_freqs:
        logger.debug(f"Coherence Values for {ch1}-{ch2} in {band[0]}-{band[1]} Hz:")
    for f in freqs:
        csd = csd_obj.get_data(frequency=f)
        psd1 = np.abs(csd[0, 0])
        psd2 = np.abs(csd[1, 1])
        csd12 = np.abs(csd[0, 1])
        coherence = csd12 ** 2 / (psd1 * psd2 + 1e-10)
        coherence_values.append(coherence)
        if log_freqs:
            logger.debug(f"Frequency {f:.1f} Hz: Coherence = {coherence:.3f}")
    if not coherence_values:
        logger.warning(f"No coherence values computed for {ch1}-{ch2} in {band}.")
        return np.nan
    coherence_avg = np.mean(coherence_values)
    logger.debug(f"Average Coherence for {ch1}-{ch2}: {coherence_avg:.3f}")
    return float(coherence_avg)

def compute_percentage_change(power_EO, power_EC):
    """
    Compute percentage change from EO to EC.
    
    Args:
        power_EO (float): Power in EO condition.
        power_EC (float): Power in EC condition.
    
    Returns:
        float: Percentage change.
    """
    if power_EO == 0:
        logger.warning("EO power is zero; cannot compute percentage change.")
        return np.nan
    change = ((power_EC - power_EO) / power_EO) * 100
    logger.debug(f"Computed percentage change: {change:.2f}%")
    return change

def compute_theta_beta_ratio(raw, ch):
    """
    Compute the Theta/Beta power ratio for a specific channel.
    
    Args:
        raw (mne.io.Raw): MNE Raw object.
        ch (str): Channel name.
    
    Returns:
        float: Theta/Beta ratio.
    """
    if raw is None or ch not in raw.ch_names:
        logger.warning(f"Cannot compute theta/beta ratio for channel {ch}: Invalid data or channel.")
        return np.nan
    sfreq = raw.info["sfreq"]
    data = raw.get_data(picks=[ch])[0] * 1e6
    theta = compute_band_power(data, sfreq, BANDS["Theta"])
    beta = compute_band_power(data, sfreq, BANDS["Beta"])
    if beta == 0:
        logger.warning(f"Beta power is zero for channel {ch}; cannot compute ratio.")
        return np.nan
    ratio = theta / beta
    logger.debug(f"Computed Theta/Beta ratio for {ch}: {ratio:.2f}")
    return ratio

def higuchi_fd(X, kmax):
    """
    Compute the Higuchi Fractal Dimension of a signal.
    
    Args:
        X (np.array): 1D signal.
        kmax (int): Maximum interval.
    
    Returns:
        float: Estimated fractal dimension.
    """
    if X is None or not X.size:
        logger.warning("Invalid signal for Higuchi FD computation.")
        return np.nan
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
    try:
        slope = np.polyfit(x_vals, L, 1)[0]
        logger.debug(f"Computed Higuchi FD: {slope:.2f}")
        return slope
    except Exception as e:
        logger.warning(f"Failed to compute Higuchi FD: {e}")
        return np.nan

def compute_dfa_on_epochs(signal, sfreq, epoch_len_sec=2.0):
    """
    Compute Detrended Fluctuation Analysis (DFA) on epochs of the signal.
    
    Args:
        signal (np.array): 1D signal.
        sfreq (float): Sampling frequency.
        epoch_len_sec (float): Epoch duration in seconds.
    
    Returns:
        float or None: Average DFA value across epochs.
    """
    if signal is None or not signal.size:
        logger.warning("Invalid signal for DFA computation.")
        return None
    epoch_len = int(epoch_len_sec * sfreq)
    n_epochs = len(signal) // epoch_len
    if n_epochs == 0:
        logger.warning("No epochs available for DFA computation.")
        return None
    dfa_vals = []
    for i in range(n_epochs):
        epoch = signal[i * epoch_len: (i + 1) * epoch_len]
        try:
            dfa_vals.append(nolds.dfa(epoch))
        except Exception as e:
            logger.warning(f"Failed to compute DFA for epoch {i}: {e}")
            continue
    dfa_avg = np.mean(dfa_vals) if dfa_vals else None
    if dfa_avg is not None:
        logger.debug(f"Computed average DFA: {dfa_avg:.2f}")
    return dfa_avg

def compute_entropy_on_epochs(signal, sfreq, epoch_len_sec=2.0):
    """
    Compute epoch-averaged entropy metrics: Sample Entropy, Spectral Entropy, Lempel-Ziv Complexity.
    
    Args:
        signal (np.array): 1D signal.
        sfreq (float): Sampling frequency.
        epoch_len_sec (float): Epoch duration in seconds.
    
    Returns:
        tuple: (average sample entropy, average spectral entropy, average LZ complexity)
    """
    if signal is None or not signal.size:
        logger.warning("Invalid signal for entropy computation.")
        return None, None, None
    epoch_len = int(epoch_len_sec * sfreq)
    n_epochs = len(signal) // epoch_len
    if n_epochs == 0:
        logger.warning("No epochs available for entropy computation.")
        return None, None, None
    sampents, specents, lzivs = [], [], []
    for i in range(n_epochs):
        epoch = signal[i * epoch_len: (i + 1) * epoch_len]
        try:
            sampents.append(sample_entropy(epoch))
            specents.append(spectral_entropy(epoch, sfreq, method="welch", normalize=True))
            lzivs.append(lziv_complexity(epoch))
        except Exception as e:
            logger.warning(f"Failed to compute entropy for epoch {i}: {e}")
            continue
    sampent_avg = np.mean(sampents) if sampents else None
    specent_avg = np.mean(specents) if specents else None
    lziv_avg = np.mean(lzivs) if lzivs else None
    if any(avg is not None for avg in [sampent_avg, specent_avg, lziv_avg]):
        logger.debug(f"Computed entropy: Sample={sampent_avg:.2f}, Spectral={specent_avg:.2f}, LZ={lziv_avg:.2f}")
    return sampent_avg, specent_avg, lziv_avg

def compute_robust_zscore(power, norm_median, norm_mad):
    """
    Compute robust z-score using median and median absolute deviation (MAD).
    
    Args:
        power (float): Observed power.
        norm_median (float): Normative median.
        norm_mad (float): Normative MAD.
    
    Returns:
        float: Robust z-score.
    """
    if norm_mad == 0:
        logger.warning("MAD is zero; using default scale of 1.")
        norm_mad = 1
    zscore_val = (power - norm_median) / norm_mad
    logger.debug(f"Computed robust z-score: {zscore_val:.2f}")
    return zscore_val

def compute_all_zscore_maps(raw, norm_stats_dict, epoch_len_sec=2.0):
    """
    Compute robust z-score maps for each frequency band.
    
    Args:
        raw (mne.io.Raw): Raw EEG data.
        norm_stats_dict (dict): Normative statistics for each band.
        epoch_len_sec (float): Epoch duration in seconds.
    
    Returns:
        dict: {band: list of z-scores for each channel}
    """
    if raw is None:
        logger.warning("Cannot compute z-score maps: No data available.")
        return {}
    sfreq = raw.info["sfreq"]
    data = raw.get_data() * 1e6
    zscore_maps = {}
    for band, band_range in BANDS.items():
        band_powers = [compute_band_power(data[i], sfreq, band_range) for i in range(data.shape[0])]
        stats = norm_stats_dict.get(band, {
            "median": np.median(band_powers),
            "mad": np.median(np.abs(band_powers - np.median(band_powers)))
        })
        z_scores = [compute_robust_zscore(p, stats["median"], stats["mad"]) for p in band_powers]
        zscore_maps[band] = z_scores
    logger.info("Computed z-score maps for all bands.")
    return zscore_maps

def robust_mad(x, constant=1.4826, max_iter=10, tol=1e-3):
    """
    Compute median absolute deviation (MAD) iteratively.
    
    Args:
        x (np.array): Input data.
        constant (float): Scaling constant.
        max_iter (int): Maximum iterations.
        tol (float): Convergence tolerance.
    
    Returns:
        tuple: (MAD, median)
    """
    x = np.asarray(x)
    if not x.size:
        logger.warning("Empty input for MAD computation.")
        return 0.0, 0.0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    for _ in range(max_iter):
        mask = np.abs(x - med) <= 3 * mad
        new_med = np.median(x[mask])
        new_mad = np.median(np.abs(x[mask] - new_med))
        if np.abs(new_med - med) < tol and np.abs(new_mad - mad) < tol:
            break
        med, mad = new_med, new_mad
    mad_val = mad * constant
    logger.debug(f"Computed MAD: {mad_val:.2f}, Median: {med:.2f}")
    return mad_val, med

def robust_zscore(x, use_iqr=False):
    """
    Compute robust z-scores using MAD or IQR.
    
    Args:
        x (np.array): Input data.
        use_iqr (bool): Use IQR instead of MAD.
    
    Returns:
        np.array: Z-scores.
    """
    x = np.asarray(x)
    if not x.size:
        logger.warning("Empty input for robust z-score computation.")
        return np.array([])
    med = np.median(x)
    if use_iqr:
        q75, q25 = np.percentile(x, [75, 25])
        iqr = q75 - q25
        scale = iqr if iqr != 0 else 1.0
    else:
        scale, med = robust_mad(x)
        if scale == 0:
            scale = 1.0
    zscores = (x - med) / scale
    logger.debug(f"Computed robust z-scores: mean={np.mean(zscores):.2f}, std={np.std(zscores):.2f}")
    return zscores

def compute_bandpower_robust_zscores(raw, bands=None, fmin=1, fmax=40, n_fft=2048, use_iqr=False):
    """
    Compute robust z-scores for band powers.
    
    Args:
        raw (mne.io.Raw): Raw EEG data.
        bands (dict, optional): Frequency bands.
        fmin (float): Minimum frequency.
        fmax (float): Maximum frequency.
        n_fft (int): FFT length.
        use_iqr (bool): Use IQR instead of MAD.
    
    Returns:
        dict: {band: z-scores}
    """
    if raw is None:
        logger.warning("Cannot compute robust z-scores: No data available.")
        return {}
    bands = bands or BANDS
    psds, freqs = psd_array_welch(raw.get_data(), raw.info['sfreq'], fmin=fmin, fmax=fmax, n_fft=n_fft, verbose=False)
    psds_db = 10 * np.log10(psds)
    robust_features = {}
    for band, (low, high) in bands.items():
        band_mask = (freqs >= low) & (freqs <= high)
        if not np.any(band_mask):
            logger.warning(f"No frequencies in band {band}.")
            robust_features[band] = np.zeros(psds_db.shape[0])
            continue
        band_power = psds_db[:, band_mask].mean(axis=1)
        robust_features[band] = robust_zscore(band_power, use_iqr=use_iqr)
    logger.info("Computed robust z-scores for all bands.")
    return robust_features

def compare_zscores(standard_z, robust_z, clinical_outcomes):
    """
    Compare standard and robust z-scores against clinical outcomes.
    
    Args:
        standard_z (dict): Standard z-scores by band.
        robust_z (dict): Robust z-scores by band.
        clinical_outcomes (np.array): Clinical outcome values.
    """
    for band in standard_z.keys():
        try:
            r_std, p_std = pearsonr(standard_z[band], clinical_outcomes)
            r_rob, p_rob = pearsonr(robust_z[band], clinical_outcomes)
            logger.info(f"Band {band}:")
            logger.info(f"  Standard z-score: r = {r_std:.3f}, p = {p_std:.3f}")
            logger.info(f"  Robust z-score: r = {r_rob:.3f}, p = {p_rob:.3f}")
        except Exception as e:
            logger.warning(f"Failed to compare z-scores for band {band}: {e}")

def compute_modulation_index(phase_data, amp_data, n_bins=18):
    """
    Compute modulation index (MI) using KL divergence.
    
    Args:
        phase_data (np.array): Phase signal.
        amp_data (np.array): Amplitude envelope signal.
        n_bins (int): Number of bins for phase.
    
    Returns:
        float: Modulation index.
    """
    if phase_data is None or amp_data is None or not phase_data.size or not amp_data.size:
        logger.warning("Invalid data for modulation index computation.")
        return np.nan
    try:
        phase = np.angle(hilbert(phase_data))
        amp_env = np.abs(hilbert(amp_data))
        bins = np.linspace(-np.pi, np.pi, n_bins + 1)
        digitized = np.digitize(phase, bins) - 1
        bin_means = np.array([amp_env[digitized == i].mean() for i in range(n_bins)])
        bin_prob = bin_means / (bin_means.sum() + 1e-10)
        uniform_prob = 1.0 / n_bins
        kl_div = np.sum(bin_prob * np.log((bin_prob + 1e-10) / uniform_prob))
        max_kl = np.log(n_bins)
        mi = kl_div / max_kl
        logger.debug(f"Computed modulation index: {mi:.2f}")
        return mi
    except Exception as e:
        logger.warning(f"Failed to compute modulation index: {e}")
        return np.nan

def run_pac_analysis(data_arr, channels, sfreq, low_freqs, high_freqs, pac_scale_factor, output_dir, condition_name):
    """
    Compute phase-amplitude coupling (PAC) for each channel.
    
    Args:
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
    if data_arr is None or not data_arr.size or not channels:
        logger.warning("Invalid data or channels for PAC analysis.")
        return np.zeros((len(low_freqs), len(high_freqs), len(channels)))
    n_lf = len(low_freqs)
    n_hf = len(high_freqs)
    n_ch = len(channels)
    pac_matrix = np.zeros((n_lf, n_hf, n_ch))
    output_dir = Path(output_dir) / OUTPUT_FOLDERS["coherence_eo"] if condition_name == "EO" else Path(output_dir) / OUTPUT_FOLDERS["coherence_ec"]
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, (lf1, lf2) in enumerate(low_freqs):
        for j, (hf1, hf2) in enumerate(high_freqs):
            for k in range(n_ch):
                try:
                    phase_sig = mne.filter.filter_data(data_arr[k], sfreq, lf1, lf2, verbose=False)
                    amp_sig = mne.filter.filter_data(data_arr[k], sfreq, hf1, hf2, verbose=False)
                    pac_matrix[i, j, k] = compute_modulation_index(phase_sig, amp_sig)
                except Exception as e:
                    logger.warning(f"Failed to compute PAC for channel {channels[k]}, low {lf1}-{lf2}, high {hf1}-{hf2}: {e}")
    pac_matrix *= pac_scale_factor
    for k, ch in enumerate(channels):
        try:
            vals = pac_matrix[:, :, k]
            fig, ax = plt.subplots(figsize=PLOT_CONFIG["coherence_figsize"], facecolor='black')
            ax.set_facecolor('black')
            im = ax.imshow(vals, cmap=PLOT_CONFIG["topomap_cmap"], aspect='auto', interpolation='nearest')
            ax.set_title(f"PAC Heatmap - {ch} ({condition_name})", color='white', fontsize=10)
            ax.set_xticks(range(len(high_freqs)))
            ax.set_xticklabels([f"{hf[0]}-{hf[1]}Hz" for hf in high_freqs], color='white', rotation=45)
            ax.set_yticks(range(len(low_freqs)))
            ax.set_yticklabels([f"{lf[0]}-{lf[1]}Hz" for lf in low_freqs], color='white')
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("PAC", color='white')
            cbar.ax.tick_params(colors='white')
            fig.tight_layout()
            pac_path = output_dir / f"PAC_{ch}_{condition_name}.png"
            fig.savefig(pac_path, dpi=PLOT_CONFIG["dpi"], facecolor='black')
            plt.close(fig)
            logger.info(f"Saved PAC heatmap for {ch} ({condition_name}) to {pac_path}")
        except Exception as e:
            logger.warning(f"Failed to save PAC heatmap for {ch} ({condition_name}): {e}")
    return pac_matrix

def compute_zscore_features(raw_eo, method_choice, norm_stats):
    """
    Compute z-score features using standard or robust methods.
    
    Args:
        raw_eo (mne.io.Raw): Eyes-open raw data.
        method_choice (str): Z-score method ('1', '2', '3', '4').
        norm_stats (dict): Normative statistics.
    
    Returns:
        tuple: (standard_features, chosen_features)
    """
    if raw_eo is None:
        logger.warning("Cannot compute z-score features: EO data is None.")
        return {}, {}
    standard_features = {}
    psds, freqs = psd_array_welch(raw_eo.get_data(), raw_eo.info['sfreq'], fmin=1, fmax=40, n_fft=2048, verbose=False)
    psds_db = 10 * np.log10(psds)
    for band, (low, high) in BANDS.items():
        band_mask = (freqs >= low) & (freqs <= high)
        if not np.any(band_mask):
            logger.warning(f"No frequencies in band {band}.")
            standard_features[band] = np.zeros(psds_db.shape[0])
            continue
        band_power = psds_db[:, band_mask].mean(axis=1)
        standard_features[band] = zscore(band_power)
    if method_choice == "1":
        chosen_features = standard_features
        logger.info("Using standard z-scores (mean/std).")
    elif method_choice == "2":
        chosen_features = compute_bandpower_robust_zscores(raw_eo, bands=BANDS, use_iqr=False)
        logger.info("Using robust z-scores (MAD-based).")
    elif method_choice == "3":
        chosen_features = compute_bandpower_robust_zscores(raw_eo, bands=BANDS, use_iqr=True)
        logger.info("Using robust z-scores (IQR-based).")
    elif method_choice == "4":
        chosen_features = compute_all_zscore_maps(raw_eo, norm_stats, epoch_len_sec=2.0)
        logger.info("Using published norms for z-score normalization.")
    else:
        logger.warning("Invalid choice. Defaulting to standard z-scores.")
        chosen_features = standard_features
    logger.info("Computed z-score features.")
    return standard_features, chosen_features

def compute_coherence_matrix(data, sfreq, band, nperseg):
    """
    Compute a coherence matrix for all channels in a specified frequency band.
    
    Args:
        data (np.array): 2D array (n_channels x n_samples).
        sfreq (float): Sampling frequency.
        band (tuple): Frequency band.
        nperseg (int): nperseg parameter for coherence.
    
    Returns:
        np.array: Coherence matrix.
    """
    if data is None or not data.size:
        logger.warning("Invalid data for coherence matrix computation.")
        return np.zeros((0, 0))
    n_channels = data.shape[0]
    coh_matrix = np.zeros((n_channels, n_channels))
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            try:
                f, Cxy = coherence(data[i], data[j], sfreq, nperseg=nperseg)
                mask = (f >= band[0]) & (f <= band[1])
                if not np.any(mask):
                    logger.warning(f"No frequencies in band {band} for coherence.")
                    continue
                coh_val = np.mean(Cxy[mask])
                coh_matrix[i, j] = coh_val
                coh_matrix[j, i] = coh_val
            except Exception as e:
                logger.warning(f"Failed to compute coherence between channels {i} and {j}: {e}")
    logger.debug("Computed coherence matrix.")
    return coh_matrix

def compute_pseudo_erp(raw):
    """
    Compute a pseudo-ERP by epoching the raw data using fixed-length events and averaging.
    
    Args:
        raw (mne.io.Raw): Raw EEG data.
    
    Returns:
        matplotlib.figure.Figure: Figure displaying the evoked response with dark styling.
    """
    if raw is None:
        logger.warning("Cannot compute pseudo-ERP: No data available.")
        return None
    try:
        events = mne.make_fixed_length_events(raw, duration=2.0)
        epochs = mne.Epochs(raw, events, tmin=-0.1, tmax=0.4, baseline=(None, 0), preload=True, verbose=False)
        evoked = epochs.average()
        fig = evoked.plot(spatial_colors=True, show=False)
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
        logger.info("Computed pseudo-ERP.")
        return fig
    except Exception as e:
        logger.warning(f"Failed to compute pseudo-ERP: {e}")
        return None

def compute_tfr(raw, freqs, n_cycles, tmin=0.0, tmax=2.0):
    """
    Compute Time-Frequency Representation (TFR) using Morlet wavelets.
    
    Args:
        raw (mne.io.Raw): Raw EEG data.
        freqs (array-like): Frequencies of interest.
        n_cycles (array-like or float): Number of cycles.
        tmin (float): Start time for epochs.
        tmax (float): End time for epochs.
    
    Returns:
        mne.time_frequency.AverageTFR: The computed TFR.
    """
    if raw is None:
        logger.warning("Cannot compute TFR: No data available.")
        return None
    try:
        events = mne.make_fixed_length_events(raw, duration=tmax - tmin)
        epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose=False)
        tfr = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, return_itc=False, average=True, verbose=False)
        logger.info("Computed TFR.")
        return tfr
    except Exception as e:
        logger.warning(f"Failed to compute TFR: {e}")
        return None

def compute_ica(raw):
    """
    Compute Independent Component Analysis (ICA) on raw EEG data.
    
    Args:
        raw (mne.io.Raw): Raw EEG data.
    
    Returns:
        mne.preprocessing.ICA: Fitted ICA object.
    """
    if raw is None:
        logger.warning("Cannot compute ICA: No data available.")
        return None
    try:
        ica = mne.preprocessing.ICA(n_components=15, random_state=97, max_iter=800, verbose=False)
        ica.fit(raw)
        logger.info("Computed ICA.")
        return ica
    except Exception as e:
        logger.warning(f"Failed to compute ICA: {e}")
        return None
