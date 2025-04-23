import numpy as np
import pandas as pd
from scipy.stats import zscore, pearsonr, median_abs_deviation  # Added median_abs_deviation
import mne
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Try to import psd_welch; if unavailable, fall back to psd_array_welch.
try:
    from mne.time_frequency import psd_welch
except ImportError:
    try:
        from mne.time_frequency import psd_array_welch as psd_welch
    except ImportError:
        logger.error("Could not find psd_welch or psd_array_welch in mne.time_frequency.")
        # Define a dummy function or raise an error if this is critical
        def psd_welch(*args, **kwargs):
            raise NotImplementedError("PSD calculation function not available.")

# ---------------- Robust Z-Score Functions ----------------
def robust_mad(x, constant=1.4826):
    """
    Calculate the Median Absolute Deviation (MAD) using a robust method.

    Args:
        x (np.ndarray): Input data array.
        constant (float): Scaling constant (default: 1.4826 for normal distribution).

    Returns:
        tuple: (mad_scale, median)
               mad_scale: The robust MAD scale estimate.
               median: The median of the input data.
    """
    x = np.asarray(x)
    med = np.median(x)
    # Use scipy's implementation for efficiency and robustness
    mad = median_abs_deviation(x, scale=constant, nan_policy='omit')
    # Handle cases where mad is zero to avoid division by zero
    if mad == 0:
        # Fallback to IQR or a small constant if MAD is zero
        q75, q25 = np.percentile(x[~np.isnan(x)], [75, 25])
        iqr = q75 - q25
        if iqr != 0:
            mad = iqr / 1.349 # Approximate MAD from IQR for normal distribution
        else:
            mad = 1e-6 # Use a small epsilon if IQR is also zero
    return mad, med


def robust_zscore(x, use_iqr=False):
    """
    Calculate robust z-scores using MAD or IQR.

    Args:
        x (np.ndarray): Input data array.
        use_iqr (bool): If True, use Interquartile Range (IQR) instead of MAD for scaling.

    Returns:
        np.ndarray: Robust z-scores.
    """
    x = np.asarray(x)
    med = np.median(x)
    if use_iqr:
        q75, q25 = np.percentile(x, [75, 25])
        scale = q75 - q25
        if scale == 0:
            logger.warning("IQR is zero. Using MAD as fallback.")
            scale, med = robust_mad(x) # Fallback to MAD
    else:
        scale, med = robust_mad(x)

    if scale == 0 or np.isnan(scale):
        logger.warning(f"Robust scale (MAD/IQR) is zero or NaN. Returning original data minus median, or zeros if data is constant.")
        # If scale is zero, return 0 for constant data, or data-median for non-constant
        if np.all(x == med):
             return np.zeros_like(x)
        else:
             # Avoid division by zero but still center the data
             # Or potentially return np.nan or raise an error depending on desired behavior
             return x - med # Return centered data

    return (x - med) / scale


def compute_bandpower_robust_zscores(raw, bands=None, fmin=1, fmax=40, use_iqr=False, method='welch'):
    """
    Compute robust z-scores for EEG band power features.

    Args:
        raw (mne.io.Raw): MNE Raw object containing EEG data.
        bands (dict, optional): Dictionary defining frequency bands. Defaults to standard bands.
        fmin (int, optional): Minimum frequency for PSD calculation. Defaults to 1.
        fmax (int, optional): Maximum frequency for PSD calculation. Defaults to 40.
        use_iqr (bool, optional): Whether to use IQR for robust z-scoring. Defaults to False (uses MAD).
        method (str, optional): Method for PSD calculation ('welch' or 'multitaper'). Defaults to 'welch'.


    Returns:
        dict: Dictionary where keys are band names and values are arrays of robust z-scores for each channel.
    """
    if bands is None:
        bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'SMR': (13, 15), # Sensory Motor Rhythm
            'beta': (16, 28), # Combined Beta 1 and 2
            'gamma': (29, 40) # Adjusted upper limit based on fmax
        }

    sfreq = raw.info['sfreq']
    data = raw.get_data() # Shape (n_channels, n_times)

    if method == 'welch':
         psds, freqs = psd_welch(data, sfreq, fmin=fmin, fmax=fmax, verbose=False)
    elif method == 'multitaper':
        psds, freqs = mne.time_frequency.psd_array_multitaper(data, sfreq, fmin=fmin, fmax=fmax, adaptive=True, normalization='full', verbose=False)
    else:
        raise ValueError("Method must be 'welch' or 'multitaper'")


    # Ensure psds is not empty and freqs cover the desired bands
    if psds.size == 0 or freqs.size == 0:
        logger.error("PSD computation resulted in empty output.")
        return {band: np.full(raw.info['nchan'], np.nan) for band in bands}

    psds_db = 10 * np.log10(psds + 1e-10) # Add epsilon to avoid log10(0)

    robust_features = {}
    for band, (low, high) in bands.items():
        # Ensure band frequencies are within the computed frequency range
        actual_low = max(low, freqs.min())
        actual_high = min(high, freqs.max())
        if actual_low >= actual_high:
             logger.warning(f"Frequency band '{band}' ({low}-{high} Hz) is outside the computed PSD range ({freqs.min()}-{freqs.max()} Hz). Skipping.")
             robust_features[band] = np.full(raw.info['nchan'], np.nan) # Assign NaN or handle as appropriate
             continue

        band_mask = (freqs >= actual_low) & (freqs <= actual_high)

        if not np.any(band_mask):
            logger.warning(f"No frequencies found within the band '{band}' ({actual_low}-{actual_high} Hz). Skipping.")
            robust_features[band] = np.full(raw.info['nchan'], np.nan)
            continue

        # Calculate mean power within the band for each channel
        # psds_db shape: (n_channels, n_freqs)
        band_power = np.mean(psds_db[:, band_mask], axis=1)

        if np.all(np.isnan(band_power)):
             logger.warning(f"All band power values are NaN for band '{band}'.")
             robust_features[band] = np.full(raw.info['nchan'], np.nan)
        else:
             robust_features[band] = robust_zscore(band_power, use_iqr=use_iqr)

    return robust_features


def load_clinical_outcomes(csv_file, n_channels):
    """
    Load clinical outcome data from a CSV file.

    Args:
        csv_file (str or Path): Path to the CSV file. Expected to have an 'outcome' column.
        n_channels (int): Expected number of channels (used for padding if CSV is shorter).

    Returns:
        np.ndarray: Array of clinical outcomes, padded or truncated to n_channels.
                    Returns random data if loading fails.
    """
    try:
        df = pd.read_csv(csv_file)
        if 'outcome' not in df.columns:
            raise ValueError("CSV file must contain an 'outcome' column.")
        outcomes = df['outcome'].values.astype(float) # Ensure numeric type

        # Handle NaNs if necessary, e.g., by imputation or removal
        if np.isnan(outcomes).any():
            logger.warning(f"NaN values found in outcomes from {csv_file}. Consider imputation.")
            # Simple mean imputation for example:
            # mean_outcome = np.nanmean(outcomes)
            # outcomes[np.isnan(outcomes)] = mean_outcome

        # Pad or truncate to match n_channels
        if len(outcomes) < n_channels:
            logger.warning(f"Clinical outcomes ({len(outcomes)}) fewer than channels ({n_channels}). Padding with NaN.")
            outcomes = np.pad(outcomes, (0, n_channels - len(outcomes)), mode='constant', constant_values=np.nan)
        elif len(outcomes) > n_channels:
             logger.warning(f"Clinical outcomes ({len(outcomes)}) more than channels ({n_channels}). Truncating.")
             outcomes = outcomes[:n_channels]
        return outcomes
    except FileNotFoundError:
        logger.error(f"Clinical outcomes CSV file not found: {csv_file}")
        return np.random.rand(n_channels) # Fallback to random data
    except Exception as e:
        logger.error(f"Could not load clinical outcomes from CSV '{csv_file}': {e}")
        return np.random.rand(n_channels) # Fallback


def compare_zscores(z_scores_dict, clinical_outcomes, z_score_type_name="Computed"):
    """
    Compare the correlation of a given set of z-scores with clinical outcomes.

    Args:
        z_scores_dict (dict): Dictionary of z-scores per band (e.g., output of compute_all_zscore_maps).
                              Expected structure: {band: [z_scores_array]}
        clinical_outcomes (np.ndarray): Array of clinical outcome scores.
        z_score_type_name (str): Name for the type of Z-score being compared (e.g., 'robust_mad').
    """
    if z_scores_dict is None or not isinstance(z_scores_dict, dict) or clinical_outcomes is None:
        logger.warning("Skipping Z-score vs Clinical Outcome comparison due to missing input data.")
        return

    print(f"\n--- Z-Score ({z_score_type_name}) Comparison with Clinical Outcomes ---")
    for band, z_band_scores in z_scores_dict.items():
        if z_band_scores is None:
            logger.warning(f"Band '{band}' has no Z-score data. Skipping comparison.")
            continue

        z_band_scores = np.asarray(z_band_scores)

        # Ensure consistent length and handle potential NaNs before correlation
        min_len = min(len(z_band_scores), len(clinical_outcomes))
        z_band_corr = z_band_scores[:min_len]
        outcomes_corr = clinical_outcomes[:min_len]

        # Create masks for valid (non-NaN) data points for correlation
        valid_mask = ~np.isnan(z_band_corr) & ~np.isnan(outcomes_corr)

        if np.sum(valid_mask) < 2: # Need at least 2 points for correlation
             logger.warning(f"Not enough valid data points to compute correlation for band '{band}'.")
             print(f"Band {band}: Correlation skipped (Insufficient valid data)")
             continue

        z_valid = z_band_corr[valid_mask]
        outcomes_valid = outcomes_corr[valid_mask]

        # Check for variance before calculating correlation
        if np.var(z_valid) == 0 or np.var(outcomes_valid) == 0:
            logger.warning(f"Zero variance detected in data for band '{band}'. Correlation cannot be computed.")
            print(f"Band {band}: Correlation skipped (Zero variance)")
            continue

        try:
            r_val, p_val = pearsonr(z_valid, outcomes_valid)
            print(f"Band {band}:")
            print(f"  {z_score_type_name} z-score vs Outcome : r = {r_val:.3f}, p = {p_val:.3f}")
        except ValueError as e:
            logger.error(f"Could not compute Pearson correlation for band '{band}': {e}")
            print(f"Band {band}: Correlation calculation failed.")


def load_zscore_stats(method_choice):
    """
    Load precomputed z-score statistics based on the chosen method.

    Args:
        method_choice (str): The method identifier (e.g., 'standard', 'robust_mad', 'robust_iqr').

    Returns:
        dict or None: Dictionary containing the loaded statistics (e.g., means and std devs),
                      or None if loading fails or method is invalid.
    """
    # Placeholder: Implement actual loading logic here based on method_choice
    # This might involve reading from specific files (CSV, numpy arrays, etc.)
    # associated with each method.
    stats_file_map = {
        'standard': 'path/to/standard_stats.npz',
        'robust_mad': 'path/to/robust_mad_stats.npz',
        'robust_iqr': 'path/to/robust_iqr_stats.npz',
        # Add paths for published norms if applicable
         'published_kolk': 'path/to/kolk_norms.npz', # Example
         'published_smith': 'path/to/smith_norms.npz', # Example
    }

    if method_choice not in stats_file_map:
        logger.error(f"Invalid z-score method choice for loading stats: {method_choice}")
        return None

    stats_file = stats_file_map[method_choice]

    try:
        # Example loading from a .npz file
        # Adjust based on actual file format
        if Path(stats_file).exists():
            data = np.load(stats_file)
            # Assuming the file contains 'means' and 'stds' arrays/dicts
            norm_stats = {'means': data.get('means'), 'stds': data.get('stds')}
            if norm_stats['means'] is None or norm_stats['stds'] is None:
                 logger.error(f"Stats file {stats_file} is missing required 'means' or 'stds' data.")
                 return None
            logger.info(f"Loaded z-score normalization stats from: {stats_file}")
            return norm_stats
        else:
            logger.warning(f"Z-score statistics file not found: {stats_file}. Z-scoring might be relative if no stats provided.")
            return None # Indicate stats are not loaded

    except Exception as e:
        logger.error(f"Failed to load z-score stats from {stats_file}: {e}")
        return None 