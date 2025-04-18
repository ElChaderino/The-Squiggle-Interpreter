import numpy as np
import mne
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from .config import ALPHA_BAND, THETA_BAND, VIGILANCE_COLORS, VIGILANCE_CHANNELS

logger = logging.getLogger(__name__)

def select_vigilance_channels(raw_eo, raw_ec, raw_eo_csd, raw_ec_csd):
    """
    Select and validate occipital channels for vigilance processing across EO, EC, and CSD data.
    
    Args:
        raw_eo (mne.io.Raw): Eyes-open raw data.
        raw_ec (mne.io.Raw): Eyes-closed raw data.
        raw_eo_csd (mne.io.Raw): Eyes-open CSD data.
        raw_ec_csd (mne.io.Raw): Eyes-closed CSD data.
    
    Returns:
        list: Valid channel names common to all provided datasets.
    """
    available_channels = None
    for raw in [raw_eo, raw_ec, raw_eo_csd, raw_ec_csd]:
        if raw is not None:
            available_channels = raw.ch_names
            break
    if not available_channels:
        logger.warning("No channels available for vigilance processing.")
        return []

    candidates = VIGILANCE_CHANNELS
    channels_to_process = [ch for ch in candidates if ch.upper() in [c.upper() for c in available_channels]]
    if not channels_to_process:
        logger.warning(f"No occipital channels available. Candidates: {candidates}, Available: {available_channels}")
        return []

    is_uppercase = available_channels[0].isupper()
    channels_to_process = [ch.upper() if is_uppercase else ch.lower() for ch in channels_to_process]

    for raw, name in [(raw_eo, "raw_eo"), (raw_ec, "raw_ec"), (raw_eo_csd, "raw_eo_csd"), (raw_ec_csd, "raw_ec_csd")]:
        if raw is None:
            continue
        valid_channels = []
        for ch_name in channels_to_process:
            if ch_name not in raw.ch_names:
                logger.warning(f"Channel '{ch_name}' not found in {name}. Available channels: {raw.ch_names}. Skipping.")
                continue
            valid_channels.append(ch_name)
        channels_to_process = valid_channels
        if not channels_to_process:
            logger.warning(f"No common channels available after validation for {name}.")
            return []

    logger.info(f"Selected vigilance channels: {channels_to_process}")
    return channels_to_process

def compute_band_power(epoch, sfreq, band, channel_idx=0, min_samples_for_filter=256, filter_length_factor=0.8):
    """
    Compute the power in a specific frequency band for an epoch, dynamically adjusting filter parameters.
    
    Args:
        epoch (ndarray): The epoch data (channels x samples).
        sfreq (float): Sampling frequency in Hz.
        band (tuple): Frequency band as (low, high) in Hz.
        channel_idx (int): Index of the channel to process (default: 0).
        min_samples_for_filter (int): Minimum samples required to use filtering (default: 256).
        filter_length_factor (float): Factor to determine target filter length (default: 0.8).
    
    Returns:
        float: Average power in the specified band for the selected channel.
    
    Raises:
        ValueError: If filtering fails due to invalid parameters or signal issues.
    """
    if epoch is None or not epoch.size or channel_idx >= epoch.shape[0]:
        logger.warning(f"Invalid epoch or channel index {channel_idx} for band {band}.")
        raise ValueError("Invalid epoch data or channel index.")
    if sfreq <= 0:
        logger.warning("Invalid sampling frequency.")
        raise ValueError("Sampling frequency must be positive.")
    if not np.isfinite(epoch[channel_idx]).all():
        logger.warning(f"Non-finite values detected in epoch for band {band}.")
        raise ValueError("Epoch contains non-finite values.")

    epoch_channel = epoch[channel_idx, :]  # Shape: (n_samples,)
    signal_length = epoch_channel.shape[0]
    nyq = sfreq / 2.0

    target_filter_length = int(filter_length_factor * signal_length)
    trans_bandwidth = max(2.0, sfreq / target_filter_length)
    band_width = band[1] - band[0]
    trans_bandwidth = max(trans_bandwidth, band_width / 2)

    logger.debug(
        f"Computing band power for {band}: signal_length={signal_length}, sfreq={sfreq}, "
        f"trans_bandwidth={trans_bandwidth:.2f}, estimated filter_length={int(sfreq / trans_bandwidth)}"
    )

    if signal_length < min_samples_for_filter:
        logger.debug(f"Signal too short ({signal_length} samples) for filtering; using PSD-based method.")
        from mne.time_frequency import psd_array_welch
        nperseg = min(256, signal_length // 2) if signal_length >= 2 else signal_length
        try:
            psd, freqs = psd_array_welch(
                epoch_channel[None, :], sfreq, fmin=band[0], fmax=band[1], nperseg=nperseg, verbose=False
            )
            if not np.isfinite(psd).all():
                logger.warning(f"Non-finite PSD values for band {band}.")
                raise ValueError("Invalid PSD values.")
            power = np.mean(psd)
            return power
        except Exception as e:
            logger.error(f"PSD computation failed for band {band}: {e}")
            raise ValueError(f"PSD computation failed: {e}")

    try:
        filtered = mne.filter.filter_data(
            epoch_channel[None, :], sfreq, band[0], band[1],
            l_trans_bandwidth=trans_bandwidth, h_trans_bandwidth=trans_bandwidth, verbose=False
        )
    except Exception as e:
        logger.error(f"Error filtering data for band {band}: {e}")
        raise ValueError(f"Filtering failed: {e}")

    power = np.mean(filtered ** 2)
    if not np.isfinite(power):
        logger.warning(f"Non-finite power value for band {band}: {power}")
        raise ValueError("Invalid power value.")
    return power

def classify_epoch(epoch, sfreq, channel_idx=0):
    """
    Classify the vigilance state for a given epoch using the specified channel.
    
    Args:
        epoch (ndarray): Epoch data (channels x samples).
        sfreq (float): Sampling frequency.
        channel_idx (int): Index of the channel to use (default: 0).
    
    Returns:
        str: Vigilance stage (e.g., 'A1', 'B2', etc.).
    """
    try:
        alpha_power = compute_band_power(epoch, sfreq, ALPHA_BAND, channel_idx=channel_idx)
        theta_power = compute_band_power(epoch, sfreq, THETA_BAND, channel_idx=channel_idx)
        if alpha_power <= 0 or theta_power <= 0:
            logger.warning(f"Invalid power values: alpha={alpha_power:.2f}, theta={theta_power:.2f}. Defaulting to 'C'.")
            return 'C'
        ratio = alpha_power / (theta_power + 1e-6)
        logger.debug(f"Classifying epoch: alpha_power={alpha_power:.2f}, theta_power={theta_power:.2f}, ratio={ratio:.2f}")
        if ratio > 2.0:
            stage = 'A1'
        elif 1.5 < ratio <= 2.0:
            stage = 'A2'
        elif 1.0 < ratio <= 1.5:
            stage = 'A3'
        elif 0.75 < ratio <= 1.0:
            stage = 'B1'
        elif 0.5 < ratio <= 0.75:
            stage = 'B2'
        elif ratio <= 0.5:
            stage = 'B3'
        else:
            stage = 'C'
        return stage
    except ValueError as e:
        logger.warning(f"Failed to classify epoch: {e}")
        return 'C'

def compute_vigilance_states(raw, epoch_length=1.0, channel_name='OZ'):
    """
    Compute vigilance states for the given raw EEG data using the specified channel.
    
    Args:
        raw (mne.io.Raw): Raw EEG data.
        epoch_length (float): Duration of each epoch in seconds (default: 1.0).
        channel_name (str): Name of the channel to use for vigilance classification (default: 'OZ').
    
    Returns:
        list of tuples: List of (start_time, stage) for each epoch.
    """
    if raw is None or not raw.ch_names:
        logger.warning("Cannot compute vigilance states: No raw data available.")
        return []
    if not np.isfinite(raw.get_data()).all():
        logger.warning(f"Non-finite values detected in raw data for {channel_name}.")
        return []

    sfreq = raw.info['sfreq']
    if sfreq <= 0:
        logger.warning("Invalid sampling frequency.")
        return []

    n_samples_epoch = int(epoch_length * sfreq)
    if n_samples_epoch < 1:
        logger.warning(f"Epoch length too short: {epoch_length}s at {sfreq}Hz.")
        return []
    logger.debug(
        f"Computing vigilance states: sfreq={sfreq} Hz, epoch_length={epoch_length} s, samples_per_epoch={n_samples_epoch}"
    )

    channel_idx = next((i for i, ch in enumerate(raw.ch_names) if ch.lower() == channel_name.lower()), None)
    if channel_idx is None:
        logger.warning(f"Channel '{channel_name}' not found in raw data. Available channels: {raw.ch_names}")
        return []

    logger.info(f"Using channel {raw.ch_names[channel_idx]} (index {channel_idx}) for vigilance classification")

    data = raw.get_data()
    n_epochs = data.shape[1] // n_samples_epoch
    if n_epochs == 0:
        logger.warning(f"No epochs possible for {channel_name}: Data too short ({data.shape[1]} samples).")
        return []
    vigilance_states = []
    for i in range(n_epochs):
        start = i * n_samples_epoch
        end = start + n_samples_epoch
        epoch = data[:, start:end]
        try:
            stage = classify_epoch(epoch, sfreq, channel_idx=channel_idx)
            start_time = start / sfreq
            vigilance_states.append((start_time, stage))
        except ValueError as e:
            logger.warning(f"Skipping epoch {i} for channel {channel_name}: {e}")
            continue
    if not vigilance_states:
        logger.warning(f"No valid vigilance states computed for {channel_name}.")
    else:
        logger.info(f"Computed {len(vigilance_states)} vigilance states for {channel_name}.")
    return vigilance_states

def plot_vigilance_hypnogram(vigilance_states, output_dir, condition, epoch_length=1.0):
    """
    Plot and save a vigilance hypnogram and strip for the given vigilance states.
    
    Args:
        vigilance_states (list of tuples): List of (start_time, stage) for each epoch.
        output_dir (str or Path): Directory to save the hypnogram and strip.
        condition (str): Condition identifier (e.g., "EO", "EC").
        epoch_length (float): Duration of each epoch in seconds (default: 1.0).
    """
    if not vigilance_states:
        logger.warning(f"Cannot plot hypnogram for {condition}: No vigilance states provided.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot and save the hypnogram
    stage_to_level = {'A1': 6, 'A2': 5, 'A3': 4, 'B1': 3, 'B2': 2, 'B3': 1, 'C': 0}
    times = []
    levels = []
    for start_time, stage in vigilance_states:
        level = stage_to_level.get(stage, 0)
        times.append(start_time)
        levels.append(level)
    if times:
        times.append(times[-1] + epoch_length)
        levels.append(levels[-1])
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.step(times, levels, where='post', color='cyan', linewidth=2)
    ax.set_ylim(-0.5, 6.5)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    ax.set_xlabel("Time (s)", color='white')
    ax.set_ylabel("Vigilance Stage", color='white')
    ax.set_yticks([6, 5, 4, 3, 2, 1, 0])
    ax.set_yticklabels(['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C'], color='white')
    ax.set_title(f"Vigilance Hypnogram ({condition})", color='white', fontsize=14)
    plt.tight_layout()
    hypnogram_path = output_dir / f"vigilance_hypnogram_{condition}.png"
    plt.savefig(hypnogram_path)
    plt.close(fig)
    logger.info(f"Vigilance hypnogram saved to: {hypnogram_path}")

    # Plot and save the vigilance strip
    fig = plot_vigilance_strip(vigilance_states, epoch_length)
    if fig:
        strip_path = output_dir / f"vigilance_strip_{condition}.png"
        fig.savefig(strip_path)
        plt.close(fig)
        logger.info(f"Vigilance strip saved to: {strip_path}")

def plot_vigilance_strip(vigilance_states, epoch_length=1.0):
    """
    Plot a vigilance strip using colored rectangles for each epoch, with a legend.
    
    Args:
        vigilance_states (list of tuples): List of (start_time, stage) for each epoch.
        epoch_length (float): Duration of each epoch in seconds.
    
    Returns:
        matplotlib.figure.Figure: Figure displaying the vigilance strip.
    """
    if not vigilance_states:
        logger.warning("Cannot plot vigilance strip: No vigilance states provided.")
        return None

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 1.5))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    for start_time, stage in vigilance_states:
        color = VIGILANCE_COLORS.get(stage, 'gray')
        ax.add_patch(plt.Rectangle((start_time, 0), epoch_length, 1, color=color))

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=stage) for stage, color in VIGILANCE_COLORS.items()]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.1),
              ncol=7, fontsize=8, frameon=False, labelcolor='white')

    total_time = vigilance_states[-1][0] + epoch_length if vigilance_states else 0
    ax.set_xlim(0, total_time)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Vigilance Strip", color='white')
    plt.tight_layout()
    return fig

def process_vigilance(raw, subject_folder, condition, channels_to_process=None):
    """
    Process vigilance states for the given raw EEG data and plot the hypnogram.
    
    Args:
        raw (mne.io.Raw): Raw EEG data.
        subject_folder (str): Directory where hypnogram files should be saved.
        condition (str): Condition identifier (e.g., 'EO', 'EC', 'EO_CSD', 'EC_CSD').
        channels_to_process (list): List of channel names to process.
    """
    if raw is None:
        logger.warning(f"Skipping vigilance processing for {condition}: No data available.")
        return

    channels_to_process = channels_to_process or select_vigilance_channels(raw, None, None, None)
    if not channels_to_process:
        logger.warning(f"Skipping vigilance processing for {condition}: No channels to process.")
        return

    output_dir = Path(subject_folder) / 'vigilance'
    output_dir.mkdir(parents=True, exist_ok=True)

    for channel_name in channels_to_process:
        try:
            vigilance_states = compute_vigilance_states(raw, epoch_length=2.0, channel_name=channel_name)
            if not vigilance_states:
                logger.warning(f"No vigilance states generated for {channel_name} in {condition}.")
                continue
            hypnogram_filename = f"{condition}_{channel_name}"
            plot_vigilance_hypnogram(vigilance_states, output_dir, hypnogram_filename, epoch_length=2.0)
        except Exception as e:
            logger.error(f"Error processing vigilance for channel '{channel_name}' in condition '{condition}': {e}")
            continue
