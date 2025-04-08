import numpy as np
import mne
import matplotlib.pyplot as plt
from pathlib import Path

# Define frequency bands
ALPHA_BAND = (8, 12)
THETA_BAND = (4, 8)

# Clinical vigilance color mapping (for the strip plot)
VIGILANCE_COLORS = {
    'A1': '#000080',  # navy (posterior alpha dominant)
    'A2': '#008080',  # teal (emerging anterior alpha)
    'A3': '#00FFFF',  # cyan (anterior alpha dominant)
    'B1': '#FFBF00',  # amber (alpha drop-out begins)
    'B2': '#FF8000',  # orange (frontal theta appears)
    'B3': '#FF0000',  # red (theta prominent)
    'C': '#800080'  # purple (sleep onset markers)
}


def compute_band_power(epoch, sfreq, band, channel_idx=0, min_samples_for_filter=256, filter_length_factor=0.8):
    """
    Compute the power in a specific frequency band for an epoch, dynamically adjusting filter parameters.

    Args:
        epoch (ndarray): The epoch data (channels x samples)
        sfreq (float): Sampling frequency in Hz
        band (tuple): Frequency band as (low, high) in Hz, e.g., (8, 12) for alpha band
        channel_idx (int): Index of the channel to process (default: 0)
        min_samples_for_filter (int): Minimum samples required to use filtering instead of PSD (default: 256)
        filter_length_factor (float): Factor to determine target filter length (default: 0.8)

    Returns:
        float: Average power in the specified band for the selected channel

    Raises:
        ValueError: If filtering fails due to invalid parameters or signal issues
    """
    # Select the specified channel
    epoch_channel = epoch[channel_idx, :]  # Shape: (n_samples,)

    signal_length = epoch_channel.shape[0]  # Number of samples
    nyq = sfreq / 2.0  # Nyquist frequency

    # Dynamically calculate transition bandwidth
    target_filter_length = int(filter_length_factor * signal_length)
    trans_bandwidth = max(2.0, sfreq / target_filter_length)
    # Ensure transition bandwidth is reasonable for the band
    band_width = band[1] - band[0]
    trans_bandwidth = max(trans_bandwidth, band_width / 2)

    print(
        f"Computing band power for {band}: signal_length={signal_length}, sfreq={sfreq}, "
        f"trans_bandwidth={trans_bandwidth:.2f}, estimated filter_length={int(sfreq / trans_bandwidth)}"
    )

    # Use PSD for short signals
    if signal_length < min_samples_for_filter:
        print(f"Signal too short ({signal_length} samples) for filtering; using PSD-based method.")
        from mne.time_frequency import psd_array_welch
        nperseg = min(256, signal_length // 2) if signal_length >= 2 else signal_length
        psd, freqs = psd_array_welch(
            epoch_channel[None, :], sfreq, fmin=band[0], fmax=band[1], nperseg=nperseg, verbose=False
        )
        power = np.mean(psd)
        return power

    # Use filtering for longer signals
    try:
        filtered = mne.filter.filter_data(
            epoch_channel[None, :],  # Shape: (1, n_samples)
            sfreq,
            band[0],
            band[1],
            l_trans_bandwidth=trans_bandwidth,
            h_trans_bandwidth=trans_bandwidth,
            verbose=False
        )
    except Exception as e:
        raise ValueError(f"Error filtering data for band {band}: {e}")

    # Compute power as mean squared amplitude
    power = np.mean(filtered ** 2)
    return power

def classify_epoch(epoch, sfreq, channel_idx=0):
    """
    Classify the vigilance state for a given epoch using the specified channel.

    Args:
        epoch (ndarray): Epoch data (channels x samples)
        sfreq (float): Sampling frequency
        channel_idx (int): Index of the channel to use (default: 0)

    Returns:
        str: Vigilance stage (e.g., 'A1', 'B2', etc.)
    """
    alpha_power = compute_band_power(epoch, sfreq, ALPHA_BAND, channel_idx=channel_idx)
    theta_power = compute_band_power(epoch, sfreq, THETA_BAND, channel_idx=channel_idx)
    ratio = alpha_power / (theta_power + 1e-6)
    print(f"Classifying epoch: alpha_power={alpha_power:.2f}, theta_power={theta_power:.2f}, ratio={ratio:.2f}")
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


def compute_vigilance_states(raw, epoch_length=1.0, channel_name='OZ'):
    """
    Compute vigilance states for the given raw EEG data using the specified channel.

    Args:
        raw (mne.io.Raw): Raw EEG data
        epoch_length (float): Duration of each epoch in seconds (default: 1.0)
        channel_name (str): Name of the channel to use for vigilance classification (default: 'OZ')

    Returns:
        list of tuples: List of (start_time, stage) for each epoch
    """
    sfreq = raw.info['sfreq']
    n_samples_epoch = int(epoch_length * sfreq)
    print(
        f"Computing vigilance states: sfreq={sfreq} Hz, epoch_length={epoch_length} s, samples_per_epoch={n_samples_epoch}")

    # Case-insensitive channel search
    channel_idx = next((i for i, ch in enumerate(raw.ch_names) if ch.lower() == channel_name.lower()), None)
    if channel_idx is None:
        raise ValueError(f"Channel '{channel_name}' not found in raw data. Available channels: {raw.ch_names}")

    print(f"Using channel {raw.ch_names[channel_idx]} (index {channel_idx}) for vigilance classification")

    # Extract data and compute vigilance states
    data = raw.get_data()
    n_epochs = data.shape[1] // n_samples_epoch
    vigilance_states = []
    for i in range(n_epochs):
        start = i * n_samples_epoch
        end = start + n_samples_epoch
        epoch = data[:, start:end]
        stage = classify_epoch(epoch, sfreq, channel_idx=channel_idx)  # Assumed function for classification
        start_time = start / sfreq
        vigilance_states.append((start_time, stage))
    return vigilance_states


def plot_vigilance_hypnogram(vigilance_states, output_dir, condition, epoch_length=1.0):
    """
    Plot and save a vigilance hypnogram and strip for the given vigilance states.

    Args:
        vigilance_states (list of tuples): List of (start_time, stage) for each epoch
        output_dir (str or Path): Directory to save the hypnogram and strip
        condition (str): Condition identifier (e.g., "EO", "EC")
        epoch_length (float): Duration of each epoch in seconds (default: 1.0)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot and save the hypnogram
    stage_to_level = {
        'A1': 6,
        'A2': 5,
        'A3': 4,
        'B1': 3,
        'B2': 2,
        'B3': 1,
        'C': 0
    }
    times = []
    levels = []
    for (start_time, stage) in vigilance_states:
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
    print(f"Vigilance hypnogram saved to: {hypnogram_path}")

    # Plot and save the vigilance strip
    fig = plot_vigilance_strip(vigilance_states, epoch_length)
    strip_path = output_dir / f"vigilance_strip_{condition}.png"
    fig.savefig(strip_path)
    plt.close(fig)
    print(f"Vigilance strip saved to: {strip_path}")

def plot_vigilance_strip(vigilance_states, epoch_length=1.0):
    """
    Plot a vigilance strip using colored rectangles for each epoch, with a legend.

    Parameters:
      vigilance_states (list of tuples): List of (start_time, stage) for each epoch.
      epoch_length (float): Duration of each epoch in seconds.

    Returns:
      matplotlib.figure.Figure: Figure displaying the vigilance strip.
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 1.5))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    for (start_time, stage) in vigilance_states:
        color = VIGILANCE_COLORS.get(stage, 'gray')
        ax.add_patch(plt.Rectangle((start_time, 0), epoch_length, 1, color=color))

    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=stage) for stage, color in VIGILANCE_COLORS.items()]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=7, fontsize=8, frameon=False, labelcolor='white')

    if vigilance_states:
        total_time = vigilance_states[-1][0] + epoch_length
    else:
        total_time = 0
    ax.set_xlim(0, total_time)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Vigilance Strip", color='white')
    plt.tight_layout()
    return fig
