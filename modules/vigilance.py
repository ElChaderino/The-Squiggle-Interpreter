import numpy as np
import mne
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import time

# Initialize logger
logger = logging.getLogger(__name__)

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


def compute_band_power(epoch: np.ndarray, sfreq: float, band: tuple[float, float], channel_idx: int = 0, min_samples_for_filter: int = 256, filter_length_factor: float = 0.8) -> float:
    """
    Compute the power in a specific frequency band for an epoch, dynamically adjusting filter parameters.

    Args:
        epoch (np.ndarray): The epoch data (channels x samples).
        sfreq (float): Sampling frequency in Hz.
        band (tuple[float, float]): Frequency band as (low, high) in Hz, e.g., (8, 12) for alpha band.
        channel_idx (int): Index of the channel to process (default: 0).
        min_samples_for_filter (int): Minimum number of samples required to use FIR filtering. 
                                    Shorter signals will use PSD (Welch) instead (default: 256).
        filter_length_factor (float): Factor of signal length used to determine the target FIR filter length, 
                                       influencing the transition bandwidth (default: 0.8). Higher values lead to
                                       sharper filters but require longer signals.

    Returns:
        float: Average power in the specified band for the selected channel.

    Raises:
        ValueError: If filtering fails due to invalid parameters or signal issues, 
                    or if the specified channel index is out of bounds.
    """
    # Validate channel index
    if not 0 <= channel_idx < epoch.shape[0]:
        raise ValueError(f"Channel index {channel_idx} out of bounds for epoch with shape {epoch.shape}")

    # Select the specified channel
    epoch_channel = epoch[channel_idx, :]  # Shape: (n_samples,)

    signal_length = epoch_channel.shape[0]  # Number of samples
    nyq = sfreq / 2.0  # Nyquist frequency

    # Handle zero-length signals gracefully
    if signal_length == 0:
        logger.warning(f"Received zero-length signal for band {band}. Returning 0 power.")
        return 0.0

    # Dynamically calculate transition bandwidth
    target_filter_length = int(filter_length_factor * signal_length)
    # Ensure filter length is at least a few samples
    target_filter_length = max(target_filter_length, 3)

    # Calculate transition bandwidth, ensure it's reasonable
    # Avoid division by zero if target_filter_length becomes small
    if target_filter_length > 0:
        trans_bandwidth = max(2.0, sfreq / target_filter_length) 
    else:
        trans_bandwidth = 2.0 # Default transition bandwidth
    
    # Ensure transition bandwidth is reasonable relative to the band width
    band_width = band[1] - band[0]
    if band_width > 0:
        # Transition bandwidth should not be excessively large compared to the band width
        trans_bandwidth = min(trans_bandwidth, max(band_width / 2, 1.0)) # Ensure at least 1 Hz if band_width > 0
    else:
        # If band_width is zero or negative, use a minimal reasonable transition bandwidth
        trans_bandwidth = max(trans_bandwidth, 1.0)

    estimated_filter_length = int(sfreq / trans_bandwidth) if trans_bandwidth > 0 else signal_length

    logger.debug(
        f"Computing band power for {band} on channel {channel_idx}: signal_length={signal_length}, sfreq={sfreq}, "
        f"trans_bandwidth={trans_bandwidth:.2f}, estimated filter_length={estimated_filter_length}"
    )

    # Use PSD for short signals
    if signal_length < min_samples_for_filter:
        logger.debug(f"Signal too short ({signal_length} samples < {min_samples_for_filter}) for filtering; using PSD-based method.")
        try:
            from mne.time_frequency import psd_array_welch
            # Ensure n_per_seg is valid
            nperseg = min(256, signal_length) if signal_length > 0 else 1
            nperseg = max(1, nperseg) # Ensure nperseg is at least 1
            if nperseg > signal_length:
                nperseg = signal_length
            
            psd, freqs = psd_array_welch(
                epoch_channel[None, :], sfreq, fmin=band[0], fmax=band[1], 
                n_per_seg=nperseg, average='mean', verbose=False
            )
            # Integrate power within the band
            if freqs.size > 0:
                freq_res = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
                power = np.sum(psd[0] * freq_res)
            else:
                power = 0.0 # No frequencies in the band
            logger.debug(f"PSD Computed Power: {power:.4f}")
            return power
        except Exception as e:
            logger.error(f"Error computing PSD for band {band} on channel {channel_idx}: {e}", exc_info=True)
            raise ValueError(f"Error computing PSD for band {band}: {e}")

    # Use filtering for longer signals
    try:
        # MNE filter requires float64 data
        epoch_channel_float64 = epoch_channel.astype(np.float64)
        filtered = mne.filter.filter_data(
            epoch_channel_float64[None, :],  # Shape: (1, n_samples)
            sfreq,
            l_freq=band[0],
            h_freq=band[1],
            l_trans_bandwidth=trans_bandwidth,
            h_trans_bandwidth=trans_bandwidth,
            filter_length='auto', # Let MNE determine optimal length
            phase='zero', # Zero-phase filter to avoid phase distortion
            fir_window='hamming', # Common window for FIR filters
            fir_design='firwin', # Standard FIR design method
            verbose=False
        )
    except Exception as e:
        logger.error(f"Error filtering data for band {band} on channel {channel_idx}: {e}", exc_info=True)
        # Provide more context in the raised ValueError
        raise ValueError(f"Error filtering data for band {band} with sfreq={sfreq}, trans_bw={trans_bandwidth:.2f}: {e}")

    # Compute power as mean squared amplitude
    power = np.mean(filtered ** 2)
    logger.debug(f"Filter Computed Power: {power:.4f}")
    return power

def classify_epoch(epoch: np.ndarray, sfreq: float, channel_idx: int = 0) -> str:
    """
    Classify the vigilance state for a given epoch using the specified channel.

    Args:
        epoch (np.ndarray): Epoch data (channels x samples).
        sfreq (float): Sampling frequency.
        channel_idx (int): Index of the channel to use (default: 0).

    Returns:
        str: Vigilance stage (e.g., 'A1', 'B2', 'C').
    """
    try:
        alpha_power = compute_band_power(epoch, sfreq, ALPHA_BAND, channel_idx=channel_idx)
        theta_power = compute_band_power(epoch, sfreq, THETA_BAND, channel_idx=channel_idx)
        # Add small epsilon to avoid division by zero
        ratio = alpha_power / (theta_power + 1e-9)
        logger.debug(f"Classifying epoch (channel {channel_idx}): alpha_power={alpha_power:.4f}, theta_power={theta_power:.4f}, ratio={ratio:.4f}")
    except ValueError as e:
        logger.error(f"Could not compute band power for epoch classification on channel {channel_idx}: {e}")
        return 'Undefined' # Return an undefined state if power calculation fails

    # Determine stage based on ratio thresholds
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
    # Handle potential NaN or Inf ratios resulting from zero theta power after adding epsilon
    elif np.isnan(ratio) or np.isinf(ratio):
        logger.warning(f"Alpha/Theta ratio is NaN or Inf (alpha={alpha_power:.4f}, theta={theta_power:.4f}). Assigning default stage C.")
        stage = 'C' # Or handle as appropriate, e.g., based on alpha power alone
    else:
        # This case should theoretically not be reached if ratio is a valid number
        logger.warning(f"Unexpected alpha/theta ratio {ratio:.4f}. Assigning default stage C.")
        stage = 'C'
    
    logger.debug(f"Epoch classified as stage: {stage}")
    return stage


def compute_vigilance_states(raw: mne.io.Raw, epoch_length: float = 1.0, channel_name: str = 'OZ') -> list[tuple[float, str]]:
    """
    Compute vigilance states for the given raw EEG data using the specified channel.

    Args:
        raw (mne.io.Raw): Raw EEG data.
        epoch_length (float): Duration of each epoch in seconds (default: 1.0).
        channel_name (str): Name of the channel to use for vigilance classification (default: 'OZ').

    Returns:
        list[tuple[float, str]]: List of (start_time, stage) for each epoch.
                                Returns an empty list if processing fails.
    """
    # --- Changed DEBUG to INFO --- 
    logging.info(f"compute_vigilance_states received object of type: {type(raw)}")
    # --- END LOG ---
    
    # --- FIX: Check against BaseRaw which is the recommended parent class --- #
    # --- More robust check: Use isinstance first, then check class name string as fallback ---
    if not (isinstance(raw, mne.io.BaseRaw) or 'mne.io.' in str(type(raw))):
    # --- END FIX --- #
        logger.error(f"Input must be an MNE Raw object (isinstance check failed, type was {type(raw)}).")
        return []

    sfreq = raw.info['sfreq']
    n_samples_epoch = int(epoch_length * sfreq)
    
    if n_samples_epoch <= 0:
        logger.error(f"Invalid epoch length ({epoch_length}s) or sampling frequency ({sfreq}Hz) resulting in {n_samples_epoch} samples per epoch.")
        return []
        
    logger.info(
        f"Computing vigilance states: sfreq={sfreq:.2f} Hz, epoch_length={epoch_length} s, samples_per_epoch={n_samples_epoch}"
    )

    # Case-insensitive channel search
    ch_names_lower = [ch.lower() for ch in raw.ch_names]
    try:
        # Find the index of the channel
        channel_idx = ch_names_lower.index(channel_name.lower())
    except ValueError:
        logger.error(f"Channel '{channel_name}' not found in raw data. Available channels: {raw.ch_names}")
        # Consider alternatives: use first channel, or raise error? Returning empty for now.
        return []

    logger.info(f"Using channel '{raw.ch_names[channel_idx]}' (index {channel_idx}) for vigilance classification")

    # Extract data and compute vigilance states
    try:
        # Load data only once
        data = raw.get_data(picks=[channel_idx]) # Pick only the required channel
    except Exception as e:
        logger.error(f"Failed to get data for channel '{raw.ch_names[channel_idx]}': {e}", exc_info=True)
        return []

    n_total_samples = data.shape[1]
    n_epochs = n_total_samples // n_samples_epoch
    
    if n_epochs == 0:
        logger.warning(f"Data length ({n_total_samples} samples) is shorter than one epoch ({n_samples_epoch} samples). Cannot compute vigilance states.")
        return []

    vigilance_states = []
    logger.info(f"Processing {n_epochs} epochs...")
    for i in range(n_epochs):
        start = i * n_samples_epoch
        end = start + n_samples_epoch
        epoch = data[:, start:end] # Shape (1, n_samples_epoch)
        
        # classify_epoch expects (n_channels, n_times), so epoch shape is already correct
        stage = classify_epoch(epoch, sfreq, channel_idx=0) # Use index 0 as we picked only one channel
        start_time = start / sfreq
        vigilance_states.append((start_time, stage))
        
        # Log progress periodically
        if (i + 1) % 100 == 0 or (i + 1) == n_epochs:
            logger.info(f"Processed {i + 1}/{n_epochs} epochs.")
            
    logger.info(f"Vigilance state computation complete. Found {len(vigilance_states)} states.")
    return vigilance_states


def plot_vigilance_hypnogram(vigilance_states: list[tuple[float, str]], output_dir: str | Path, filename_base: str, epoch_length: float = 1.0) -> None:
    """
    Plot and save a vigilance hypnogram and strip for the given vigilance states.

    Args:
        vigilance_states (list[tuple[float, str]]): List of (start_time, stage) for each epoch.
        output_dir (str | Path): Directory to save the hypnogram and strip.
        filename_base (str): Base filename for saving the hypnogram and strip.
        epoch_length (float): Duration of each epoch in seconds (default: 1.0).
    """
    output_dir = Path(output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create output directory {output_dir}: {e}")
        return # Abort plotting if directory cannot be created

    if not vigilance_states:
        logger.warning(f"No vigilance states provided for condition '{filename_base}'. Skipping plot generation.")
        return

    # Plot and save the hypnogram
    stage_to_level = {
        'A1': 6,
        'A2': 5,
        'A3': 4,
        'B1': 3,
        'B2': 2,
        'B3': 1,
        'C': 0,
        'Undefined': -1 # Handle potential 'Undefined' stage
    }
    times = []
    levels = []
    for (start_time, stage) in vigilance_states:
        level = stage_to_level.get(stage, -1) # Default to -1 if stage is unknown
        times.append(start_time)
        levels.append(level)
        
    if not times: # Check if times list is empty
        logger.warning(f"No valid vigilance states to plot for condition '{filename_base}'.")
        return
        
    # Append last point to make step plot extend to the end
    times.append(times[-1] + epoch_length)
    levels.append(levels[-1])
    
    hypnogram_path = output_dir / f"{filename_base}_hypnogram.png"
    try:
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        ax.step(times, levels, where='post', color='cyan', linewidth=1.5)
        ax.set_ylim(-1.5, 6.5) # Adjust ylim to accommodate 'Undefined' state
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')
        ax.set_xlabel("Time (s)", color='white', fontsize=10)
        ax.set_ylabel("Vigilance Stage", color='white', fontsize=10)
        # Include 'Undefined' in y-ticks if present
        yticks = [6, 5, 4, 3, 2, 1, 0]
        yticklabels = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C']
        if -1 in levels:
            yticks.append(-1)
            yticklabels.append('Undef')
        ax.set_yticks(sorted(yticks, reverse=True))
        ax.set_yticklabels(yticklabels[yticks.index(y)] for y in sorted(yticks, reverse=True))
        
        ax.set_title(f"Vigilance Hypnogram ({filename_base})", color='white', fontsize=12)
        plt.tight_layout(pad=1.5)
        fig.savefig(hypnogram_path, dpi=150, bbox_inches='tight', facecolor='black')
        plt.close(fig)
        logger.info(f"Saved vigilance hypnogram to {hypnogram_path}")
    except Exception as e:
        logger.error(f"Failed to plot or save hypnogram {hypnogram_path}: {e}", exc_info=True)
        if 'fig' in locals() and fig:
             plt.close(fig) # Ensure figure is closed on error

    # Plot and save the vigilance strip
    strip_path = output_dir / f"{filename_base}_strip.png"
    try:
        fig = plot_vigilance_strip(vigilance_states, epoch_length)
        fig.savefig(strip_path, facecolor='black')
        plt.close(fig)
        logger.info(f"Vigilance strip saved to: {strip_path}")
    except Exception as e:
        logger.error(f"Failed to plot or save vigilance strip {strip_path}: {e}", exc_info=True)
        if 'fig' in locals() and fig:
             plt.close(fig) # Ensure figure is closed on error

def plot_vigilance_strip(vigilance_states: list[tuple[float, str]], epoch_length: float = 1.0) -> plt.Figure:
    """
    Plot a vigilance strip using colored rectangles for each epoch, with a legend.

    Args:
        vigilance_states (list[tuple[float, str]]): List of (start_time, stage) for each epoch.
        epoch_length (float): Duration of each epoch in seconds.

    Returns:
        matplotlib.figure.Figure: Figure displaying the vigilance strip.
    """
    if not vigilance_states:
        logger.warning("Cannot plot vigilance strip: No vigilance states provided.")
        # Return an empty figure or handle as appropriate
        fig, ax = plt.subplots(figsize=(10, 1.5))
        return fig 
        
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 1.5)) # Slightly wider figure
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    valid_stages_present = set()
    for (start_time, stage) in vigilance_states:
        color = VIGILANCE_COLORS.get(stage, 'gray') # Use gray for undefined/unknown stages
        ax.add_patch(plt.Rectangle((start_time, 0), epoch_length, 1, color=color))
        if stage in VIGILANCE_COLORS:
            valid_stages_present.add(stage)

    # Add a legend only for stages present in the data
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=VIGILANCE_COLORS[stage], label=stage) 
        for stage in VIGILANCE_COLORS 
        if stage in valid_stages_present
    ]
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(legend_elements), fontsize=8, frameon=False, labelcolor='white')

    # Set limits based on actual data
    total_time = vigilance_states[-1][0] + epoch_length
    ax.set_xlim(0, total_time)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # ax.set_title("Vigilance Strip", color='white', fontsize=10)
    plt.tight_layout(pad=0.5)
    plt.subplots_adjust(bottom=0.2) # Adjust bottom margin for legend
    return fig

# --- New Orchestration Function --- 
def run_vigilance_analysis(raw: mne.io.Raw | None, output_plot_dir: str | Path, condition: str, channels_to_process: list[str], epoch_length: float = 2.0) -> None:
    """Runs the complete vigilance analysis for specified channels and condition.

    Computes vigilance states and generates hypnogram plots for each channel in
    the provided list.

    Args:
        raw (mne.io.Raw | None): Raw EEG data. If None, the function logs a warning and exits.
        output_plot_dir (str | Path): Directory where hypnogram plots will be saved.
        condition (str): Condition identifier (e.g., 'EO', 'EC') used in logging and filenames.
        channels_to_process (list[str]): List of channel names to analyze.
        epoch_length (float): Duration of each epoch in seconds (default: 2.0).
    """
    if raw is None:
        logger.warning(f"Skipping Vigilance Analysis for {condition}: Raw object is None.")
        return

    output_dir = Path(output_plot_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"--- Starting Vigilance Analysis for Condition: {condition} ---")
    start_time = time.time()
    processed_count = 0

    # Check if channels_to_process are valid for the given raw object
    valid_channels = [ch for ch in channels_to_process if ch.upper() in [c.upper() for c in raw.ch_names]]
    if not valid_channels:
        logger.warning(
            f"None of the specified channels {channels_to_process} found in {condition} data with channels {raw.ch_names}. Skipping vigilance."
        )
        return
        
    if len(valid_channels) < len(channels_to_process):
        missing = set(channels_to_process) - set(valid_channels)
        logger.warning(f"Channels {missing} not found in {condition} data. Processing only {valid_channels}.")

    logger.info(f"Processing vigilance for channels: {valid_channels}")

    all_states = {} # Store states per channel if needed later

    for channel_name in valid_channels:
        logger.info(f"  Computing states for channel '{channel_name}'...")
        try:
            # ---- FIX: Pass the raw object and the channel name ----
            vigilance_states = compute_vigilance_states(
                raw=raw, 
                epoch_length=epoch_length, 
                channel_name=channel_name
            )
            # ---- End Fix ----

            if vigilance_states:
                all_states[channel_name] = vigilance_states
                logger.info(f"  Successfully computed {len(vigilance_states)} states for channel '{channel_name}'.")
                
                # Generate and save hypnogram plot for this channel
                try:
                    filename_base = f"{condition}_{channel_name}"
                    plot_vigilance_hypnogram(
                        vigilance_states,
                        output_dir=output_dir,
                        filename_base=filename_base,
                        epoch_length=epoch_length
                    )
                    logger.info(f"    Saved hypnogram plot: {output_dir / (filename_base + '.png')}")
                    processed_count += 1
                except Exception as e_plot:
                    logger.error(f"    Failed to generate/save hypnogram plot for {channel_name}: {e_plot}", exc_info=True)
                    
            else:
                logger.warning(f"  No vigilance states computed for channel '{channel_name}'. Skipping plot.")

        except Exception as e_compute:
            logger.error(f"  Failed to compute vigilance states for channel '{channel_name}': {e_compute}", exc_info=True)

    end_time = time.time()
    logger.info(
        f"--- Vigilance Analysis for Condition: {condition} completed in {end_time - start_time:.2f}s "
        f"({processed_count}/{len(valid_channels)} channels processed) ---"
    )
    # Note: The function currently doesn't return the computed states, only saves plots.
    # Modify return value if states are needed by the caller.
