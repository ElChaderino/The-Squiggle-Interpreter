import numpy as np
import mne
import matplotlib.pyplot as plt

# Define frequency bands
ALPHA_BAND = (8, 12)
THETA_BAND = (4, 8)

# Clinical vigilance color mapping (for the strip plot)
VIGILANCE_COLORS = {
    'A1': '#000080',   # navy (posterior alpha dominant)
    'A2': '#008080',   # teal (emerging anterior alpha)
    'A3': '#00FFFF',   # cyan (anterior alpha dominant)
    'B1': '#FFBF00',   # amber (alpha drop-out begins)
    'B2': '#FF8000',   # orange (frontal theta appears)
    'B3': '#FF0000',   # red (theta prominent)
    'C':  '#800080'    # purple (sleep onset markers)
}

def compute_band_power(epoch, sfreq, band):
    nyq = sfreq / 2.0  # Nyquist frequency
    max_trans_bw = min(0.5, nyq / 10)
    try:
        filtered = mne.filter.filter_data(epoch, sfreq, band[0], band[1],
                                          trans_bandwidth=max_trans_bw, verbose=False)
    except TypeError:
        filtered = mne.filter.filter_data(epoch, sfreq, band[0], band[1], verbose=False)
    power = np.mean(filtered ** 2)
    return power

def classify_epoch(epoch, sfreq):
    alpha_power = compute_band_power(epoch, sfreq, ALPHA_BAND)
    theta_power = compute_band_power(epoch, sfreq, THETA_BAND)
    ratio = alpha_power / (theta_power + 1e-6)
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

def compute_vigilance_states(raw, epoch_length=1.0):
    sfreq = raw.info['sfreq']
    n_samples_epoch = int(epoch_length * sfreq)
    data = raw.get_data()
    n_epochs = data.shape[1] // n_samples_epoch
    vigilance_states = []
    for i in range(n_epochs):
        start = i * n_samples_epoch
        end = start + n_samples_epoch
        epoch = data[:, start:end]
        stage = classify_epoch(epoch, sfreq)
        start_time = start / sfreq
        vigilance_states.append((start_time, stage))
    return vigilance_states

def plot_vigilance_hypnogram(vigilance_states, epoch_length=1.0):
    # (existing function)
    stage_to_level = {
        'A1': 6,
        'A2': 5,
        'A3': 4,
        'B1': 3,
        'B2': 2,
        'B3': 1,
        'C':  0
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
    ax.set_title("Vigilance Hypnogram", color='white', fontsize=14)
    plt.tight_layout()
    return fig

# NEW FUNCTION: Plot a vigilance strip
def plot_vigilance_strip(vigilance_states, epoch_length=1.0):
    """
    Plot a vigilance strip using colored rectangles for each epoch.
    
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
