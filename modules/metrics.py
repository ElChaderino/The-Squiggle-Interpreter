import numpy as np
from scipy import signal
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
from scipy.stats import zscore
import mne
from mne.preprocessing import ICA

class MicrostateAnalyzer:
    def __init__(self, n_states: int = 4):
        """
        Initialize microstate analyzer.
        
        Args:
            n_states: Number of microstates to identify (default=4)
        """
        self.n_states = n_states
        self.kmeans = KMeans(n_clusters=n_states, random_state=42)
        self.maps = None
        self.labels = None
        
    def fit(self, eeg_data: np.ndarray) -> None:
        """
        Fit microstate model to EEG data.
        
        Args:
            eeg_data: EEG data array of shape (channels, samples)
        """
        # Normalize data
        normalized_data = zscore(eeg_data, axis=1)
        
        # Reshape for clustering
        samples = normalized_data.T
        
        # Fit K-means
        self.labels = self.kmeans.fit_predict(samples)
        self.maps = self.kmeans.cluster_centers_
        
    def get_metrics(self) -> Dict[str, float]:
        """
        Calculate microstate metrics.
        
        Returns:
            Dictionary of microstate metrics
        """
        if self.labels is None:
            raise ValueError("Must fit model before calculating metrics")
            
        metrics = {}
        
        # Calculate duration statistics
        durations = []
        current_state = self.labels[0]
        current_duration = 1
        
        for label in self.labels[1:]:
            if label == current_state:
                current_duration += 1
            else:
                durations.append(current_duration)
                current_state = label
                current_duration = 1
        durations.append(current_duration)
        
        metrics['mean_duration'] = np.mean(durations)
        metrics['duration_variance'] = np.var(durations)
        
        # Calculate transition probabilities
        transitions = np.zeros((self.n_states, self.n_states))
        for i in range(len(self.labels)-1):
            transitions[self.labels[i], self.labels[i+1]] += 1
            
        # Normalize by row sums
        row_sums = transitions.sum(axis=1)
        transitions = transitions / row_sums[:, np.newaxis]
        
        metrics['mean_transition_prob'] = np.mean(transitions)
        metrics['transition_entropy'] = -np.sum(transitions * np.log2(transitions + 1e-10))
        
        return metrics

class CrossFrequencyCoupling:
    def __init__(self, fs: float):
        """
        Initialize cross-frequency coupling analyzer.
        
        Args:
            fs: Sampling frequency
        """
        self.fs = fs
        
    def phase_amplitude_coupling(self, 
                               data: np.ndarray,
                               phase_band: Tuple[float, float],
                               amp_band: Tuple[float, float]) -> float:
        """
        Calculate phase-amplitude coupling between frequency bands.
        
        Args:
            data: EEG data array
            phase_band: Frequency range for phase (low freq)
            amp_band: Frequency range for amplitude (high freq)
            
        Returns:
            Modulation index indicating coupling strength
        """
        # Extract phase of lower frequency
        phase_filtered = self._bandpass_filter(data, phase_band[0], phase_band[1])
        phase = np.angle(signal.hilbert(phase_filtered))
        
        # Extract amplitude envelope of higher frequency
        amp_filtered = self._bandpass_filter(data, amp_band[0], amp_band[1])
        amplitude = np.abs(signal.hilbert(amp_filtered))
        
        # Calculate modulation index
        n_bins = 18
        phase_bins = np.linspace(-np.pi, np.pi, n_bins+1)
        mean_amp = np.zeros(n_bins)
        
        for i in range(n_bins):
            phase_mask = (phase >= phase_bins[i]) & (phase < phase_bins[i+1])
            mean_amp[i] = np.mean(amplitude[phase_mask])
            
        # Normalize
        mean_amp = mean_amp / np.sum(mean_amp)
        
        # Calculate entropy
        entropy = -np.sum(mean_amp * np.log(mean_amp + 1e-10))
        max_entropy = np.log(n_bins)
        
        # Modulation index
        mi = (max_entropy - entropy) / max_entropy
        return mi
        
    def _bandpass_filter(self, data: np.ndarray, low_freq: float, high_freq: float) -> np.ndarray:
        """Apply bandpass filter to data."""
        nyq = self.fs / 2
        low = low_freq / nyq
        high = high_freq / nyq
        b, a = signal.butter(3, [low, high], btype='band')
        return signal.filtfilt(b, a, data)

def compute_advanced_metrics(eeg_data: np.ndarray, 
                           fs: float,
                           channel_names: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Compute advanced EEG metrics including microstates and cross-frequency coupling.
    
    Args:
        eeg_data: EEG data array of shape (channels, samples)
        fs: Sampling frequency
        channel_names: List of channel names
        
    Returns:
        Dictionary of metrics per channel
    """
    metrics = {}
    
    # Microstate analysis
    microstate_analyzer = MicrostateAnalyzer()
    microstate_analyzer.fit(eeg_data)
    microstate_metrics = microstate_analyzer.get_metrics()
    
    # Cross-frequency coupling analysis
    cfc = CrossFrequencyCoupling(fs)
    
    # Define frequency bands for coupling analysis
    phase_bands = [
        ('theta', (4, 8)),
        ('alpha', (8, 13)),
        ('beta', (13, 30))
    ]
    amp_bands = [
        ('gamma_low', (30, 50)),
        ('gamma_high', (50, 80))
    ]
    
    # Compute metrics for each channel
    for i, channel in enumerate(channel_names):
        metrics[channel] = {}
        
        # Add microstate metrics
        for key, value in microstate_metrics.items():
            metrics[channel][f'microstate_{key}'] = value
            
        # Add cross-frequency coupling metrics
        channel_data = eeg_data[i]
        for phase_name, phase_band in phase_bands:
            for amp_name, amp_band in amp_bands:
                mi = cfc.phase_amplitude_coupling(
                    channel_data, phase_band, amp_band
                )
                metrics[channel][f'pac_{phase_name}_{amp_name}'] = mi
    
    return metrics 