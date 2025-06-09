#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TimingAttackFlagger Module

Identifies windows of heightened cognitive timing vulnerability or modulation opportunity
using cross-frequency phase-amplitude coupling (PAC). Enhanced with multi-band, cross-channel,
multi-scale, and event-related PAC, statistical validation, advanced visualization, and
integration with Swingle/Gunkelman-inspired clinical protocols and the Squiggle Interpreter.
"""

import numpy as np
import scipy.signal as signal
from typing import List, Tuple, Dict, Optional, Union
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import os
from scipy.stats import zscore
import sklearn
import pywt

logger = logging.getLogger(__name__)

class TimingAttackFlagger:
    def __init__(
        self,
        sfreq: float,
        bands: Dict[str, Tuple[float, float]] = None,
        threshold: float = 0.6,
        n_surrogates: int = 100,
        p_value: float = 0.05,
        coherence_threshold: float = 0.6,
        baseline_duration: float = 60.0,
        normative_mi_mean: float = 0.4,
        normative_mi_std: float = 0.1
    ):
        """
        Initialize the TimingAttackFlagger.

        Args:
            sfreq (float): Sampling frequency in Hz.
            bands (Dict[str, Tuple[float, float]]): Frequency bands for PAC analysis.
                Default: {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 12), 'gamma': (30, 80)}.
            threshold (float): Modulation index threshold for flagging (0 to 1). Default: 0.6.
            n_surrogates (int): Number of surrogate datasets for validation. Default: 100.
            p_value (float): P-value threshold for significant PAC. Default: 0.05.
            coherence_threshold (float): Coherence threshold for cross-channel PAC. Default: 0.6.
            baseline_duration (float): Duration (s) for baseline MI calculation. Default: 60.0.
            normative_mi_mean (float): Mean MI for z-score normalization. Default: 0.4.
            normative_mi_std (float): Std dev of MI for z-score normalization. Default: 0.1.
        """
        self.sfreq = sfreq
        self.bands = bands if bands else {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'gamma': (30, 80)
        }
        self.threshold = threshold
        self.n_surrogates = n_surrogates
        self.p_value = p_value
        self.coherence_threshold = coherence_threshold
        self.baseline_duration = baseline_duration
        self.normative_mi_mean = normative_mi_mean
        self.normative_mi_std = normative_mi_std
        self.baseline_mi = None

    def bandpass(self, data: np.ndarray, band: Tuple[float, float]) -> np.ndarray:
        """
        Apply bandpass filter to EEG data.

        Args:
            data (np.ndarray): Input EEG signal (1D).
            band (Tuple[float, float]): Frequency band (low, high) in Hz.

        Returns:
            np.ndarray: Filtered signal.
        """
        nyq = 0.5 * self.sfreq
        low, high = band[0] / nyq, band[1] / nyq
        b, a = signal.butter(4, [low, high], btype='band')
        return signal.filtfilt(b, a, data)

    def compute_coherence(self, signal1: np.ndarray, signal2: np.ndarray, freq_band: Tuple[float, float]) -> float:
        """Compute coherence between two signals in a frequency band."""
        f, Cxy = signal.coherence(signal1, signal2, fs=self.sfreq, nperseg=int(self.sfreq))
        idx = (f >= freq_band[0]) & (f <= freq_band[1])
        return np.mean(Cxy[idx]) if np.any(idx) else 0.0

    def compute_phase_amplitude_coupling(
        self,
        phase_signal: np.ndarray,
        amplitude_signal: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute phase-amplitude coupling (PAC) modulation index and p-value.

        Args:
            phase_signal (np.ndarray): Phase signal (e.g., theta phase).
            amplitude_signal (np.ndarray): Amplitude signal (e.g., gamma amplitude).

        Returns:
            Tuple[float, float]: Modulation index and p-value from surrogate testing.
        """
        n_bins = 18
        bins = np.linspace(-np.pi, np.pi, n_bins + 1)
        pac = np.zeros(n_bins)

        for i in range(n_bins):
            idx = np.where((phase_signal >= bins[i]) & (phase_signal < bins[i + 1]))[0]
            if len(idx) > 0:
                pac[i] = np.mean(amplitude_signal[idx])

        pac /= pac.sum() + 1e-10
        entropy = -np.sum(pac * np.log(pac + 1e-10)) / np.log(n_bins)
        modulation_index = 1 - entropy

        # Surrogate testing for statistical significance
        surrogate_mis = []
        for _ in range(self.n_surrogates):
            shuffled_phase = np.random.permutation(phase_signal)
            surrogate_pac = np.zeros(n_bins)
            for i in range(n_bins):
                idx = np.where((shuffled_phase >= bins[i]) & (shuffled_phase < bins[i + 1]))[0]
                if len(idx) > 0:
                    surrogate_pac[i] = np.mean(amplitude_signal[idx])
            surrogate_pac /= surrogate_pac.sum() + 1e-10
            surrogate_entropy = -np.sum(surrogate_pac * np.log(surrogate_pac + 1e-10)) / np.log(n_bins)
            surrogate_mis.append(1 - surrogate_entropy)
        
        p_value = np.mean(np.array(surrogate_mis) >= modulation_index)
        return modulation_index, p_value

    def compute_baseline_mi(
        self, eeg_data: np.ndarray, phase_band: str = 'theta', amplitude_band: str = 'gamma'
    ) -> float:
        """Compute baseline modulation index from a resting-state segment."""
        baseline_samples = int(self.baseline_duration * self.sfreq)
        if len(eeg_data) < baseline_samples:
            logger.warning("EEG data shorter than baseline duration; using full data.")
            baseline_samples = len(eeg_data)

        phase_signal = self.bandpass(eeg_data[:baseline_samples], self.bands[phase_band])
        amplitude_signal = self.bandpass(eeg_data[:baseline_samples], self.bands[amplitude_band])
        phase_data = np.angle(signal.hilbert(phase_signal))
        amplitude_data = np.abs(signal.hilbert(amplitude_signal))
        mi, _ = self.compute_phase_amplitude_coupling(phase_data, amplitude_data)
        self.baseline_mi = mi
        logger.info(f"Computed baseline MI: {mi:.4f}")
        return mi

    def normalize_pac(self, pac_scores: List[float]) -> List[float]:
        """Normalize PAC scores to z-scores using normative or baseline data."""
        if self.baseline_mi is not None:
            mean = self.baseline_mi
            std = self.normative_mi_std
        else:
            mean = self.normative_mi_mean
            std = self.normative_mi_std
        return zscore(pac_scores, mu=mean, sigma=std).tolist()

    def flag_sequences(
        self,
        eeg_data: np.ndarray,
        phase_band: str = 'theta',
        amplitude_band: str = 'gamma',
        channel_name: str = 'Unknown',
        task_timestamps: Optional[List[Tuple[float, float]]] = None
    ) -> Tuple[List[Tuple[float, float, float, float]], List[float]]:
        """
        Flag windows with high PAC for a single channel, optionally within task segments.

        Args:
            eeg_data (np.ndarray): EEG signal (1D).
            phase_band (str): Band for phase (e.g., 'theta', 'alpha'). Default: 'theta'.
            amplitude_band (str): Band for amplitude (e.g., 'gamma'). Default: 'gamma'.
            channel_name (str): Name of the channel. Default: 'Unknown'.
            task_timestamps (Optional[List[Tuple[float, float]]]): List of task segments.

        Returns:
            Tuple[List[Tuple[float, float, float, float]], List[float]]:
                - List of (start_time, end_time, modulation_index, p_value) for flagged windows.
                - List of modulation indices for all windows.
        """
        phase_signal = self.bandpass(eeg_data, self.bands[phase_band])
        amplitude_signal = self.bandpass(eeg_data, self.bands[amplitude_band])

        phase_data = np.angle(signal.hilbert(phase_signal))
        amplitude_data = np.abs(signal.hilbert(amplitude_signal))

        win_len = int(2 * self.sfreq)
        step = int(0.5 * self.sfreq)
        flags = []
        pac_scores = []

        if task_timestamps:
            for start_t, end_t in task_timestamps:
                start = int(start_t * self.sfreq)
                end = min(int(end_t * self.sfreq), len(eeg_data))
                for win_start in range(start, end - win_len, step):
                    win_end = win_start + win_len
                    mi, p_val = self.compute_phase_amplitude_coupling(
                        phase_data[win_start:win_end], amplitude_data[win_start:win_end]
                    )
                    pac_scores.append(mi)
                    if mi > self.threshold and p_val < self.p_value:
                        flags.append((win_start / self.sfreq, win_end / self.sfreq, mi, p_val))
        else:
            for start in range(0, len(eeg_data) - win_len, step):
                end = start + win_len
                mi, p_val = self.compute_phase_amplitude_coupling(
                    phase_data[start:end], amplitude_data[start:end]
                )
                pac_scores.append(mi)
                if mi > self.threshold and p_val < self.p_value:
                    flags.append((start / self.sfreq, end / self.sfreq, mi, p_val))

        logger.info(f"Flagged {len(flags)} high PAC windows for {channel_name} ({phase_band}-{amplitude_band}).")
        return flags, pac_scores

    def flag_cross_channel(
        self,
        eeg_data_phase: np.ndarray,
        eeg_data_amplitude: np.ndarray,
        phase_channel: str,
        amplitude_channel: str,
        phase_band: str = 'theta',
        amplitude_band: str = 'gamma'
    ) -> Tuple[List[Tuple[float, float, float, float]], List[float]]:
        """
        Flag windows with high cross-channel PAC, gated by coherence.

        Args:
            eeg_data_phase (np.ndarray): EEG signal for phase channel.
            eeg_data_amplitude (np.ndarray): EEG signal for amplitude channel.
            phase_channel (str): Name of phase channel.
            amplitude_channel (str): Name of amplitude channel.
            phase_band (str): Band for phase. Default: 'theta'.
            amplitude_band (str): Band for amplitude. Default: 'gamma'.

        Returns:
            Tuple[List[Tuple[float, float, float, float]], List[float]]:
                - List of (start_time, end_time, modulation_index, p_value) for flagged windows.
                - List of modulation indices for all windows.
        """
        coherence = self.compute_coherence(eeg_data_phase, eeg_data_amplitude, self.bands[phase_band])
        if coherence < self.coherence_threshold:
            logger.warning(f"Coherence ({coherence:.2f}) below threshold ({self.coherence_threshold}). Skipping PAC.")
            return [], []

        phase_signal = self.bandpass(eeg_data_phase, self.bands[phase_band])
        amplitude_signal = self.bandpass(eeg_data_amplitude, self.bands[amplitude_band])

        phase_data = np.angle(signal.hilbert(phase_signal))
        amplitude_data = np.abs(signal.hilbert(amplitude_signal))

        win_len = int(2 * self.sfreq)
        step = int(0.5 * self.sfreq)
        flags = []
        pac_scores = []

        for start in range(0, len(eeg_data_phase) - win_len, step):
            end = start + win_len
            mi, p_val = self.compute_phase_amplitude_coupling(
                phase_data[start:end], amplitude_data[start:end]
            )
            pac_scores.append(mi)
            if mi > self.threshold and p_val < self.p_value:
                flags.append((start / self.sfreq, end / self.sfreq, mi, p_val))

        logger.info(f"Flagged {len(flags)} cross-channel PAC windows ({phase_channel}-{amplitude_channel}).")
        return flags, pac_scores

    def multi_scale_flag_sequences(
        self,
        eeg_data: np.ndarray,
        phase_band: str = 'theta',
        amplitude_band: str = 'gamma',
        channel_name: str = 'Unknown',
        window_lengths: List[float] = [1.0, 2.0, 5.0]
    ) -> Dict[float, Tuple[List[Tuple[float, float, float, float]], List[float]]]:
        """Flag PAC windows across multiple time scales."""
        results = {}
        for win_len in window_lengths:
            self.threshold = self.threshold * (2.0 / win_len)  # Adjust threshold for scale
            flags, pac_scores = self.flag_sequences(
                eeg_data, phase_band, amplitude_band, channel_name
            )
            results[win_len] = (flags, pac_scores)
            self.threshold = self.threshold / (2.0 / win_len)  # Reset threshold
        return results

    def event_related_pac(
        self,
        eeg_data: np.ndarray,
        event_times: List[float],
        channel_name: str,
        window_pre: float = 0.5,
        window_post: float = 1.0,
        phase_band: str = 'theta',
        amplitude_band: str = 'gamma'
    ) -> List[Tuple[float, float, float]]:
        """Compute PAC around specific events."""
        phase_signal = self.bandpass(eeg_data, self.bands[phase_band])
        amplitude_signal = self.bandpass(eeg_data, self.bands[amplitude_band])
        phase_data = np.angle(signal.hilbert(phase_signal))
        amplitude_data = np.abs(signal.hilbert(amplitude_signal))

        results = []
        for event_time in event_times:
            start = int((event_time - window_pre) * self.sfreq)
            end = int((event_time + window_post) * self.sfreq)
            if start >= 0 and end < len(eeg_data):
                mi, p_val = self.compute_phase_amplitude_coupling(
                    phase_data[start:end], amplitude_data[start:end]
                )
                if mi > self.threshold and p_val < self.p_value:
                    results.append((event_time, mi, p_val))

        logger.info(f"Flagged {len(results)} event-related PAC windows for {channel_name}.")
        return results

    def correlate_with_vigilance(
        self,
        pac_scores: List[float],
        vigilance_states: List[Tuple[float, str]],
        time_step: float = 0.5
    ) -> Dict[str, List[float]]:
        """Correlate PAC scores with vigilance states."""
        pac_times = np.arange(0, len(pac_scores) * time_step, time_step)
        vigilance_dict = {state: [] for state in set(s for _, s in vigilance_states)}

        for i, pac in enumerate(pac_scores):
            pac_time = pac_times[i]
            closest_vigilance = min(vigilance_states, key=lambda x: abs(x[0] - pac_time), default=(0, 'Undefined'))
            vigilance_dict[closest_vigilance[1]].append(pac)

        return {state: pacs for state, pacs in vigilance_dict.items() if pacs}

    def suggest_protocols(
        self,
        flags: List[Tuple[float, float, float, float]],
        channel_name: str,
        phase_band: str = 'theta',
        amplitude_band: str = 'gamma'
    ) -> List[str]:
        """Suggest Swingle-style neurofeedback protocols based on PAC findings."""
        protocols = []
        if flags:
            avg_mi = np.mean([mi for _, _, mi, _ in flags])
            if avg_mi > 0.8:
                protocols.append(f"Inhibit high {phase_band}-{amplitude_band} PAC at {channel_name} to reduce hyperarousal.")
            else:
                protocols.append(f"Reward gamma bursts during high {phase_band} phase at {channel_name} to enhance attention.")
            protocols.append(f"SMR training at Cz (12–15 Hz) during flagged PAC windows.")
            protocols.append("Adjunct: Cognitive task training during high PAC periods.")
        else:
            protocols.append("No high PAC detected; consider standard SMR training at Cz.")
        return protocols

    def map_to_phenotype(
        self,
        pac_scores: List[float],
        channel_name: str,
        phenotypes: List[Dict]
    ) -> List[Tuple[str, float]]:
        """Map PAC scores to phenotypes from phenotype_ruleset.py."""
        matches = []
        avg_pac = np.mean(pac_scores) if pac_scores else 0.0
        for phenotype in phenotypes:
            condition = phenotype.get('condition')
            if callable(condition) and condition({'pac_theta_gamma': avg_pac}):
                matches.append((phenotype['name'], phenotype['confidence']))
        return matches

    def plot_comodulogram(
        self,
        eeg_data: np.ndarray,
        channel_name: str,
        output_dir: Union[str, Path],
        phase_band: str = 'theta',
        amplitude_band: str = 'gamma'
    ) -> Optional[str]:
        """
        Generate and save a PAC comodulogram.

        Args:
            eeg_data (np.ndarray): EEG signal (1D).
            channel_name (str): Name of the channel.
            output_dir (Union[str, Path]): Directory to save the plot.
            phase_band (str): Band for phase. Default: 'theta'.
            amplitude_band (str): Band for amplitude. Default: 'gamma'.

        Returns:
            Optional[str]: Path to saved plot, or None if plotting fails.
        """
        if not plt:
            logger.error("Matplotlib not available for plotting.")
            return None

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        phase_signal = self.bandpass(eeg_data, self.bands[phase_band])
        amplitude_signal = self.bandpass(eeg_data, self.bands[amplitude_band])
        phase_data = np.angle(signal.hilbert(phase_signal))
        amplitude_data = np.abs(signal.hilbert(amplitude_signal))

        n_bins = 18
        bins = np.linspace(-np.pi, np.pi, n_bins + 1)
        pac = np.zeros(n_bins)

        for i in range(n_bins):
            idx = np.where((phase_data >= bins[i]) & (phase_data < bins[i + 1]))[0]
            if len(idx) > 0:
                pac[i] = np.mean(amplitude_data[idx])

        plt.style.use('dark_background')
        plt.figure(figsize=(8, 6))
        plt.bar(bins[:-1], pac, width=2 * np.pi / n_bins, align='edge', color='cyan')
        plt.xlabel(f'{phase_band.capitalize()} Phase (radians)', color='white')
        plt.ylabel(f'{amplitude_band.capitalize()} Amplitude', color='white')
        plt.title(f'PAC Comodulogram ({channel_name}, {phase_band}-{amplitude_band})', color='white')
        plt.tight_layout()

        plot_path = output_dir / f"pac_comodulogram_{channel_name}_{phase_band}_{amplitude_band}.png"
        plt.savefig(plot_path, dpi=150, facecolor='black')
        plt.close()
        logger.info(f"Saved PAC comodulogram to {plot_path}")
        return str(plot_path)

    def plot_pac_timeseries(
        self,
        pac_scores: List[float],
        flags: List[Tuple[float, float, float, float]],
        channel_name: str,
        output_dir: Union[str, Path],
        phase_band: str = 'theta',
        amplitude_band: str = 'gamma',
        time_step: float = 0.5
    ) -> Optional[str]:
        """Plot PAC modulation index over time with flagged windows."""
        if not plt:
            logger.error("Matplotlib not available for plotting.")
            return None

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        times = np.arange(0, len(pac_scores) * time_step, time_step)
        plt.style.use('dark_background')
        plt.figure(figsize=(12, 5))
        plt.plot(times, pac_scores, color='cyan', label='Modulation Index')
        for start, end, mi, _ in flags:
            plt.axvspan(start, end, color='red', alpha=0.3)
        plt.xlabel('Time (s)', color='white')
        plt.ylabel('Modulation Index', color='white')
        plt.title(f'PAC Time Series ({channel_name}, {phase_band}-{amplitude_band})', color='white')
        plt.legend(facecolor='black', edgecolor='white', labelcolor='white')
        plt.tight_layout()

        plot_path = output_dir / f"pac_timeseries_{channel_name}_{phase_band}_{amplitude_band}.png"
        plt.savefig(plot_path, dpi=150, facecolor='black')
        plt.close()
        logger.info(f"Saved PAC time series to {plot_path}")
        return str(plot_path)

    def report_output(
        self,
        flags: List[Tuple[float, float, float, float]],
        channel_name: str,
        phase_band: str = 'theta',
        amplitude_band: str = 'gamma',
        cross_channel: Optional[str] = None
    ) -> str:
        """
        Generate a text report for flagged PAC windows.

        Args:
            flags (List[Tuple[float, float, float, float]]): List of flagged windows.
            channel_name (str): Name of the phase channel.
            phase_band (str): Band for phase. Default: 'theta'.
            amplitude_band (str): Band for amplitude. Default: 'gamma'.
            cross_channel (Optional[str]): Name of amplitude channel for cross-channel PAC.

        Returns:
            str: Formatted text report.
        """
        if not flags:
            return f"No high PAC ({phase_band}-{amplitude_band}) windows detected on {channel_name}.\n"

        if cross_channel:
            report = f"Cross-Channel Timing Attack Report ({channel_name}-{cross_channel}, {phase_band}-{amplitude_band}):\n"
        else:
            report = f"Timing Attack Report for Channel {channel_name} ({phase_band}-{amplitude_band}):\n"
        report += f"Detected {len(flags)} windows exceeding PAC threshold ({self.threshold}, p<{self.p_value}):\n\n"
        report += f"{'Start (s)':>10} | {'End (s)':>10} | {'Modulation Index':>18} | {'P-value':>10}\n"
        report += f"{'-'*60}\n"

        for start, end, mi, p_val in flags:
            report += f"{start:10.2f} | {end:10.2f} | {mi:18.4f} | {p_val:10.4f}\n"

        return report

    def to_html_report(
        self,
        flags: List[Tuple[float, float, float, float]],
        channel_name: str,
        pac_scores: List[float],
        plot_paths: Dict[str, str] = None,
        phase_band: str = 'theta',
        amplitude_band: str = 'gamma',
        cross_channel: Optional[str] = None,
        vigilance_correlation: Optional[Dict[str, List[float]]] = None,
        protocols: Optional[List[str]] = None,
        phenotypes: Optional[List[Tuple[str, float]]] = None,
        enhanced_results: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate an HTML report for PAC analysis.

        Args:
            flags (List[Tuple[float, float, float, float]]): List of flagged windows.
            channel_name (str): Name of the phase channel.
            pac_scores (List[float]): List of modulation indices for all windows.
            plot_paths (Dict[str, str]): Dictionary of plot paths.
            phase_band (str): Band for phase. Default: 'theta'.
            amplitude_band (str): Band for amplitude. Default: 'gamma'.
            cross_channel (Optional[str]): Name of amplitude channel for cross-channel PAC.
            vigilance_correlation (Optional[Dict[str, List[float]]]): Dictionary of vigilance state correlations.
            protocols (Optional[List[str]]): List of recommended protocols.
            phenotypes (Optional[List[Tuple[str, float]]]): List of associated phenotypes.
            enhanced_results (Optional[Dict[str, Any]]): Results from flag_sequences_enhanced.

        Returns:
            str: HTML string for the PAC report.
        """
        lines = []
        title = f"Cross-Channel PAC Analysis ({channel_name}-{cross_channel})" if cross_channel else f"PAC Analysis ({channel_name})"
        lines.append(f"<h2 style='color: #2E7D32;'>{title}</h2>")
        lines.append(f"<p><strong>Bands:</strong> {phase_band.capitalize()}-{amplitude_band.capitalize()}</p>")
        lines.append(f"<p><strong>Threshold:</strong> Modulation Index > {self.threshold}, p < {self.p_value}</p>")

        # Add validation results if available
        if enhanced_results and enhanced_results.get('validation'):
            validation = enhanced_results['validation']
            lines.append("<h3>Signal Validation</h3>")
            lines.append("<table border='1' cellpadding='4' style='border-collapse: collapse; width: 60%;'>")
            lines.append("<tr style='background-color: #E8F5E9;'><th>Metric</th><th>Value</th><th>Status</th></tr>")
            
            # Phase continuity
            status = "✓" if validation['phase_continuity'] >= 0.7 else "⚠"
            lines.append(f"<tr><td>Phase Continuity</td><td>{validation['phase_continuity']:.3f}</td><td>{status}</td></tr>")
            
            # Amplitude stability
            status = "✓" if validation['amplitude_stability'] >= 0.5 else "⚠"
            lines.append(f"<tr><td>Amplitude Stability</td><td>{validation['amplitude_stability']:.3f}</td><td>{status}</td></tr>")
            
            lines.append("</table>")

            if validation['warnings']:
                lines.append("<div style='color: #FFA500; margin: 10px 0;'>")
                lines.append("<strong>Validation Warnings:</strong>")
                lines.append("<ul>")
                for warning in validation['warnings']:
                    lines.append(f"<li>{warning}</li>")
                lines.append("</ul>")
                lines.append("</div>")

        # Add artifact detection results if available
        if enhanced_results and enhanced_results.get('artifacts'):
            lines.append("<h3>Artifact Detection</h3>")
            artifacts = enhanced_results['artifacts']
            if any(artifacts.values()):
                lines.append("<table border='1' cellpadding='4' style='border-collapse: collapse; width: 60%;'>")
                lines.append("<tr style='background-color: #E8F5E9;'><th>Time (s)</th><th>Type</th><th>Severity</th></tr>")
                
                for artifact_type, detections in artifacts.items():
                    for time, type_name, severity in detections:
                        lines.append(f"<tr><td>{time:.2f}</td><td>{type_name}</td><td>{severity:.2f}</td></tr>")
                
                lines.append("</table>")
            else:
                lines.append("<p>No significant artifacts detected.</p>")

        # Add additional coupling measures if available
        if enhanced_results:
            if enhanced_results.get('phase_phase_coupling'):
                lines.append("<h3>Phase-Phase Coupling</h3>")
                lines.append("<table border='1' cellpadding='4' style='border-collapse: collapse; width: 60%;'>")
                lines.append("<tr style='background-color: #E8F5E9;'><th>Band Pair</th><th>PLV</th><th>P-value</th></tr>")
                
                for pair, results in enhanced_results['phase_phase_coupling'].items():
                    lines.append(f"<tr><td>{pair}</td><td>{results['plv']:.3f}</td><td>{results['p_value']:.3f}</td></tr>")
                
                lines.append("</table>")

            if enhanced_results.get('amplitude_amplitude_coupling'):
                lines.append("<h3>Amplitude-Amplitude Coupling</h3>")
                lines.append("<table border='1' cellpadding='4' style='border-collapse: collapse; width: 60%;'>")
                lines.append("<tr style='background-color: #E8F5E9;'><th>Band Pair</th><th>Correlation</th><th>P-value</th></tr>")
                
                for pair, results in enhanced_results['amplitude_amplitude_coupling'].items():
                    lines.append(f"<tr><td>{pair}</td><td>{results['correlation']:.3f}</td><td>{results['p_value']:.3f}</td></tr>")
                
                lines.append("</table>")

        # Original PAC analysis results
        z_scores = self.normalize_pac(pac_scores)
        lines.append("<h3>PAC Score Statistics</h3>")
        lines.append(f"<p><strong>Mean Modulation Index:</strong> {np.mean(pac_scores):.4f}</p>")
        lines.append(f"<p><strong>Std Deviation:</strong> {np.std(pac_scores):.4f}</p>")
        lines.append(f"<p><strong>Mean Z-Score:</strong> {np.mean(z_scores):.2f}</p>")

        if flags:
            lines.append(f"<p><strong>Detected Windows:</strong> {len(flags)}</p>")
            lines.append("<table border='1' cellpadding='4' style='border-collapse: collapse; width: 60%;'>")
            lines.append("<tr style='background-color: #E8F5E9;'><th>Start (s)</th><th>End (s)</th><th>Modulation Index</th><th>P-value</th></tr>")
            for start, end, mi, p_val in flags:
                lines.append(f"<tr><td>{start:.2f}</td><td>{end:.2f}</td><td>{mi:.4f}</td><td>{p_val:.4f}</td></tr>")
            lines.append("</table>")
        else:
            lines.append("<p>No high PAC windows detected.</p>")

        if plot_paths:
            for plot_type, plot_path in plot_paths.items():
                lines.append(f"<h3>{plot_type}</h3>")
                lines.append(f"<img src='{plot_path}' alt='{plot_type}' style='max-width: 600px;'>")

        if vigilance_correlation:
            lines.append("<h3>Vigilance State Correlation</h3>")
            lines.append("<ul>")
            for state, scores in vigilance_correlation.items():
                lines.append(f"<li>State {state}: Mean PAC = {np.mean(scores):.4f} (n={len(scores)})</li>")
            lines.append("</ul>")

        if protocols:
            lines.append("<h3>Recommended Protocols</h3>")
            lines.append("<ul>")
            for protocol in protocols:
                lines.append(f"<li>{protocol}</li>")
            lines.append("</ul>")

        if phenotypes:
            lines.append("<h3>Associated Phenotypes</h3>")
            lines.append("<ul>")
            for name, confidence in phenotypes:
                lines.append(f"<li>{name} (Confidence: {confidence:.2f})</li>")
            lines.append("</ul>")

        # Add any warnings from enhanced analysis
        if enhanced_results and enhanced_results.get('warnings'):
            lines.append("<div style='color: #FFA500; margin: 10px 0;'>")
            lines.append("<strong>Analysis Warnings:</strong>")
            lines.append("<ul>")
            for warning in enhanced_results['warnings']:
                lines.append(f"<li>{warning}</li>")
            lines.append("</ul>")
            lines.append("</div>")

        return "\n".join(lines)

    def export_qeeg_report(
        self,
        flags: List[Tuple[float, float, float, float]],
        pac_scores: List[float],
        channel_name: str,
        phase_band: str = 'theta',
        amplitude_band: str = 'gamma'
    ) -> Dict:
        """Export PAC results in a qEEG-compatible format."""
        z_scores = self.normalize_pac(pac_scores)
        return {
            'channel': channel_name,
            'phase_band': phase_band,
            'amplitude_band': amplitude_band,
            'flags': flags,
            'mean_mi': float(np.mean(pac_scores)),
            'std_mi': float(np.std(pac_scores)),
            'mean_z_score': float(np.mean(z_scores)),
            'std_z_score': float(np.std(z_scores))
        }

    def compute_phase_phase_coupling(
        self,
        signal1: np.ndarray,
        signal2: np.ndarray,
        band1: Tuple[float, float],
        band2: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        Compute phase-phase coupling between two frequency bands.

        Args:
            signal1 (np.ndarray): First input signal
            signal2 (np.ndarray): Second input signal
            band1 (Tuple[float, float]): Frequency band for first signal
            band2 (Tuple[float, float]): Frequency band for second signal

        Returns:
            Tuple[float, float]: Phase locking value and p-value
        """
        # Filter signals in respective bands
        sig1_filt = self.bandpass(signal1, band1)
        sig2_filt = self.bandpass(signal2, band2)

        # Extract instantaneous phases
        phase1 = np.angle(signal.hilbert(sig1_filt))
        phase2 = np.angle(signal.hilbert(sig2_filt))

        # Compute phase difference
        phase_diff = phase1 - phase2
        plv = np.abs(np.mean(np.exp(1j * phase_diff)))

        # Surrogate testing
        surrogate_plvs = []
        for _ in range(self.n_surrogates):
            shuffled_phase2 = np.random.permutation(phase2)
            phase_diff_surr = phase1 - shuffled_phase2
            plv_surr = np.abs(np.mean(np.exp(1j * phase_diff_surr)))
            surrogate_plvs.append(plv_surr)

        p_value = np.mean(np.array(surrogate_plvs) >= plv)
        return plv, p_value

    def compute_amplitude_amplitude_coupling(
        self,
        signal1: np.ndarray,
        signal2: np.ndarray,
        band1: Tuple[float, float],
        band2: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        Compute amplitude-amplitude coupling between two frequency bands.

        Args:
            signal1 (np.ndarray): First input signal
            signal2 (np.ndarray): Second input signal
            band1 (Tuple[float, float]): Frequency band for first signal
            band2 (Tuple[float, float]): Frequency band for second signal

        Returns:
            Tuple[float, float]: Correlation coefficient and p-value
        """
        # Filter signals and get amplitude envelopes
        sig1_filt = self.bandpass(signal1, band1)
        sig2_filt = self.bandpass(signal2, band2)
        
        amp1 = np.abs(signal.hilbert(sig1_filt))
        amp2 = np.abs(signal.hilbert(sig2_filt))

        # Compute correlation
        corr = np.corrcoef(amp1, amp2)[0, 1]

        # Surrogate testing
        surrogate_corrs = []
        for _ in range(self.n_surrogates):
            shuffled_amp2 = np.random.permutation(amp2)
            corr_surr = np.corrcoef(amp1, shuffled_amp2)[0, 1]
            surrogate_corrs.append(corr_surr)

        p_value = np.mean(np.array(surrogate_corrs) >= corr)
        return corr, p_value

    def detect_artifacts(
        self,
        eeg_data: np.ndarray,
        channel_name: str
    ) -> Dict[str, List[Tuple[float, str, float]]]:
        """
        Detect potential artifacts that could affect PAC analysis.

        Args:
            eeg_data (np.ndarray): EEG signal
            channel_name (str): Name of the channel

        Returns:
            Dict[str, List[Tuple[float, str, float]]]: Dictionary of artifact detections
                with timestamps, types, and severity scores
        """
        artifacts = {
            'muscle': [],
            'movement': [],
            'line_noise': [],
            'saturation': []
        }

        # Window parameters
        win_len = int(0.5 * self.sfreq)  # 500ms windows
        step = int(0.1 * self.sfreq)  # 100ms steps

        for start in range(0, len(eeg_data) - win_len, step):
            window = eeg_data[start:start + win_len]
            time = start / self.sfreq

            # Muscle artifact detection (high frequency power)
            gamma_power = np.mean(np.abs(self.bandpass(window, (30, 100))))
            if gamma_power > np.std(eeg_data) * 3:
                artifacts['muscle'].append((time, 'EMG', gamma_power))

            # Movement artifact detection (sudden amplitude changes)
            amp_diff = np.diff(window)
            if np.max(np.abs(amp_diff)) > np.std(eeg_data) * 4:
                artifacts['movement'].append((time, 'Movement', np.max(np.abs(amp_diff))))

            # Line noise detection (50/60 Hz)
            fft = np.fft.fft(window)
            freqs = np.fft.fftfreq(len(window), 1/self.sfreq)
            line_freq_mask = (np.abs(freqs - 50) < 1) | (np.abs(freqs - 60) < 1)
            line_power = np.mean(np.abs(fft[line_freq_mask]))
            if line_power > np.mean(np.abs(fft)) * 2:
                artifacts['line_noise'].append((time, 'Line Noise', line_power))

            # Saturation detection
            if np.max(np.abs(window)) > np.std(eeg_data) * 5:
                artifacts['saturation'].append((time, 'Saturation', np.max(np.abs(window))))

        return artifacts

    def validate_pac_computation(
        self,
        eeg_data: np.ndarray,
        phase_band: str,
        amplitude_band: str
    ) -> Dict[str, Union[bool, str, float]]:
        """
        Validate PAC computation and check for potential issues.

        Args:
            eeg_data (np.ndarray): EEG signal
            phase_band (str): Band for phase
            amplitude_band (str): Band for amplitude

        Returns:
            Dict[str, Union[bool, str, float]]: Validation results
        """
        results = {
            'is_valid': True,
            'warnings': [],
            'phase_continuity': 0.0,
            'amplitude_stability': 0.0
        }

        # Check signal length
        if len(eeg_data) < 5 * self.sfreq:  # Minimum 5 seconds
            results['is_valid'] = False
            results['warnings'].append("Signal too short for reliable PAC computation")
            return results

        # Phase signal validation
        phase_signal = self.bandpass(eeg_data, self.bands[phase_band])
        phase = np.angle(signal.hilbert(phase_signal))
        phase_diff = np.diff(phase)
        phase_diff = np.mod(phase_diff + np.pi, 2 * np.pi) - np.pi
        phase_continuity = 1 - (np.std(phase_diff) / np.pi)
        results['phase_continuity'] = phase_continuity

        if phase_continuity < 0.7:
            results['warnings'].append("Poor phase continuity detected")

        # Amplitude signal validation
        amp_signal = self.bandpass(eeg_data, self.bands[amplitude_band])
        amplitude = np.abs(signal.hilbert(amp_signal))
        amp_stability = 1 - (np.std(amplitude) / np.mean(amplitude))
        results['amplitude_stability'] = amp_stability

        if amp_stability < 0.5:
            results['warnings'].append("Unstable amplitude envelope detected")

        # Check for edge effects
        edge_samples = int(0.1 * self.sfreq)  # 100ms
        edge_var_ratio = (np.var(phase_signal[:edge_samples]) + np.var(phase_signal[-edge_samples:])) / \
                        (2 * np.var(phase_signal[edge_samples:-edge_samples]))
        
        if edge_var_ratio > 2.0:
            results['warnings'].append("Filter edge effects detected")

        return results

    def flag_sequences_enhanced(
        self,
        eeg_data: np.ndarray,
        phase_band: str = 'theta',
        amplitude_band: str = 'gamma',
        channel_name: str = 'Unknown',
        task_timestamps: Optional[List[Tuple[float, float]]] = None,
        artifact_rejection: bool = True,
        validation_check: bool = True
    ) -> Dict[str, Any]:
        """
        Enhanced version of flag_sequences with additional detection vectors.

        Args:
            eeg_data (np.ndarray): EEG signal
            phase_band (str): Band for phase
            amplitude_band (str): Band for amplitude
            channel_name (str): Name of the channel
            task_timestamps (Optional[List[Tuple[float, float]]]): Task segments
            artifact_rejection (bool): Whether to perform artifact rejection
            validation_check (bool): Whether to validate PAC computation

        Returns:
            Dict[str, Any]: Enhanced flagging results including artifacts and validation
        """
        results = {
            'flags': [],
            'pac_scores': [],
            'artifacts': None,
            'validation': None,
            'warnings': []
        }

        # Artifact detection if enabled
        if artifact_rejection:
            artifacts = self.detect_artifacts(eeg_data, channel_name)
            results['artifacts'] = artifacts
            
            # Create artifact-free signal
            artifact_mask = np.ones(len(eeg_data), dtype=bool)
            for artifact_type in artifacts.values():
                for t, _, _ in artifact_type:
                    start_idx = int(t * self.sfreq)
                    end_idx = start_idx + int(0.5 * self.sfreq)
                    artifact_mask[start_idx:end_idx] = False
            
            eeg_data = eeg_data[artifact_mask]
            if len(eeg_data) < 5 * self.sfreq:
                results['warnings'].append("Insufficient clean data after artifact rejection")
                return results

        # Validation check if enabled
        if validation_check:
            validation = self.validate_pac_computation(eeg_data, phase_band, amplitude_band)
            results['validation'] = validation
            if not validation['is_valid']:
                results['warnings'].extend(validation['warnings'])
                return results

        # Compute standard PAC flags
        flags, pac_scores = self.flag_sequences(
            eeg_data,
            phase_band,
            amplitude_band,
            channel_name,
            task_timestamps
        )
        results['flags'] = flags
        results['pac_scores'] = pac_scores

        # Additional coupling measures
        if len(eeg_data) >= 5 * self.sfreq:
            # Phase-phase coupling between adjacent bands
            phase_phase_results = {}
            for band1, band2 in [('theta', 'alpha'), ('alpha', 'beta')]:
                plv, p_val = self.compute_phase_phase_coupling(
                    eeg_data, eeg_data,
                    self.bands[band1], self.bands[band2]
                )
                phase_phase_results[f"{band1}_{band2}"] = {'plv': plv, 'p_value': p_val}
            results['phase_phase_coupling'] = phase_phase_results

            # Amplitude-amplitude coupling
            amp_amp_results = {}
            for band1, band2 in [('theta', 'gamma'), ('alpha', 'gamma')]:
                corr, p_val = self.compute_amplitude_amplitude_coupling(
                    eeg_data, eeg_data,
                    self.bands[band1], self.bands[band2]
                )
                amp_amp_results[f"{band1}_{band2}"] = {'correlation': corr, 'p_value': p_val}
            results['amplitude_amplitude_coupling'] = amp_amp_results

        return results

    def compute_mvl_pac(
        self,
        phase_signal: np.ndarray,
        amplitude_signal: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute PAC using Mean Vector Length method (Canolty et al., 2006).

        Args:
            phase_signal (np.ndarray): Phase signal
            amplitude_signal (np.ndarray): Amplitude signal

        Returns:
            Tuple[float, float]: MVL modulation index and p-value
        """
        # Compute complex-valued composite signal
        z = amplitude_signal * np.exp(1j * phase_signal)
        mvl = np.abs(np.mean(z))
        
        # Surrogate testing
        surrogate_mvls = []
        for _ in range(self.n_surrogates):
            shuffled_phase = np.random.permutation(phase_signal)
            z_surr = amplitude_signal * np.exp(1j * shuffled_phase)
            surrogate_mvls.append(np.abs(np.mean(z_surr)))
        
        p_value = np.mean(np.array(surrogate_mvls) >= mvl)
        return mvl, p_value

    def compute_glm_pac(
        self,
        phase_signal: np.ndarray,
        amplitude_signal: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute PAC using General Linear Model method (Penny et al., 2008).

        Args:
            phase_signal (np.ndarray): Phase signal
            amplitude_signal (np.ndarray): Amplitude signal

        Returns:
            Tuple[float, float]: GLM modulation index and p-value
        """
        # Prepare design matrix
        X = np.column_stack([
            np.cos(phase_signal),
            np.sin(phase_signal),
            np.ones_like(phase_signal)
        ])
        
        # Fit GLM
        beta = np.linalg.lstsq(X, amplitude_signal, rcond=None)[0]
        predicted = X @ beta
        residuals = amplitude_signal - predicted
        r_squared = 1 - np.var(residuals) / np.var(amplitude_signal)
        
        # Surrogate testing
        surrogate_r2s = []
        for _ in range(self.n_surrogates):
            shuffled_phase = np.random.permutation(phase_signal)
            X_surr = np.column_stack([
                np.cos(shuffled_phase),
                np.sin(shuffled_phase),
                np.ones_like(shuffled_phase)
            ])
            beta_surr = np.linalg.lstsq(X_surr, amplitude_signal, rcond=None)[0]
            predicted_surr = X_surr @ beta_surr
            residuals_surr = amplitude_signal - predicted_surr
            r2_surr = 1 - np.var(residuals_surr) / np.var(amplitude_signal)
            surrogate_r2s.append(r2_surr)
        
        p_value = np.mean(np.array(surrogate_r2s) >= r_squared)
        return r_squared, p_value

    def compute_cluster_stats(
        self,
        coupling_scores: np.ndarray,
        threshold: float,
        n_permutations: int = 1000
    ) -> Dict[str, Any]:
        """
        Perform cluster-based permutation test for coupling scores.

        Args:
            coupling_scores (np.ndarray): Array of coupling scores
            threshold (float): Initial threshold for cluster formation
            n_permutations (int): Number of permutations for testing

        Returns:
            Dict[str, Any]: Dictionary containing cluster statistics
        """
        def find_clusters(data: np.ndarray, thresh: float) -> List[Tuple[int, int]]:
            """Find continuous clusters above threshold."""
            clusters = []
            current_cluster = None
            
            for i, value in enumerate(data):
                if value > thresh:
                    if current_cluster is None:
                        current_cluster = [i]
                    else:
                        current_cluster.append(i)
                elif current_cluster is not None:
                    clusters.append((current_cluster[0], current_cluster[-1]))
                    current_cluster = None
            
            if current_cluster is not None:
                clusters.append((current_cluster[0], current_cluster[-1]))
            
            return clusters

        # Find observed clusters
        observed_clusters = find_clusters(coupling_scores, threshold)
        cluster_stats = {
            'clusters': [],
            'significant_clusters': [],
            'max_stat_dist': []
        }

        if not observed_clusters:
            return cluster_stats

        # Compute cluster statistics
        for start, end in observed_clusters:
            cluster_sum = np.sum(coupling_scores[start:end+1])
            cluster_stats['clusters'].append({
                'start': start,
                'end': end,
                'sum_stat': cluster_sum
            })

        # Permutation testing
        max_cluster_sums = []
        for _ in range(n_permutations):
            perm_scores = np.random.permutation(coupling_scores)
            perm_clusters = find_clusters(perm_scores, threshold)
            if perm_clusters:
                max_sum = max(
                    np.sum(perm_scores[start:end+1])
                    for start, end in perm_clusters
                )
                max_cluster_sums.append(max_sum)
            else:
                max_cluster_sums.append(0)

        # Determine significance
        max_cluster_sums = np.array(max_cluster_sums)
        for cluster in cluster_stats['clusters']:
            p_value = np.mean(max_cluster_sums >= cluster['sum_stat'])
            if p_value < self.p_value:
                cluster_stats['significant_clusters'].append({
                    'start': cluster['start'],
                    'end': cluster['end'],
                    'sum_stat': cluster['sum_stat'],
                    'p_value': p_value
                })

        cluster_stats['max_stat_dist'] = max_cluster_sums.tolist()
        return cluster_stats

    def compute_advanced_coupling_stats(
        self,
        eeg_data: np.ndarray,
        channel_name: str,
        phase_band: str = 'theta',
        amplitude_band: str = 'gamma'
    ) -> Dict[str, Any]:
        """
        Compute comprehensive coupling statistics using multiple methods.

        Args:
            eeg_data (np.ndarray): EEG signal
            channel_name (str): Channel name
            phase_band (str): Phase frequency band
            amplitude_band (str): Amplitude frequency band

        Returns:
            Dict[str, Any]: Dictionary containing all coupling metrics
        """
        # Filter signals
        phase_signal = self.bandpass(eeg_data, self.bands[phase_band])
        amplitude_signal = self.bandpass(eeg_data, self.bands[amplitude_band])
        
        # Extract phase and amplitude
        phase = np.angle(signal.hilbert(phase_signal))
        amplitude = np.abs(signal.hilbert(amplitude_signal))
        
        # Compute coupling using different methods
        klmi, klmi_p = self.compute_phase_amplitude_coupling(phase, amplitude)
        mvl, mvl_p = self.compute_mvl_pac(phase, amplitude)
        glm, glm_p = self.compute_glm_pac(phase, amplitude)
        
        # Combine scores for cluster analysis
        coupling_scores = np.array([klmi, mvl, glm])
        cluster_stats = self.compute_cluster_stats(
            coupling_scores,
            threshold=self.threshold
        )
        
        return {
            'channel': channel_name,
            'phase_band': phase_band,
            'amplitude_band': amplitude_band,
            'methods': {
                'klmi': {'score': klmi, 'p_value': klmi_p},
                'mvl': {'score': mvl, 'p_value': mvl_p},
                'glm': {'score': glm, 'p_value': glm_p}
            },
            'cluster_stats': cluster_stats,
            'consensus_score': np.mean([klmi, mvl, glm]),
            'method_agreement': np.std([klmi, mvl, glm])
        }

    def flag_sequences_advanced(
        self,
        eeg_data: np.ndarray,
        phase_band: str = 'theta',
        amplitude_band: str = 'gamma',
        channel_name: str = 'Unknown',
        task_timestamps: Optional[List[Tuple[float, float]]] = None,
        min_agreement: float = 0.5
    ) -> Dict[str, Any]:
        """
        Advanced flagging using multiple coupling methods and statistical validation.

        Args:
            eeg_data (np.ndarray): EEG signal
            phase_band (str): Phase frequency band
            amplitude_band (str): Amplitude frequency band
            channel_name (str): Channel name
            task_timestamps (Optional[List[Tuple[float, float]]]): Task segments
            min_agreement (float): Minimum agreement between methods

        Returns:
            Dict[str, Any]: Dictionary containing flagged windows and statistics
        """
        win_len = int(2 * self.sfreq)
        step = int(0.5 * self.sfreq)
        results = {
            'flags': [],
            'method_scores': [],
            'cluster_stats': [],
            'consensus_flags': []
        }
        
        if task_timestamps:
            for start_t, end_t in task_timestamps:
                start = int(start_t * self.sfreq)
                end = min(int(end_t * self.sfreq), len(eeg_data))
                for win_start in range(start, end - win_len, step):
                    win_end = win_start + win_len
                    window_stats = self.compute_advanced_coupling_stats(
                        eeg_data[win_start:win_end],
                        channel_name,
                        phase_band,
                        amplitude_band
                    )
                    
                    # Check if methods agree on significance
                    significant_methods = sum(
                        1 for method in window_stats['methods'].values()
                        if method['score'] > self.threshold and method['p_value'] < self.p_value
                    )
                    method_agreement = significant_methods / len(window_stats['methods'])
                    
                    if method_agreement >= min_agreement:
                        results['consensus_flags'].append((
                            win_start / self.sfreq,
                            win_end / self.sfreq,
                            window_stats['consensus_score'],
                            window_stats['method_agreement']
                        ))
                    
                    results['method_scores'].append(window_stats['methods'])
                    results['cluster_stats'].append(window_stats['cluster_stats'])
        else:
            for start in range(0, len(eeg_data) - win_len, step):
                end = start + win_len
                window_stats = self.compute_advanced_coupling_stats(
                    eeg_data[start:end],
                    channel_name,
                    phase_band,
                    amplitude_band
                )
                
                significant_methods = sum(
                    1 for method in window_stats['methods'].values()
                    if method['score'] > self.threshold and method['p_value'] < self.p_value
                )
                method_agreement = significant_methods / len(window_stats['methods'])
                
                if method_agreement >= min_agreement:
                    results['consensus_flags'].append((
                        start / self.sfreq,
                        end / self.sfreq,
                        window_stats['consensus_score'],
                        window_stats['method_agreement']
                    ))
                
                results['method_scores'].append(window_stats['methods'])
                results['cluster_stats'].append(window_stats['cluster_stats'])
        
        logger.info(
            f"Flagged {len(results['consensus_flags'])} consensus windows "
            f"for {channel_name} ({phase_band}-{amplitude_band})."
        )
        return results

    def detect_artifacts_ml(
        self,
        eeg_data: np.ndarray,
        channel_name: str,
        classifier_type: str = 'svm',
        feature_set: str = 'standard'
    ) -> Dict[str, List[Tuple[float, str, float, float]]]:
        """
        Enhanced artifact detection using machine learning classifiers.

        Args:
            eeg_data (np.ndarray): EEG signal
            channel_name (str): Channel name
            classifier_type (str): Type of classifier ('svm', 'rf', 'nn')
            feature_set (str): Feature set to use ('standard', 'extended')

        Returns:
            Dict[str, List[Tuple[float, str, float, float]]]: 
                Dictionary of artifact detections with timestamps, types, 
                severity scores, and confidence scores
        """
        artifacts = {
            'muscle': [],
            'movement': [],
            'line_noise': [],
            'saturation': [],
            'blink': [],
            'saccade': []
        }

        # Window parameters
        win_len = int(0.5 * self.sfreq)  # 500ms windows
        step = int(0.1 * self.sfreq)  # 100ms steps

        def extract_features(window: np.ndarray) -> np.ndarray:
            """Extract features for artifact detection."""
            features = []
            
            # Time domain features
            features.extend([
                np.mean(window),
                np.std(window),
                np.max(np.abs(window)),
                np.mean(np.abs(np.diff(window))),
                scipy.stats.skew(window),
                scipy.stats.kurtosis(window)
            ])
            
            # Frequency domain features
            freqs, psd = signal.welch(window, self.sfreq, nperseg=min(256, len(window)))
            total_power = np.sum(psd)
            delta_power = np.sum(psd[(freqs >= 1) & (freqs <= 4)]) / total_power
            theta_power = np.sum(psd[(freqs >= 4) & (freqs <= 8)]) / total_power
            alpha_power = np.sum(psd[(freqs >= 8) & (freqs <= 13)]) / total_power
            beta_power = np.sum(psd[(freqs >= 13) & (freqs <= 30)]) / total_power
            gamma_power = np.sum(psd[freqs >= 30]) / total_power
            line_power = np.sum(psd[(freqs >= 49) & (freqs <= 51)]) / total_power
            
            features.extend([
                delta_power, theta_power, alpha_power,
                beta_power, gamma_power, line_power
            ])
            
            if feature_set == 'extended':
                # Wavelet features
                coeffs = pywt.wavedec(window, 'db4', level=4)
                features.extend([np.std(c) for c in coeffs])
                
                # Hjorth parameters
                activity = np.var(window)
                mobility = np.sqrt(np.var(np.diff(window)) / activity)
                complexity = np.sqrt(np.var(np.diff(np.diff(window))) / np.var(np.diff(window))) / mobility
                features.extend([activity, mobility, complexity])
            
            return np.array(features)

        def get_classifier():
            """Initialize the selected classifier."""
            if classifier_type == 'svm':
                return sklearn.svm.SVC(probability=True)
            elif classifier_type == 'rf':
                return sklearn.ensemble.RandomForestClassifier()
            elif classifier_type == 'nn':
                return sklearn.neural_network.MLPClassifier()
            else:
                raise ValueError(f"Unsupported classifier type: {classifier_type}")

        # Initialize classifier (in practice, this would be pre-trained)
        classifier = get_classifier()

        for start in range(0, len(eeg_data) - win_len, step):
            window = eeg_data[start:start + win_len]
            time = start / self.sfreq
            
            # Extract features
            features = extract_features(window)
            
            # Get classifier predictions and probabilities
            artifact_type = classifier.predict([features])[0]
            probabilities = classifier.predict_proba([features])[0]
            confidence = np.max(probabilities)
            
            # Compute severity score based on feature values
            severity = np.mean([
                np.abs(features[0]) / np.std(eeg_data),  # Normalized amplitude
                features[5],  # Gamma power ratio
                features[4]   # Line noise ratio
            ])
            
            if confidence > 0.8:  # High confidence threshold
                artifacts[artifact_type].append((time, artifact_type, severity, confidence))

        return artifacts

    def plot_advanced_coupling(
        self,
        eeg_data: np.ndarray,
        channel_name: str,
        output_dir: Union[str, Path],
        phase_band: str = 'theta',
        amplitude_band: str = 'gamma',
        plot_type: str = 'all'
    ) -> Dict[str, str]:
        """
        Generate advanced coupling visualizations.

        Args:
            eeg_data (np.ndarray): EEG signal
            channel_name (str): Channel name
            output_dir (Union[str, Path]): Output directory
            phase_band (str): Phase frequency band
            amplitude_band (str): Amplitude frequency band
            plot_type (str): Type of plots to generate ('all', 'pac', 'ppc', 'aac')

        Returns:
            Dict[str, str]: Dictionary of plot paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_paths = {}

        if plot_type in ['all', 'pac']:
            # Enhanced PAC visualization
            phase_signal = self.bandpass(eeg_data, self.bands[phase_band])
            amplitude_signal = self.bandpass(eeg_data, self.bands[amplitude_band])
            phase = np.angle(signal.hilbert(phase_signal))
            amplitude = np.abs(signal.hilbert(amplitude_signal))

            plt.figure(figsize=(15, 10))
            
            # Main plot: Phase-amplitude distribution
            plt.subplot(2, 2, 1)
            n_bins = 18
            phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
            mean_amp = np.zeros(n_bins)
            std_amp = np.zeros(n_bins)
            
            for i in range(n_bins):
                idx = np.where((phase >= phase_bins[i]) & (phase < phase_bins[i + 1]))[0]
                if len(idx) > 0:
                    mean_amp[i] = np.mean(amplitude[idx])
                    std_amp[i] = np.std(amplitude[idx])
            
            plt.errorbar(phase_bins[:-1], mean_amp, yerr=std_amp, fmt='o-', capsize=5)
            plt.xlabel(f'{phase_band.capitalize()} Phase (rad)')
            plt.ylabel(f'{amplitude_band.capitalize()} Amplitude')
            plt.title('Phase-Amplitude Distribution')
            
            # Subplot: Time-frequency representation
            plt.subplot(2, 2, 2)
            f, t, Sxx = signal.spectrogram(eeg_data, fs=self.sfreq, nperseg=256)
            plt.pcolormesh(t, f, 10 * np.log10(Sxx))
            plt.ylabel('Frequency (Hz)')
            plt.xlabel('Time (s)')
            plt.title('Time-Frequency Representation')
            
            # Subplot: Phase histogram
            plt.subplot(2, 2, 3)
            plt.hist(phase, bins=36, density=True)
            plt.xlabel('Phase (rad)')
            plt.ylabel('Density')
            plt.title(f'{phase_band.capitalize()} Phase Distribution')
            
            # Subplot: Amplitude histogram
            plt.subplot(2, 2, 4)
            plt.hist(amplitude, bins=50, density=True)
            plt.xlabel('Amplitude')
            plt.ylabel('Density')
            plt.title(f'{amplitude_band.capitalize()} Amplitude Distribution')
            
            plt.tight_layout()
            plot_path = output_dir / f"advanced_pac_{channel_name}_{phase_band}_{amplitude_band}.png"
            plt.savefig(plot_path, dpi=300)
            plt.close()
            plot_paths['pac'] = str(plot_path)

        if plot_type in ['all', 'ppc']:
            # Phase-Phase Coupling visualization
            plt.figure(figsize=(10, 8))
            freqs = np.array(list(self.bands.values()))
            n_freqs = len(freqs)
            ppc_matrix = np.zeros((n_freqs, n_freqs))
            
            for i, (band1_name, band1) in enumerate(self.bands.items()):
                for j, (band2_name, band2) in enumerate(self.bands.items()):
                    if i < j:  # Upper triangle only
                        plv, _ = self.compute_phase_phase_coupling(
                            eeg_data, eeg_data, band1, band2
                        )
                        ppc_matrix[i, j] = plv
            
            plt.imshow(ppc_matrix, cmap='viridis', aspect='auto')
            plt.colorbar(label='Phase Locking Value')
            plt.xticks(range(n_freqs), self.bands.keys())
            plt.yticks(range(n_freqs), self.bands.keys())
            plt.title(f'Phase-Phase Coupling Matrix ({channel_name})')
            
            plot_path = output_dir / f"ppc_matrix_{channel_name}.png"
            plt.savefig(plot_path, dpi=300)
            plt.close()
            plot_paths['ppc'] = str(plot_path)

        if plot_type in ['all', 'aac']:
            # Amplitude-Amplitude Coupling visualization
            plt.figure(figsize=(10, 8))
            aac_matrix = np.zeros((n_freqs, n_freqs))
            
            for i, (band1_name, band1) in enumerate(self.bands.items()):
                for j, (band2_name, band2) in enumerate(self.bands.items()):
                    if i < j:  # Upper triangle only
                        corr, _ = self.compute_amplitude_amplitude_coupling(
                            eeg_data, eeg_data, band1, band2
                        )
                        aac_matrix[i, j] = corr
            
            plt.imshow(aac_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            plt.colorbar(label='Correlation Coefficient')
            plt.xticks(range(n_freqs), self.bands.keys())
            plt.yticks(range(n_freqs), self.bands.keys())
            plt.title(f'Amplitude-Amplitude Coupling Matrix ({channel_name})')
            
            plot_path = output_dir / f"aac_matrix_{channel_name}.png"
            plt.savefig(plot_path, dpi=300)
            plt.close()
            plot_paths['aac'] = str(plot_path)

        return plot_paths

    def generate_comprehensive_report(
        self,
        eeg_data: np.ndarray,
        channel_name: str,
        output_dir: Union[str, Path],
        phase_band: str = 'theta',
        amplitude_band: str = 'gamma'
    ) -> str:
        """
        Generate a comprehensive analysis report including artifacts and coupling.

        Args:
            eeg_data (np.ndarray): EEG signal
            channel_name (str): Channel name
            output_dir (Union[str, Path]): Output directory
            phase_band (str): Phase frequency band
            amplitude_band (str): Amplitude frequency band

        Returns:
            str: Path to the generated HTML report
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run all analyses
        artifacts = self.detect_artifacts_ml(eeg_data, channel_name)
        coupling_stats = self.compute_advanced_coupling_stats(
            eeg_data, channel_name, phase_band, amplitude_band
        )
        plot_paths = self.plot_advanced_coupling(
            eeg_data, channel_name, output_dir, phase_band, amplitude_band
        )

        # Generate HTML report
        html_lines = []
        html_lines.append("<html><body>")
        html_lines.append(f"<h1>Comprehensive Analysis Report: {channel_name}</h1>")
        
        # Artifact section
        html_lines.append("<h2>Artifact Detection</h2>")
        html_lines.append("<table border='1'>")
        html_lines.append("<tr><th>Time (s)</th><th>Type</th><th>Severity</th><th>Confidence</th></tr>")
        for artifact_type, detections in artifacts.items():
            for time, type_name, severity, confidence in detections:
                html_lines.append(
                    f"<tr><td>{time:.2f}</td><td>{type_name}</td>"
                    f"<td>{severity:.2f}</td><td>{confidence:.2f}</td></tr>"
                )
        html_lines.append("</table>")
        
        # Coupling analysis section
        html_lines.append("<h2>Coupling Analysis</h2>")
        html_lines.append("<h3>Method Comparison</h3>")
        html_lines.append("<table border='1'>")
        html_lines.append("<tr><th>Method</th><th>Score</th><th>P-value</th></tr>")
        for method, results in coupling_stats['methods'].items():
            html_lines.append(
                f"<tr><td>{method.upper()}</td>"
                f"<td>{results['score']:.3f}</td>"
                f"<td>{results['p_value']:.3f}</td></tr>"
            )
        html_lines.append("</table>")
        
        # Consensus metrics
        html_lines.append("<h3>Consensus Metrics</h3>")
        html_lines.append(f"<p>Consensus Score: {coupling_stats['consensus_score']:.3f}</p>")
        html_lines.append(f"<p>Method Agreement: {coupling_stats['method_agreement']:.3f}</p>")
        
        # Visualizations
        html_lines.append("<h2>Visualizations</h2>")
        for plot_type, path in plot_paths.items():
            html_lines.append(f"<h3>{plot_type.upper()} Analysis</h3>")
            html_lines.append(f"<img src='{path}' style='max-width:800px'>")
        
        html_lines.append("</body></html>")
        
        # Save report
        report_path = output_dir / f"comprehensive_report_{channel_name}.html"
        with open(report_path, 'w') as f:
            f.write("\n".join(html_lines))
        
        return str(report_path) 