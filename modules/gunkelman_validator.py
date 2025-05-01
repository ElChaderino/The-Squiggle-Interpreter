"""
Advanced EEG artifact detection methods based on Jay Gunkelman's approaches.
Implements sophisticated cross-checks and validation traps for EMG and other artifacts.
"""

import numpy as np
import mne
from scipy import signal as sig
from scipy.stats import zscore, median_abs_deviation, entropy
from typing import Dict, List, Tuple, Optional, Union
import antropy
from mne.preprocessing import ICA
import pywt
from statsmodels.stats.diagnostic import acorr_ljungbox

class GunkelmanArtifactDetector:
    """
    Implements Gunkelman's sophisticated artifact detection methods with multiple
    cross-validation layers and "tricks and traps" for EMG detection.
    """
    
    def __init__(self):
        self.frequency_bands = {
            'Delta': (1, 4),
            'Theta': (4, 8),
            'Alpha': (8, 13),
            'SMR': (12, 15),
            'Beta': (13, 30),
            'High_Beta': (20, 30),
            'Gamma': (30, 45)
        }
        
        self.thresholds = {
            'z_score_diff': 1.5,
            'entropy_high': 0.8,
            'power_ratio': 2.5,
            'alpha_reactivity': 0.5,
            'emg_ratio': 0.7,
            'phase_continuity': 0.3
        }
    
    def compute_robust_stats(self, data: np.ndarray) -> Dict[str, float]:
        """
        Computes both standard and robust statistical measures for cross-validation.
        """
        results = {}
        
        # Standard statistics
        results['mean'] = np.mean(data)
        results['std'] = np.std(data)
        results['z_scores'] = zscore(data)
        
        # Robust statistics
        results['median'] = np.median(data)
        results['mad'] = median_abs_deviation(data)
        results['robust_z'] = (data - results['median']) / results['mad']
        
        # Cross-validation metrics
        results['z_score_diff'] = np.mean(
            np.abs(results['z_scores'] - results['robust_z'])
        )
        
        return results
    
    def validate_band_relationships(self, 
                                  data: np.ndarray,
                                  sfreq: float) -> Dict[str, Dict]:
        """
        Validates relationships between frequency bands using Gunkelman's criteria.
        """
        results = {'warnings': [], 'metrics': {}}
        
        def get_band_power(band: Tuple[float, float]) -> float:
            freqs, psd = sig.welch(data, fs=sfreq, nperseg=int(sfreq * 2))
            mask = (freqs >= band[0]) & (freqs <= band[1])
            return np.mean(psd[mask])
        
        # Compute band powers
        powers = {
            name: get_band_power(band) 
            for name, band in self.frequency_bands.items()
        }
        
        # Theta/Beta ratio check
        theta_beta = powers['Theta'] / powers['Beta']
        results['metrics']['theta_beta_ratio'] = theta_beta
        
        if theta_beta > self.thresholds['power_ratio']:
            results['warnings'].append(
                f"Suspicious Theta/Beta ratio: {theta_beta:.2f}"
            )
        
        # High-Beta vs Beta ratio (EMG trap)
        hbeta_beta = powers['High_Beta'] / powers['Beta']
        results['metrics']['hbeta_beta_ratio'] = hbeta_beta
        
        if hbeta_beta > self.thresholds['emg_ratio']:
            results['warnings'].append(
                "Possible EMG contamination detected in beta band"
            )
        
        return results
    
    def check_phase_relationships(self,
                                data: np.ndarray,
                                sfreq: float) -> Dict[str, Dict]:
        """
        Implements Gunkelman's phase relationship checks for artifact detection.
        """
        results = {'warnings': [], 'metrics': {}}
        
        def compute_phase_amp_coupling(signal: np.ndarray,
                                     phase_band: Tuple[float, float],
                                     amp_band: Tuple[float, float]) -> float:
            # Extract phase of lower frequency
            phase_filt = mne.filter.filter_data(
                signal[None, :], sfreq, 
                phase_band[0], phase_band[1], 
                verbose=False
            )[0]
            phase = np.angle(sig.hilbert(phase_filt))
            
            # Extract amplitude of higher frequency
            amp_filt = mne.filter.filter_data(
                signal[None, :], sfreq,
                amp_band[0], amp_band[1],
                verbose=False
            )[0]
            amplitude = np.abs(sig.hilbert(amp_filt))
            
            # Compute modulation index
            n_bins = 18
            phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
            mean_amp = np.zeros(n_bins)
            
            for i in range(n_bins):
                mask = (phase >= phase_bins[i]) & (phase < phase_bins[i+1])
                mean_amp[i] = np.mean(amplitude[mask]) if np.any(mask) else 0
            
            mean_amp /= np.sum(mean_amp)
            uniform = np.ones(n_bins) / n_bins
            
            return entropy(mean_amp, uniform)
        
        # Check PAC between key frequency pairs
        pac_pairs = [
            ('Theta', 'Gamma'),
            ('Alpha', 'Gamma'),
            ('Beta', 'Gamma')
        ]
        
        for low_band, high_band in pac_pairs:
            pac = compute_phase_amp_coupling(
                data,
                self.frequency_bands[low_band],
                self.frequency_bands[high_band]
            )
            
            key = f"PAC_{low_band}_{high_band}"
            results['metrics'][key] = pac
            
            if pac > self.thresholds['phase_continuity']:
                results['warnings'].append(
                    f"Suspicious {key} coupling detected: {pac:.2f}"
                )
        
        return results
    
    def validate_filter_effects(self,
                              data: np.ndarray,
                              sfreq: float) -> Dict[str, Dict]:
        """
        Implements Gunkelman's filter cross-checks and traps.
        """
        results = {'warnings': [], 'metrics': {}}
        
        # 1. Broad vs Narrow filtering trap
        broad_filt = mne.filter.filter_data(
            data[None, :], sfreq, 1, 70, verbose=False
        )[0]
        narrow_filt = mne.filter.filter_data(
            data[None, :], sfreq, 1, 30, verbose=False
        )[0]
        
        power_diff = np.mean(np.abs(broad_filt - narrow_filt))
        results['metrics']['filter_power_diff'] = power_diff
        
        if power_diff > 5.0:
            results['warnings'].append(
                "Significant high-frequency content detected"
            )
        
        # 2. Edge effect detection
        edge_var = np.var(narrow_filt[:100]) / np.var(narrow_filt[100:-100])
        results['metrics']['edge_variance_ratio'] = edge_var
        
        if edge_var > 2.0:
            results['warnings'].append("Filter edge effects detected")
        
        # 3. Notch filter trap
        notch_filt = mne.filter.notch_filter(
            data[None, :], sfreq, 
            np.array([50, 60, 120]), 
            verbose=False
        )[0]
        
        notch_diff = np.mean(np.abs(data - notch_filt))
        results['metrics']['notch_impact'] = notch_diff
        
        if notch_diff > 2.0:
            results['warnings'].append(
                "Significant line noise or grounding issues detected"
            )
        
        return results
    
    def check_temporal_consistency(self,
                                 data: np.ndarray,
                                 sfreq: float) -> Dict[str, Dict]:
        """
        Implements Gunkelman's temporal consistency checks and traps.
        """
        results = {'warnings': [], 'metrics': {}}
        
        # 1. Short vs Long epoch comparison
        def get_epoch_stats(epoch_len: int) -> Dict[str, float]:
            n_samples = int(epoch_len * sfreq)
            n_epochs = len(data) // n_samples
            
            powers = []
            for i in range(n_epochs):
                epoch = data[i*n_samples:(i+1)*n_samples]
                freqs, psd = sig.welch(epoch, fs=sfreq, nperseg=int(sfreq))
                beta_mask = (freqs >= 20) & (freqs <= 40)
                powers.append(np.mean(psd[beta_mask]))
            
            return {
                'mean': np.mean(powers),
                'std': np.std(powers),
                'var': np.var(powers)
            }
        
        short_stats = get_epoch_stats(2)  # 2-second epochs
        long_stats = get_epoch_stats(10)  # 10-second epochs
        
        variance_ratio = short_stats['var'] / long_stats['var']
        results['metrics']['epoch_variance_ratio'] = variance_ratio
        
        if variance_ratio > 3.0:
            results['warnings'].append(
                "High temporal variability suggests artifacts"
            )
        
        # 2. Spectral Edge Frequency (SEF) analysis
        freqs, psd = sig.welch(data, fs=sfreq, nperseg=int(sfreq * 2))
        cumsum = np.cumsum(psd)
        sef95_idx = np.where(cumsum >= 0.95 * cumsum[-1])[0][0]
        sef95 = freqs[sef95_idx]
        
        results['metrics']['sef95'] = sef95
        
        if sef95 > 35:  # Gunkelman's threshold
            results['warnings'].append(
                f"High SEF95 ({sef95:.1f} Hz) suggests EMG contamination"
            )
        
        return results
    
    def validate_signal(self,
                       data: np.ndarray,
                       sfreq: float,
                       eyes_closed: Optional[np.ndarray] = None) -> Dict:
        """
        Comprehensive signal validation using all Gunkelman methods.
        """
        all_results = {}
        
        # Run all validation checks
        all_results['robust_stats'] = self.compute_robust_stats(data)
        all_results['band_relationships'] = self.validate_band_relationships(
            data, sfreq
        )
        all_results['phase_checks'] = self.check_phase_relationships(
            data, sfreq
        )
        all_results['filter_checks'] = self.validate_filter_effects(
            data, sfreq
        )
        all_results['temporal_checks'] = self.check_temporal_consistency(
            data, sfreq
        )
        
        # If eyes-closed data available, check alpha reactivity
        if eyes_closed is not None:
            alpha_ec = np.mean(
                mne.filter.filter_data(
                    eyes_closed[None, :], sfreq, 8, 13, verbose=False
                )[0] ** 2
            )
            alpha_eo = np.mean(
                mne.filter.filter_data(
                    data[None, :], sfreq, 8, 13, verbose=False
                )[0] ** 2
            )
            
            reactivity = (alpha_ec - alpha_eo) / alpha_ec
            all_results['alpha_reactivity'] = {
                'value': reactivity,
                'warning': reactivity < self.thresholds['alpha_reactivity']
            }
        
        # Aggregate all warnings
        all_warnings = []
        for check_type, results in all_results.items():
            if isinstance(results, dict) and 'warnings' in results:
                all_warnings.extend(results['warnings'])
        
        return {
            'detailed_results': all_results,
            'warnings': all_warnings,
            'passed_all_checks': len(all_warnings) == 0
        }

def create_validation_report(validation_results: Dict) -> str:
    """
    Creates a detailed report from validation results.
    """
    report = ["=== EEG Validation Report ===\n"]
    
    if validation_results['passed_all_checks']:
        report.append("✓ Signal passed all validation checks\n")
    else:
        report.append("⚠ Validation warnings detected:\n")
        for warning in validation_results['warnings']:
            report.append(f"  - {warning}\n")
    
    report.append("\nDetailed Metrics:")
    for check_type, results in validation_results['detailed_results'].items():
        if isinstance(results, dict) and 'metrics' in results:
            report.append(f"\n{check_type}:")
            for metric, value in results['metrics'].items():
                report.append(f"  {metric}: {value:.3f}")
    
    return "\n".join(report) 