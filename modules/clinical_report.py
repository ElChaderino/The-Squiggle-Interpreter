#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
from typing import Dict, Tuple, List, Optional, Union, Any
import mne
from . import pdf_report_builder
import re
from . import pyramid_model
from modules.io_utils import load_eeg_data as load_data
import logging
from mne_connectivity import spectral_connectivity_epochs
from .pdf_report_builder import build_pdf_report
import itertools
import matplotlib as mpl
import os
import shutil

BANDS = {
    "Delta": (1, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "SMR": (12, 15),
    "Beta": (13, 30),
    "HighBeta": (20, 30)
}

# Band-specific instability thresholds based on physiological norms
INSTABILITY_THRESHOLDS = {
    "Delta": {"base": 25.0, "high": 40.0, "critical": 60.0},
    "Theta": {"base": 20.0, "high": 35.0, "critical": 50.0},
    "Alpha": {"base": 30.0, "high": 45.0, "critical": 65.0},
    "SMR": {"base": 15.0, "high": 25.0, "critical": 35.0},
    "Beta": {"base": 10.0, "high": 20.0, "critical": 30.0},
    "HighBeta": {"base": 5.0, "high": 10.0, "critical": 15.0},
    "Gamma": {"base": 3.0, "high": 6.0, "critical": 10.0},
    "LowAlpha": {"base": 35.0, "high": 50.0, "critical": 70.0},
    "HighAlpha": {"base": 25.0, "high": 40.0, "critical": 60.0},
    "LowBeta": {"base": 12.0, "high": 22.0, "critical": 32.0},
    "MidBeta": {"base": 8.0, "high": 18.0, "critical": 28.0},
    "ThetaAlpha": {"base": 40.0, "high": 60.0, "critical": 80.0}
}

def validate_band_instability(variance: float, band: str) -> Dict[str, Union[str, float, bool]]:
    """
    Validate band-specific instability based on physiological thresholds.
    
    Args:
        variance (float): Computed variance value
        band (str): Frequency band name
        
    Returns:
        Dict containing validation results:
            - status: "normal", "elevated", or "critical"
            - severity: float between 0 and 1
            - is_valid: boolean indicating if within acceptable range
            - message: description of the finding
    """
    thresholds = INSTABILITY_THRESHOLDS.get(band, {"base": 20.0, "high": 35.0, "critical": 50.0})
    
    if variance <= thresholds["base"]:
        return {
            "status": "normal",
            "severity": variance / thresholds["base"],
            "is_valid": True,
            "message": f"Normal {band} stability"
        }
    elif variance <= thresholds["high"]:
        severity = (variance - thresholds["base"]) / (thresholds["high"] - thresholds["base"])
        return {
            "status": "elevated",
            "severity": severity + 1,
            "is_valid": True,
            "message": f"Elevated {band} instability"
        }
    else:
        severity = min((variance - thresholds["high"]) / (thresholds["critical"] - thresholds["high"]), 1)
        return {
            "status": "critical",
            "severity": severity + 2,
            "is_valid": variance <= thresholds["critical"],
            "message": f"Critical {band} instability"
        }

def check_cross_band_relationships(instability_indices: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    """
    Check relationships between different frequency bands' instability indices.
    
    Args:
        instability_indices: Dictionary of band-specific instability indices
        
    Returns:
        List of findings about cross-band relationships
    """
    findings = []
    
    # Check alpha/beta relationship
    if ("Alpha" in instability_indices and "Beta" in instability_indices):
        alpha_inst = instability_indices["Alpha"]
        beta_inst = instability_indices["Beta"]
        ratio = alpha_inst / (beta_inst + 1e-6)  # Avoid division by zero
        if ratio > 4.0:
            findings.append({
                "type": "alpha_beta_ratio",
                "severity": min((ratio - 4.0) / 2.0, 1.0),
                "message": "Excessive alpha instability relative to beta"
            })
    
    # Check theta/alpha relationship
    if ("Theta" in instability_indices and "Alpha" in instability_indices):
        theta_inst = instability_indices["Theta"]
        alpha_inst = instability_indices["Alpha"]
        ratio = theta_inst / (alpha_inst + 1e-6)
        if ratio > 1.5:
            findings.append({
                "type": "theta_alpha_ratio",
                "severity": min((ratio - 1.5) / 1.0, 1.0),
                "message": "Elevated theta instability relative to alpha"
            })
    
    # Check beta/gamma relationship for EMG contamination
    if ("Beta" in instability_indices and "Gamma" in instability_indices):
        beta_inst = instability_indices["Beta"]
        gamma_inst = instability_indices["Gamma"]
        ratio = gamma_inst / (beta_inst + 1e-6)
        if ratio > 0.5:
            findings.append({
                "type": "possible_emg",
                "severity": min((ratio - 0.5) / 0.5, 1.0),
                "message": "Possible EMG contamination detected"
            })
    
    return findings

NORM_VALUES = {
    # Standard frequency bands
    "Delta": {"mean": 20.0, "sd": 10.0, "clinical_significance": "Sleep, deep relaxation, healing"},
    "Theta": {"mean": 15.0, "sd": 7.0, "clinical_significance": "Drowsiness, meditation, memory processing"},
    "Alpha": {"mean": 18.0, "sd": 6.0, "clinical_significance": "Relaxed wakefulness, cognitive readiness"},
    "SMR": {"mean": 6.0, "sd": 2.5, "clinical_significance": "Motor control, attention stability"},
    "Beta": {"mean": 5.0, "sd": 2.0, "clinical_significance": "Active thinking, focus, anxiety when excessive"},
    "HighBeta": {"mean": 3.5, "sd": 1.5, "clinical_significance": "Intensity, anxiety, overthinking"},
    "Gamma": {"mean": 2.0, "sd": 1.0, "clinical_significance": "Peak performance, cognitive integration"},
    
    # Specialized ratios and metrics
    "Alpha_Change": {"mean": 50.0, "sd": 15.0, "clinical_significance": "Cognitive flexibility, stress adaptation"},
    "Theta_Beta_Ratio": {"mean": 1.5, "sd": 0.5, "clinical_significance": "Attention regulation, ADHD indicator"},
    "Alpha_Peak_Freq": {"mean": 10.0, "sd": 1.0, "clinical_significance": "Cognitive processing speed"},
    "Delta_Power": {"mean": 20.0, "sd": 10.0, "clinical_significance": "Recovery, regeneration"},
    "SMR_Power": {"mean": 6.0, "sd": 2.5, "clinical_significance": "Sensorimotor integration"},
    
    # Regional asymmetry and coherence
    "Frontal_Asymmetry": {"mean": 0.0, "sd": 0.1, "clinical_significance": "Emotional regulation, depression risk"},
    "Total_Power": {"mean": 70.0, "sd": 20.0, "clinical_significance": "Overall cortical activation"},
    "Coherence_Alpha": {"mean": 0.7, "sd": 0.1, "clinical_significance": "Network integration, cognitive coordination"},
    
    # Site-specific norms
    "F3_Alpha": {"mean": 15.0, "sd": 5.0, "clinical_significance": "Left frontal executive function"},
    "F4_Alpha": {"mean": 15.0, "sd": 5.0, "clinical_significance": "Right frontal emotional processing"},
    "C3_SMR": {"mean": 5.0, "sd": 2.0, "clinical_significance": "Left sensorimotor control"},
    "C4_SMR": {"mean": 5.0, "sd": 2.0, "clinical_significance": "Right sensorimotor control"},
    "P3_Alpha": {"mean": 20.0, "sd": 7.0, "clinical_significance": "Left parietal processing"},
    "P4_Alpha": {"mean": 20.0, "sd": 7.0, "clinical_significance": "Right parietal processing"},
    "O1_Alpha": {"mean": 25.0, "sd": 8.0, "clinical_significance": "Left visual processing"},
    "O2_Alpha": {"mean": 25.0, "sd": 8.0, "clinical_significance": "Right visual processing"}
}

THRESHOLDS = {
    # Site-specific thresholds
    "CZ_Alpha_Percent": {
        "low": 30, 
        "high": 25,
        "clinical_implications": {
            "low": "Reduced cognitive integration, attention deficits",
            "high": "Overprocessing, cognitive inefficiency",
            "optimal": "Good cognitive integration and attention control"
        }
    },
    "Theta_Beta_Ratio": {
        "threshold": 2.2, 
        "severe": 3.0,
        "clinical_implications": {
            "below": "Possible anxiety or hyperarousal",
            "optimal": "Good attention regulation",
            "above": "Attention deficits, possible ADHD patterns",
            "severe": "Significant attention regulation issues"
        }
    },
    "Total_Amplitude": {
        "max": 60,
        "clinical_implications": {
            "high": "Possible cortical hyperexcitability or artifact",
            "optimal": "Normal cortical activation",
            "low": "Possible under-arousal or depression"
        }
    },
    "O1_Alpha_EC": {
        "low": 50, 
        "high": 150,
        "clinical_implications": {
            "low": "Poor visual processing, possible anxiety",
            "optimal": "Good visual processing and relaxation",
            "high": "Excessive relaxation or dissociation"
        }
    },
    "O1_Theta_Beta_Ratio": {
        "threshold": 1.8,
        "clinical_implications": {
            "above": "Visual attention deficits",
            "optimal": "Good visual attention",
            "below": "Visual hyperarousal"
        }
    },
    "F3F4_Theta_Beta_Ratio": {
        "threshold": 2.2,
        "clinical_implications": {
            "above": "Executive attention deficits",
            "optimal": "Good executive control",
            "below": "Possible anxiety or overthinking"
        }
    },
    "FZ_Delta": {
        "min": 9.0,
        "clinical_implications": {
            "low": "Poor recovery and integration",
            "optimal": "Good cognitive recovery",
            "high": "Excessive slowness or fatigue"
        }
    },
    "Alpha_Change": {
        "low": 30, 
        "high": 70,
        "clinical_implications": {
            "low": "Poor stress adaptation",
            "optimal": "Good cognitive flexibility",
            "high": "Excessive reactivity"
        }
    },
    "Alpha_Peak_Freq": {
        "low": 8.0, 
        "high": 12.0,
        "clinical_implications": {
            "low": "Cognitive slowing",
            "optimal": "Normal processing speed",
            "high": "Possible anxiety or hyperarousal"
        }
    },
    "Delta_Power": {
        "high": 30.0,
        "clinical_implications": {
            "high": "Excessive slowness or pathology",
            "optimal": "Normal recovery processes",
            "low": "Poor recovery or insomnia"
        }
    },
    "SMR_Power": {
        "low": 3.0,
        "clinical_implications": {
            "low": "Poor sensorimotor integration",
            "optimal": "Good motor control and attention",
            "high": "Possible tension or anxiety"
        }
    },
    "Frontal_Asymmetry": {
        "low": -0.2, 
        "high": 0.2,
        "clinical_implications": {
            "left": "Depression risk, withdrawal",
            "balanced": "Emotional stability",
            "right": "Anxiety risk, hyperarousal"
        }
    },
    "Total_Power": {
        "high": 100.0,
        "clinical_implications": {
            "high": "Cortical hyperarousal",
            "optimal": "Normal activation",
            "low": "Under-arousal or fatigue"
        }
    },
    "Coherence_Alpha": {
        "low": 0.5, 
        "high": 0.9,
        "clinical_implications": {
            "low": "Poor network integration",
            "optimal": "Good network coordination",
            "high": "Over-connectivity, reduced flexibility"
        }
    }
}

SITE_SPECIFIC_IMPLICATIONS = {
    "FP1": {
        "function": "Left prefrontal cortex - Executive function, emotional regulation",
        "clinical_relevance": {
            "alpha_high": "May indicate reduced executive function or emotional dysregulation",
            "theta_high": "Potential attention or impulse control issues",
            "beta_high": "Possible anxiety or overthinking patterns",
            "delta_high": "Consider sleep quality or cognitive fatigue"
        }
    },
    "FP2": {
        "function": "Right prefrontal cortex - Social cognition, behavioral inhibition",
        "clinical_relevance": {
            "alpha_high": "May affect social awareness or behavioral control",
            "theta_high": "Consider emotional regulation challenges",
            "beta_high": "Possible social anxiety or rumination",
            "delta_high": "Evaluate for mental fatigue or recovery needs"
        }
    },
    "F3": {
        "function": "Left frontal - Motor planning, verbal expression",
        "clinical_relevance": {
            "alpha_high": "Could impact verbal fluency or motor planning",
            "theta_high": "May indicate attention or language processing issues",
            "beta_high": "Consider motor tension or speech anxiety",
            "delta_high": "Evaluate for cognitive processing speed"
        }
    },
    "F4": {
        "function": "Right frontal - Emotional processing, risk assessment",
        "clinical_relevance": {
            "alpha_high": "May affect emotional awareness or decision-making",
            "theta_high": "Consider emotional processing difficulties",
            "beta_high": "Possible anxiety or emotional hypervigilance",
            "delta_high": "Evaluate for emotional processing speed"
        }
    },
    "F7": {
        "function": "Left inferior frontal - Language production",
        "clinical_relevance": {
            "alpha_high": "Could affect verbal expression or word finding",
            "theta_high": "Consider language processing challenges",
            "beta_high": "May indicate verbal anxiety or overthinking",
            "delta_high": "Evaluate for verbal processing speed"
        }
    },
    "F8": {
        "function": "Right inferior frontal - Emotional tone, facial recognition",
        "clinical_relevance": {
            "alpha_high": "May impact emotional recognition or expression",
            "theta_high": "Consider social processing difficulties",
            "beta_high": "Possible social anxiety or hypervigilance",
            "delta_high": "Evaluate for social processing speed"
        }
    },
    "T3": {
        "function": "Left temporal - Verbal memory, language comprehension",
        "clinical_relevance": {
            "alpha_high": "Could affect verbal memory or comprehension",
            "theta_high": "Consider auditory processing issues",
            "beta_high": "May indicate auditory hypervigilance",
            "delta_high": "Evaluate for auditory processing speed"
        }
    },
    "T4": {
        "function": "Right temporal - Non-verbal memory, emotional memory",
        "clinical_relevance": {
            "alpha_high": "May impact emotional memory or processing",
            "theta_high": "Consider emotional memory issues",
            "beta_high": "Possible emotional hyperarousal",
            "delta_high": "Evaluate for emotional processing"
        }
    },
    "T5": {
        "function": "Left posterior temporal - Visual and verbal integration",
        "clinical_relevance": {
            "alpha_high": "Could affect reading or visual-verbal integration",
            "theta_high": "Consider visual processing challenges",
            "beta_high": "May indicate visual hypervigilance",
            "delta_high": "Evaluate for visual processing speed"
        }
    },
    "T6": {
        "function": "Right posterior temporal - Visual memory, face recognition",
        "clinical_relevance": {
            "alpha_high": "May impact face recognition or visual memory",
            "theta_high": "Consider visual memory difficulties",
            "beta_high": "Possible visual anxiety or hypervigilance",
            "delta_high": "Evaluate for visual memory processing"
        }
    },
    "C3": {
        "function": "Left central - Sensorimotor integration, fine motor control",
        "clinical_relevance": {
            "alpha_high": "Could affect motor control or coordination",
            "theta_high": "Consider motor planning issues",
            "beta_high": "May indicate motor tension or anxiety",
            "delta_high": "Evaluate for motor processing speed"
        }
    },
    "C4": {
        "function": "Right central - Sensorimotor integration, gross motor control",
        "clinical_relevance": {
            "alpha_high": "May impact motor coordination or balance",
            "theta_high": "Consider motor integration challenges",
            "beta_high": "Possible motor tension or anxiety",
            "delta_high": "Evaluate for motor processing"
        }
    },
    "P3": {
        "function": "Left parietal - Sensory integration, spatial processing",
        "clinical_relevance": {
            "alpha_high": "Could affect spatial awareness or processing",
            "theta_high": "Consider sensory integration issues",
            "beta_high": "May indicate sensory hypervigilance",
            "delta_high": "Evaluate for sensory processing speed"
        }
    },
    "P4": {
        "function": "Right parietal - Spatial awareness, body awareness",
        "clinical_relevance": {
            "alpha_high": "May impact body awareness or spatial processing",
            "theta_high": "Consider spatial processing difficulties",
            "beta_high": "Possible spatial anxiety or hypervigilance",
            "delta_high": "Evaluate for spatial processing"
        }
    },
    "O1": {
        "function": "Left occipital - Visual processing, reading",
        "clinical_relevance": {
            "alpha_high": "Could affect visual processing or reading",
            "theta_high": "Consider visual attention issues",
            "beta_high": "May indicate visual stress or anxiety",
            "delta_high": "Evaluate for visual processing speed"
        }
    },
    "O2": {
        "function": "Right occipital - Visual integration, pattern recognition",
        "clinical_relevance": {
            "alpha_high": "May impact visual integration or patterns",
            "theta_high": "Consider visual processing challenges",
            "beta_high": "Possible visual anxiety or stress",
            "delta_high": "Evaluate for visual processing"
        }
    },
    "FZ": {
        "function": "Midline frontal - Attention, motor planning",
        "clinical_relevance": {
            "alpha_high": "Could affect attention or motor planning",
            "theta_high": "Consider attention regulation issues",
            "beta_high": "May indicate anxiety or tension",
            "delta_high": "Evaluate for cognitive processing"
        }
    },
    "CZ": {
        "function": "Midline central - Motor integration, attention",
        "clinical_relevance": {
            "alpha_high": "May impact motor integration or attention",
            "theta_high": "Consider attention or motor issues",
            "beta_high": "Possible motor tension or anxiety",
            "delta_high": "Evaluate for motor processing"
        }
    },
    "PZ": {
        "function": "Midline parietal - Sensory integration, attention",
        "clinical_relevance": {
            "alpha_high": "Could affect sensory integration or attention",
            "theta_high": "Consider sensory processing issues",
            "beta_high": "May indicate sensory hypervigilance",
            "delta_high": "Evaluate for sensory processing"
        }
    },
    "OZ": {
        "function": "Midline occipital - Visual integration",
        "clinical_relevance": {
            "alpha_high": "May impact visual integration or processing",
            "theta_high": "Consider visual attention issues",
            "beta_high": "Possible visual stress or anxiety",
            "delta_high": "Evaluate for visual processing"
        }
    }
}

DETAILED_SITES = {"CZ", "O1", "F3", "F4", "FZ", "PZ", "T3", "T4", "O2"}

def get_project_root() -> Path:
    current_dir = Path.cwd()
    expected_root = "The-Squiggle-Interpreter"
    if current_dir.name == expected_root or current_dir.name.startswith(expected_root):
        return current_dir
    for parent in current_dir.parents:
        if parent.name == expected_root or parent.name.startswith(expected_root):
            return parent
    sys.exit(f"Error: Could not find 'The-Squiggle-Interpreter' in the current directory or its parents.")

def find_edf_files(directory: Path) -> Dict[str, Path | None]:
    edf_files = list(directory.glob("*.edf"))
    files = {"EO": None, "EC": None}
    for file in edf_files:
        name = file.name.strip().lower()
        if "eo" in name:
            files["EO"] = file
        elif "ec" in name:
            files["EC"] = file
    return files

def clean_channel_name_dynamic(ch: str) -> str:
    ch_base = ch.upper()
    ch_base = re.sub(r'[-._]?(LE|RE|AVG|M1|M2|A1|A2|REF|CZREF|AV|AVERAGE|LINKED|MASTOID)$', '', ch_base)
    ch_base = re.sub(r'^(EEG\s*)?', '', ch_base)
    return ch_base

def compute_band_power(data: np.ndarray, sfreq: float, band: Tuple[float, float]) -> float:
    """
    Compute power in a specific frequency band.
    
    Args:
        data: EEG data array (samples or channels x samples)
        sfreq: Sampling frequency
        band: Tuple of (low_freq, high_freq)
        
    Returns:
        float: Band power
    """
    fmin, fmax = band
    
    # Ensure data is 1D or take mean if multiple channels
    if len(data.shape) > 1:
        if data.shape[0] > data.shape[1]:  # Likely transposed
            data = data.T
        data = np.mean(data, axis=0)
    
    # Compute power spectrum
    freqs, psd = welch(data, fs=sfreq, nperseg=min(int(sfreq * 2), len(data)), noverlap=int(sfreq))
    
    # Find frequencies within band
    band_mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band_mask):
        return 0.0
        
    # Calculate band power
    band_psd = psd[band_mask]
    freq_res = freqs[1] - freqs[0]
    power = np.sum(band_psd) * freq_res
    
    return float(power)

def compute_instability_index(raw_ec: mne.io.Raw, raw_eo: mne.io.Raw) -> Dict:
    """
    Compute instability indices for each frequency band and channel.
    
    Returns a dictionary with:
        - indices: {condition: {band_name: {channel: variance}}}
        - validations: {condition: {band_name: {channel: 'normal'|'elevated'|'critical'}}}
        - findings: {condition: {band_name: {'critical': [channels], 'elevated': [channels]}}}
    """
    # Define frequency bands and thresholds
    BANDS = {
        "Delta": (1, 4),
        "Theta": (4, 8),
        "Alpha": (8, 13),
        "SMR": (12, 15),
        "Beta": (13, 30),
        "HighBeta": (20, 30)
    }
    
    # Thresholds for instability validation
    THRESHOLDS = {
        'Delta': {'elevated': 0.3, 'critical': 0.5},
        'Theta': {'elevated': 0.25, 'critical': 0.4},
        'Alpha': {'elevated': 0.35, 'critical': 0.55},
        'SMR': {'elevated': 0.25, 'critical': 0.4},
        'Beta': {'elevated': 0.2, 'critical': 0.35},
        'HighBeta': {'elevated': 0.15, 'critical': 0.3}
    }
    
    results = {
        'indices': {'EC': {}, 'EO': {}},
        'validations': {'EC': {}, 'EO': {}},
        'findings': {'EC': {}, 'EO': {}}
    }
    
    for raw, condition in [(raw_ec, 'EC'), (raw_eo, 'EO')]:
        # Get data and compute band powers
        data = raw.get_data()
        sfreq = raw.info['sfreq']
        ch_names = raw.ch_names
        
        for band_name, (fmin, fmax) in BANDS.items():
            # Initialize structures for this band
            results['indices'][condition][band_name] = {}
            results['validations'][condition][band_name] = {}
            results['findings'][condition][band_name] = {'critical': [], 'elevated': []}
            
            # Compute band power and its variance over time
            band_data = mne.filter.filter_data(data.copy(), sfreq, fmin, fmax)
            band_power = np.square(band_data)
            
            # Compute variance for each channel
            for ch_idx, ch_name in enumerate(ch_names):
                variance = np.var(band_power[ch_idx])
                results['indices'][condition][band_name][ch_name] = variance
                
                # Validate the variance against thresholds
                if variance >= THRESHOLDS[band_name]['critical']:
                    results['validations'][condition][band_name][ch_name] = 'critical'
                    results['findings'][condition][band_name]['critical'].append(ch_name)
                elif variance >= THRESHOLDS[band_name]['elevated']:
                    results['validations'][condition][band_name][ch_name] = 'elevated'
                    results['findings'][condition][band_name]['elevated'].append(ch_name)
                else:
                    results['validations'][condition][band_name][ch_name] = 'normal'
    
    return results

def compute_coherence(raw: mne.io.Raw, ch1: str, ch2: str, band: Tuple[float, float], sfreq: float,
                      log_freqs: bool = False) -> float:
    fmin, fmax = band
    data = raw.get_data(picks=[ch1, ch2]) * 1e6
    duration = 2.0
    events = mne.make_fixed_length_events(raw, duration=duration)
    epochs = mne.Epochs(raw, events, tmin=0, tmax=duration, picks=[ch1, ch2], baseline=None, preload=True,
                        verbose=False)
    epochs_data = epochs.get_data() * 1e6
    csd_obj = mne.time_frequency.csd_array_fourier(
        epochs_data, sfreq=sfreq, fmin=fmin, fmax=fmax, n_fft=int(sfreq * 2)
    )
    freqs = csd_obj.frequencies
    coherence_values = []
    if log_freqs:
        print(f"\n=== Coherence Values for {ch1}-{ch2} in {band[0]}-{band[1]} Hz ===")
    for f in freqs:
        csd = csd_obj.get_data(frequency=f)
        psd1 = np.abs(csd[0, 0])
        psd2 = np.abs(csd[1, 1])
        csd12 = np.abs(csd[0, 1])
        coherence = csd12 ** 2 / (psd1 * psd2 + 1e-10)
        coherence_values.append(coherence)
        if log_freqs:
            print(f"Frequency {f:.1f} Hz: Coherence = {coherence:.3f}")
    if not coherence_values:
        return np.nan
    coherence = np.mean(coherence_values)
    if log_freqs:
        print(f"Average Coherence: {coherence:.3f}")
    return float(coherence)

def compute_all_band_powers(raw: mne.io.Raw) -> Dict[str, Dict[str, float]]:
    sfreq = raw.info["sfreq"]
    data = raw.get_data() * 1e6
    band_powers = {
        ch: {band: compute_band_power(data[i], sfreq, range_) for band, range_ in BANDS.items()}
        for i, ch in enumerate(raw.ch_names)
    }
    print(f"\n=== Band Powers for {raw.info['description']} ===")
    for ch, powers in band_powers.items():
        print(f"Channel {ch}:")
        for band, power in powers.items():
            print(f"  {band}: {power:.2f} µV²")
    return band_powers

def compute_alpha_peak_frequency(data: np.ndarray, sfreq: float, freq_range: Tuple[float, float]) -> float:
    fmin, fmax = freq_range
    freqs, psd = welch(data, fs=sfreq, nperseg=int(sfreq * 2), noverlap=int(sfreq))
    alpha_mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(alpha_mask):
        return np.nan
    alpha_freqs = freqs[alpha_mask]
    alpha_psd = psd[alpha_mask]
    return float(alpha_freqs[np.argmax(alpha_psd)])

def compute_frontal_asymmetry(bp_EO: Dict[str, Dict[str, float]], ch_left: str = "F3", ch_right: str = "F4") -> float:
    try:
        alpha_left = bp_EO[ch_left]["Alpha"]
        alpha_right = bp_EO[ch_right]["Alpha"]
        if alpha_left == 0 or alpha_right == 0:
            return np.nan
        return float(np.log(alpha_right / alpha_left))
    except KeyError:
        return np.nan

def compute_site_metrics(raw_eo: mne.io.Raw, raw_ec: mne.io.Raw, bp_EO: Dict, bp_EC: Dict) -> Tuple[Dict, Dict, Dict]:
    """
    Compute site-specific, global metrics, and connectivity matrices from EEG data.

    Args:
        raw_eo (mne.io.Raw): Raw EEG data for eyes-open condition.
        raw_ec (mne.io.Raw): Raw EEG data for eyes-closed condition.
        bp_EO (Dict): Band powers for eyes-open condition.
        bp_EC (Dict): Band powers for eyes-closed condition.

    Returns:
        site_metrics (Dict): Site-specific metrics (e.g., Alpha_Change, Theta_Beta_Ratio).
        global_metrics (Dict): Global metrics (e.g., Frontal_Asymmetry).
        connectivity (Dict): Connectivity matrices per band and condition.
    """
    site_metrics = {}
    global_metrics = {}
    connectivity = {'EO': {}, 'EC': {}}
    sfreq = raw_eo.info["sfreq"]

    # Log missing EC channels
    missing_ec_channels = set(bp_EO.keys()) - set(bp_EC.keys())
    if missing_ec_channels:
        with open("missing_channels_log.txt", "a") as f:
            for ch in missing_ec_channels:
                f.write(f"{ch} missing from EC\n")

    # Site-specific metrics
    channel_pairs = [("F3", "F4"), ("O1", "O2"), ("T3", "T4")]
    for ch in bp_EO.keys():
        if ch not in bp_EC:
            with open("missing_channels_log.txt", "a") as f:
                f.write(f"{ch} missing from EC\n")
            continue
        site_metrics[ch] = {}
        alpha_EO = bp_EO[ch].get("Alpha", 0)
        alpha_EC = bp_EC[ch].get("Alpha", 0)
        site_metrics[ch]["Alpha_Change"] = ((alpha_EC - alpha_EO) / alpha_EO) * 100 if alpha_EO != 0 else np.nan
        theta = bp_EO[ch].get("Theta", np.nan)
        beta = bp_EO[ch].get("Beta", np.nan)
        site_metrics[ch]["Theta_Beta_Ratio"] = theta / beta if beta != 0 else np.nan
        ec_sig = raw_ec.get_data(picks=[ch])[0] * 1e6
        site_metrics[ch]["Alpha_Peak_Freq"] = compute_alpha_peak_frequency(ec_sig, sfreq, BANDS["Alpha"])
        site_metrics[ch]["Delta_Power"] = bp_EO[ch].get("Delta", np.nan)
        site_metrics[ch]["SMR_Power"] = bp_EO[ch].get("SMR", np.nan)
        site_metrics[ch]["Total_Power_EO"] = sum(bp_EO[ch].values())
        site_metrics[ch]["Total_Power_EC"] = sum(bp_EC[ch].values())
        site_metrics[ch]["Total_Amplitude"] = alpha_EO
        for ch1, ch2 in channel_pairs:
            if ch == ch1 and ch2 in raw_eo.ch_names:
                coherence = compute_coherence(raw_eo, ch1, ch2, BANDS["Alpha"], sfreq)
                site_metrics[ch][f"Coherence_Alpha_{ch1}_{ch2}"] = coherence

    # Global metrics
    global_metrics["Frontal_Asymmetry"] = compute_frontal_asymmetry(bp_EO, "F3", "F4")

    # Connectivity matrices (coherence for all channel pairs)
    for condition, raw in [('EO', raw_eo), ('EC', raw_ec)]:
        epochs = mne.make_fixed_length_epochs(raw, duration=2.0, preload=True)
        for band_name, band_range in BANDS.items():
            conn = spectral_connectivity_epochs(
                epochs,
                method='coh',
                fmin=band_range[0],
                fmax=band_range[1],
                faverage=True,
                verbose=False
            )
            conn_matrix = conn.get_data(output='dense')[:, :, 0]  # n_channels x n_channels
            connectivity[condition][band_name] = conn_matrix

    return site_metrics, global_metrics, connectivity

def flag_abnormal(value: float, metric: str) -> str:
    if np.isnan(value):
        return "Not computed"
    if metric in NORM_VALUES:
        norm_mean = NORM_VALUES[metric]["mean"]
        norm_sd = NORM_VALUES[metric]["sd"]
        if value < norm_mean - 2 * norm_sd:
            return "Below normative range"
        if value > norm_mean + 2 * norm_sd:
            return "Above normative range"
    return "Within normative range"

def generic_interpretation(band: str, value: float) -> str:
    """Generate detailed clinical interpretation for a band power value."""
    # Get normative values for the band
    norm = NORM_VALUES.get(band, {})
    mean = norm.get('mean', 0)
    sd = norm.get('sd', 1)
    
    # Determine how many standard deviations from the mean
    z_score = (value - mean) / sd if sd != 0 else 0
    
    # Band-specific interpretations
    interpretations = {
        "Delta": {
            "high": "Elevated Delta suggests increased drowsiness, potential sleep pressure, or cognitive fatigue. "
                   "May indicate need for rest or recovery. Consider sleep quality and cognitive load.",
            "very_high": "Significantly elevated Delta may indicate pathological slow-wave activity, "
                        "possible neurological issues, or severe fatigue. Medical evaluation recommended.",
            "low": "Reduced Delta may indicate hyperarousal or difficulty achieving deep relaxation. "
                  "Consider relaxation techniques and stress management.",
            "normal": "Delta activity within normal range, suggesting appropriate levels of restfulness "
                     "and cortical deactivation."
        },
        "Theta": {
            "high": "Elevated Theta suggests reduced alertness, possible attention issues, or meditative state. "
                   "May indicate need for attention training or arousal regulation.",
            "very_high": "Significantly elevated Theta strongly suggests attention regulation issues. "
                        "Consider comprehensive attention assessment and targeted interventions.",
            "low": "Reduced Theta may indicate difficulty with memory processing or emotional integration. "
                  "Consider memory training and emotional regulation techniques.",
            "normal": "Theta activity within normal range, suggesting appropriate balance of attention "
                     "and memory processing."
        },
        "Alpha": {
            "high": "Elevated Alpha suggests increased cortical inhibition or relaxation. "
                   "May indicate reduced cognitive engagement or enhanced meditative state.",
            "very_high": "Significantly elevated Alpha may indicate excessive cortical inhibition "
                        "or cognitive disengagement. Consider arousal regulation strategies.",
            "low": "Reduced Alpha may indicate cortical hyperarousal or difficulty achieving relaxation. "
                  "Consider stress management and relaxation training.",
            "normal": "Alpha activity within normal range, suggesting appropriate balance of "
                     "relaxation and cognitive readiness."
        },
        "SMR": {
            "high": "Elevated SMR suggests enhanced sensorimotor integration and focused attention. "
                   "May indicate improved motor control and attention stability.",
            "very_high": "Significantly elevated SMR may indicate excessive motor inhibition. "
                        "Consider balance with other frequency training.",
            "low": "Reduced SMR may indicate poor sensorimotor integration or attention instability. "
                  "Consider SMR training for improved focus and motor control.",
            "normal": "SMR activity within normal range, suggesting appropriate sensorimotor "
                     "integration and attention control."
        },
        "Beta": {
            "high": "Elevated Beta suggests increased cognitive processing or potential anxiety. "
                   "May indicate mental effort or stress response.",
            "very_high": "Significantly elevated Beta may indicate excessive cortical arousal "
                        "or anxiety. Consider relaxation and stress management interventions.",
            "low": "Reduced Beta may indicate decreased cognitive processing or mental fatigue. "
                  "Consider cognitive enhancement strategies.",
            "normal": "Beta activity within normal range, suggesting appropriate levels of "
                     "cognitive processing and arousal."
        },
        "HighBeta": {
            "high": "Elevated High Beta suggests increased stress or cognitive intensity. "
                   "May indicate anxiety or overthinking patterns.",
            "very_high": "Significantly elevated High Beta strongly suggests excessive stress "
                        "or anxiety. Consider stress reduction interventions.",
            "low": "Reduced High Beta may indicate decreased cognitive intensity or stress. "
                  "Monitor for potential underarousal.",
            "normal": "High Beta activity within normal range, suggesting appropriate levels "
                     "of cognitive intensity."
        }
    }
    
    # Select interpretation based on z-score
    if band in interpretations:
        if z_score > 3:
            return interpretations[band]["very_high"]
        elif z_score > 2:
            return interpretations[band]["high"]
        elif z_score < -2:
            return interpretations[band]["low"]
        else:
            return interpretations[band]["normal"]
    
    return ""

class ClinicalDetectionSystem:
    def __init__(self):
        self.validator = GunkelmanValidator()
        self.condition_models = {}
        self.biomarker_weights = {
            'alpha_power': 0.3,
            'theta_beta_ratio': 0.25,
            'frontal_asymmetry': 0.2,
            'coherence': 0.15,
            'peak_frequency': 0.1
        }
        
    def compute_biomarker_score(self, metrics: Dict, condition: str) -> float:
        """
        Compute a weighted score for condition detection based on multiple biomarkers.
        """
        score = 0.0
        if 'alpha_power' in metrics:
            score += self.biomarker_weights['alpha_power'] * self._normalize_metric(
                metrics['alpha_power'], THRESHOLDS['Alpha_Power']
            )
        if 'theta_beta_ratio' in metrics:
            score += self.biomarker_weights['theta_beta_ratio'] * self._normalize_metric(
                metrics['theta_beta_ratio'], THRESHOLDS['Theta_Beta_Ratio']
            )
        if 'frontal_asymmetry' in metrics:
            score += self.biomarker_weights['frontal_asymmetry'] * self._normalize_metric(
                metrics['frontal_asymmetry'], THRESHOLDS['Frontal_Asymmetry']
            )
        return score

    def _normalize_metric(self, value: float, thresholds: Dict) -> float:
        """Normalize a metric value to [0,1] range based on thresholds."""
        if 'low' in thresholds and value < thresholds['low']:
            return 0.0
        if 'high' in thresholds and value > thresholds['high']:
            return 1.0
        if 'threshold' in thresholds:
            return min(1.0, value / thresholds['threshold'])
        return 0.5

    def detect_conditions(self, site_metrics: Dict, global_metrics: Dict) -> Dict[str, float]:
        """
        Detect potential clinical conditions and return confidence scores.
        
        Returns:
            Dict mapping condition names to confidence scores (0-1).
        """
        conditions = {}
        
        # ADHD Detection
        adhd_score = self.compute_biomarker_score({
            'theta_beta_ratio': site_metrics.get('CZ', {}).get('Theta_Beta_Ratio', 0),
            'alpha_power': site_metrics.get('CZ', {}).get('Alpha_Power', 0)
        }, 'ADHD')
        conditions['ADHD'] = adhd_score
        
        # Anxiety Detection
        anxiety_score = self.compute_biomarker_score({
            'alpha_power': site_metrics.get('F4', {}).get('Alpha_Power', 0),
            'frontal_asymmetry': global_metrics.get('Frontal_Asymmetry', 0)
        }, 'Anxiety')
        conditions['Anxiety'] = anxiety_score
        
        # Depression Detection
        depression_score = self.compute_biomarker_score({
            'frontal_asymmetry': global_metrics.get('Frontal_Asymmetry', 0),
            'alpha_power': site_metrics.get('F3', {}).get('Alpha_Power', 0)
        }, 'Depression')
        conditions['Depression'] = depression_score
        
        return conditions

    def get_condition_recommendations(self, conditions: Dict[str, float]) -> List[str]:
        """Generate recommendations based on detected conditions."""
        recommendations = []
        for condition, score in conditions.items():
            if score > 0.7:  # High confidence
                if condition == 'ADHD':
                    recommendations.extend([
                        "Strong indicators of ADHD-like patterns:",
                        "- Consider comprehensive ADHD evaluation",
                        "- Recommend SMR/Beta training protocol",
                        "- Monitor attention and impulse control"
                    ])
                elif condition == 'Anxiety':
                    recommendations.extend([
                        "Significant anxiety indicators present:",
                        "- Consider alpha/theta training protocol",
                        "- Recommend stress management techniques",
                        "- Monitor autonomic arousal levels"
                    ])
                elif condition == 'Depression':
                    recommendations.extend([
                        "Depression-like patterns detected:",
                        "- Consider left frontal alpha asymmetry training",
                        "- Recommend mood monitoring protocol",
                        "- Monitor for changes in motivation"
                    ])
            elif score > 0.4:  # Moderate confidence
                recommendations.append(f"Moderate indicators of {condition} (score: {score:.2f})")
                recommendations.append("- Recommend monitoring and follow-up assessment")
        return recommendations

    def analyze_with_validation(self, raw_ec: mne.io.Raw, raw_eo: mne.io.Raw) -> Dict:
        """
        Analyzes EEG data with Gunkelman's validation techniques before clinical interpretation.
        
        Parameters:
            raw_ec: Raw EEG data with eyes closed
            raw_eo: Raw EEG data with eyes open
            
        Returns:
            Dictionary containing both validation results and clinical interpretations
        """
        # First run Gunkelman's validation checks
        validation_results = {
            'emg_checks': {},
            'reference_checks': {},
            'alpha_checks': {}
        }
        
        # Check EMG contamination for each channel
        for ch in raw_ec.ch_names:
            data = raw_ec.get_data(picks=ch)
            validation_results['emg_checks'][ch] = self.validator.check_emg_contamination(
                data, raw_ec.info['sfreq']
            )
        
        # Validate reference channels
        ref_channels = [ch for ch in raw_ec.ch_names if ch in ['A1', 'A2', 'M1', 'M2']]
        if len(ref_channels) >= 2:
            validation_results['reference_checks'] = self.validator.validate_reference(
                raw_ec, ref_channels
            )
        
        # Check alpha reactivity in posterior channels
        posterior_channels = [ch for ch in raw_ec.ch_names if ch in ['O1', 'O2', 'P3', 'P4', 'P7', 'P8']]
        validation_results['alpha_checks'] = self.validator.check_alpha_reactivity(
            raw_ec, raw_eo, posterior_channels
        )
        
        # Only proceed with clinical analysis if validation passes
        validation_passed = self._evaluate_validation_results(validation_results)
        
        if validation_passed:
            clinical_results = self.analyze_clinical_metrics(raw_ec, raw_eo)
            return {
                'validation': validation_results,
                'clinical': clinical_results,
                'status': 'valid'
            }
        else:
            return {
                'validation': validation_results,
                'clinical': None,
                'status': 'invalid',
                'message': 'Data quality issues detected. Please review validation results.'
            }
    
    def _evaluate_validation_results(self, validation_results: Dict) -> bool:
        """
        Evaluates if the validation results pass Gunkelman's quality criteria.
        """
        # Check EMG contamination
        emg_contaminated_channels = [
            ch for ch, results in validation_results['emg_checks'].items()
            if results['is_contaminated']
        ]
        if len(emg_contaminated_channels) > len(validation_results['emg_checks']) * 0.2:  # More than 20% contaminated
            return False
            
        # Check reference validity
        if validation_results['reference_checks']:
            invalid_refs = [
                pair for pair, results in validation_results['reference_checks'].items()
                if not results['overall_valid']
            ]
            if invalid_refs:
                return False
                
        # Check alpha reactivity
        if validation_results['alpha_checks']:
            invalid_alpha = [
                ch for ch, results in validation_results['alpha_checks'].items()
                if not results['valid_reactivity']
            ]
            if len(invalid_alpha) > len(validation_results['alpha_checks']) * 0.5:  # More than 50% invalid
                return False
                
        return True

class GunkelmanValidator:
    """
    Implements Jay Gunkelman's rigorous cross-checking methodology for EEG validation.
    """
    def __init__(self):
        self.emg_thresholds = {
            'beta_jump_ratio': 2.5,  # Ratio of high beta to beta power
            'gamma_presence': 0.15,   # Threshold for gamma band presence
            'spectral_slope': -0.8    # Expected slope in log-log plot
        }
        
        self.reference_checks = {
            'amplitude_ratio': 0.1,    # Max ratio between references
            'phase_consistency': 0.8    # Required phase consistency
        }
        
        self.alpha_reactivity = {
            'min_suppression': 0.3,    # Minimum alpha suppression on eyes open
            'temporal_stability': 0.7   # Required stability over time
        }
        
    def check_emg_contamination(self, data: np.ndarray, sfreq: float) -> Dict[str, float]:
        """
        Gunkelman's EMG detection using spectral analysis.
        
        Args:
            data: EEG data array (channels x samples)
            sfreq: Sampling frequency
            
        Returns:
            Dictionary with EMG contamination metrics
        """
        # Compute power spectrum for each channel
        freqs, psd = welch(data, fs=sfreq, nperseg=min(int(sfreq * 2), data.shape[-1]))
        
        # Analyze high frequency range (30-100 Hz)
        high_freq_mask = (freqs >= 30) & (freqs <= min(100, sfreq/2))
        
        if not np.any(high_freq_mask):
            return {
                'emg_ratio': 0.0,
                'spectral_slope': 0.0,
                'is_contaminated': False
            }
        
        # Average across channels if multiple channels
        if len(psd.shape) > 1:
            psd = np.mean(psd, axis=0)
        
        # Compute spectral slope in high frequency range
        slope = np.polyfit(
            np.log10(freqs[high_freq_mask]), 
            np.log10(psd[high_freq_mask]), 
            1
        )[0]
        
        # Compute beta and gamma band power
        beta_mask = (freqs >= 13) & (freqs <= 30)
        gamma_mask = (freqs >= 30) & (freqs <= 45)
        
        beta_power = np.mean(psd[beta_mask]) if np.any(beta_mask) else 0
        gamma_power = np.mean(psd[gamma_mask]) if np.any(gamma_mask) else 0
        
        # Calculate EMG ratio
        emg_ratio = gamma_power / (beta_power + 1e-10)
        
        return {
            'emg_ratio': float(emg_ratio),
            'spectral_slope': float(slope),
            'is_contaminated': emg_ratio > self.emg_thresholds['gamma_presence'] or 
                             slope > self.emg_thresholds['spectral_slope']
        }
    
    def validate_reference(self, raw: mne.io.Raw, ref_channels: List[str]) -> Dict[str, bool]:
        """
        Gunkelman's reference validation tricks:
        1. Cross-correlation between references
        2. Phase stability check
        3. Amplitude ratio validation
        """
        results = {}
        for ref1, ref2 in itertools.combinations(ref_channels, 2):
            data1 = raw.get_data(picks=ref1)
            data2 = raw.get_data(picks=ref2)
            
            # Amplitude ratio check
            amp_ratio = np.max(np.abs(data1)) / np.max(np.abs(data2))
            amp_valid = self.reference_checks['amplitude_ratio'] < amp_ratio < (1/self.reference_checks['amplitude_ratio'])
            
            # Phase consistency
            hilbert1 = signal.hilbert(data1)
            hilbert2 = signal.hilbert(data2)
            phase_diff = np.abs(np.angle(hilbert1) - np.angle(hilbert2))
            phase_valid = np.mean(np.cos(phase_diff)) > self.reference_checks['phase_consistency']
            
            results[f"{ref1}-{ref2}"] = {
                'amplitude_valid': amp_valid,
                'phase_valid': phase_valid,
                'overall_valid': amp_valid and phase_valid
            }
        
        return results
    
    def check_alpha_reactivity(self, raw_ec: mne.io.Raw, raw_eo: mne.io.Raw, 
                             posterior_channels: List[str]) -> Dict[str, float]:
        """
        Gunkelman's alpha reactivity validation:
        1. True alpha suppression check
        2. Temporal stability analysis
        3. Spatial gradient verification
        """
        results = {}
        
        # Filter channels that exist in both conditions
        available_channels = [ch for ch in posterior_channels 
                            if ch in raw_ec.ch_names and ch in raw_eo.ch_names]
        
        if not available_channels:
            logging.warning("No posterior channels available for alpha reactivity check")
            return results
        
        for ch in available_channels:
            try:
                # Get data for both conditions
                ec_data = raw_ec.get_data(picks=[ch])[0]
                eo_data = raw_eo.get_data(picks=[ch])[0]
                
                # Compute alpha power
                ec_alpha = compute_band_power(ec_data, raw_ec.info['sfreq'], (8, 13))
                eo_alpha = compute_band_power(eo_data, raw_eo.info['sfreq'], (8, 13))
                
                # Compute suppression ratio
                if ec_alpha > 0:
                    suppression = (ec_alpha - eo_alpha) / ec_alpha
                else:
                    suppression = 0
                
                # Check temporal stability
                stability = self._compute_alpha_stability(ec_data, raw_ec.info['sfreq'])
                
                results[ch] = {
                    'suppression': float(suppression),
                    'stability': float(stability),
                    'valid_reactivity': (suppression > self.alpha_reactivity['min_suppression'] and 
                                       stability > self.alpha_reactivity['temporal_stability'])
                }
                
            except Exception as e:
                logging.warning(f"Could not compute alpha reactivity for channel {ch}: {str(e)}")
                continue
        
        return results
    
    def _compute_alpha_stability(self, data: np.ndarray, sfreq: float, 
                               window_size: float = 4.0) -> float:
        """Helper method to compute alpha stability over time"""
        from scipy.stats import variation
        
        # Split data into windows
        window_samples = int(window_size * sfreq)
        n_windows = len(data) // window_samples
        windows = data[:n_windows * window_samples].reshape(n_windows, -1)
        
        # Compute alpha power for each window
        alpha_powers = []
        for window in windows:
            alpha_power = compute_band_power(window, sfreq, (8, 13))
            alpha_powers.append(alpha_power)
        
        # Compute coefficient of variation (lower is more stable)
        stability = 1 - variation(alpha_powers)
        return max(0, min(1, stability))  # Normalize to [0,1]

def enhance_clinical_interpretation(site_metrics: Dict, global_metrics: Dict) -> Dict:
    """
    Enhanced clinical interpretation function that provides more detailed analysis
    of brain activity patterns and their clinical significance.
    
    Args:
        site_metrics (Dict): Site-specific metrics
        global_metrics (Dict): Global brain metrics
    
    Returns:
        Dict: Detailed clinical interpretation
    """
    interpretation = {
        "overall_patterns": [],
        "site_specific_findings": {},
        "network_analysis": [],
        "recommendations": []
    }
    
    # Analyze overall patterns
    if global_metrics.get("total_power", 0) > THRESHOLDS["Total_Power"]["high"]:
        interpretation["overall_patterns"].append(
            "High overall cortical activation suggesting possible hyperarousal state"
        )
    
    # Analyze frontal asymmetry
    asymmetry = global_metrics.get("frontal_asymmetry", 0)
    if asymmetry < THRESHOLDS["Frontal_Asymmetry"]["low"]:
        interpretation["overall_patterns"].append(
            "Left frontal hypoactivation pattern suggesting possible mood regulation issues"
        )
    elif asymmetry > THRESHOLDS["Frontal_Asymmetry"]["high"]:
        interpretation["overall_patterns"].append(
            "Right frontal hyperactivation pattern suggesting possible anxiety"
        )
    
    # Analyze site-specific patterns
    for site, metrics in site_metrics.items():
        if site in SITE_SPECIFIC_IMPLICATIONS:
            site_info = SITE_SPECIFIC_IMPLICATIONS[site]
            findings = []
            
            # Check each frequency band
            for band in BANDS:
                band_power = metrics.get(f"{band.lower()}_power", 0)
                if band_power > NORM_VALUES.get(band, {}).get("mean", 0) + 2 * NORM_VALUES.get(band, {}).get("sd", 1):
                    findings.append(f"Elevated {band} activity: {site_info['clinical_relevance'].get(f'{band.lower()}_high', 'Possible dysfunction')}")
            
            if findings:
                interpretation["site_specific_findings"][site] = findings
    
    # Generate recommendations
    if interpretation["overall_patterns"] or interpretation["site_specific_findings"]:
        interpretation["recommendations"].extend([
            "Consider quantitative analysis for more detailed assessment",
            "Monitor changes over time with follow-up measurements",
            "Consider targeted interventions based on specific patterns"
        ])
    
    return interpretation

def analyze_clinical_metrics(site_metrics, global_metrics, bp_EO, bp_EC):
    """Analyze clinical metrics and generate findings with confidence scores."""
    analysis = {
        'findings': [],
        'confidence_scores': {
            'alpha_reactivity': 0.8,
            'spectral_ratio': 0.7,
            'coherence': 0.75,
            'connectivity': 0.7
        }
    }
    
    # Analyze alpha reactivity
    for site in site_metrics:
        if 'Alpha' in site_metrics[site]:
            eo_alpha = bp_EO.get(site, {}).get('Alpha', 0)
            ec_alpha = bp_EC.get(site, {}).get('Alpha', 0)
            if ec_alpha > 0:
                reactivity = (ec_alpha - eo_alpha) / ec_alpha
                if abs(reactivity) > 0.2:
                    analysis['findings'].append({
                        'category': 'Alpha Reactivity',
                        'description': f'Significant alpha reactivity at {site}',
                        'confidence': 0.85,
                        'implication': 'Normal alpha modulation'
                    })
    
    # Analyze spectral ratios
    for site in site_metrics:
        if all(band in site_metrics[site] for band in ['Theta', 'Beta']):
            theta_beta_ratio = site_metrics[site]['Theta'] / site_metrics[site]['Beta']
            if theta_beta_ratio > 2.5:
                analysis['findings'].append({
                    'category': 'Spectral Ratio',
                    'description': f'Elevated theta/beta ratio at {site}',
                    'confidence': 0.75,
                    'implication': 'Possible attention-related pattern'
                })
    
    # Analyze global patterns
    if global_metrics:
        total_power = sum(power for power in global_metrics.values())
        if total_power > 0:
            for band, power in global_metrics.items():
                rel_power = power / total_power
                if band == 'Alpha' and rel_power < 0.1:
                    analysis['findings'].append({
                        'category': 'Global Pattern',
                        'description': 'Low global alpha power',
                        'confidence': 0.8,
                        'implication': 'Possible arousal/anxiety pattern'
                    })
    
    return analysis

def interpret_metrics(site_metrics, global_metrics, bp_EO, bp_EC):
    """Interpret clinical metrics and generate clinical implications."""
    clinical_analysis = analyze_clinical_metrics(site_metrics, global_metrics, bp_EO, bp_EC)
    
    interpretations = []
    
    # Process confidence scores
    for condition, score in clinical_analysis['confidence_scores'].items():
        interpretation = f"[{condition.replace('_', ' ').title()}] "
        interpretation += f"Confidence: {score:.2f} - "
        interpretation += get_confidence_implication(condition, score)
        interpretations.append(interpretation)
    
    # Add other interpretations
    for finding in clinical_analysis.get('findings', []):
        interpretation = f"[{finding['category']}] "
        interpretation += f"{finding['description']} "
        interpretation += f"(Confidence: {finding.get('confidence', 0.7):.2f}) - "
        interpretation += finding.get('implication', '')
        interpretations.append(interpretation)
    
    return interpretations

def get_confidence_implication(metric_type, score):
    """Generate implications based on confidence scores."""
    if score >= 0.8:
        return f"High confidence in {metric_type.replace('_', ' ')} measurements"
    elif score >= 0.6:
        return f"Moderate confidence in {metric_type.replace('_', ' ')} measurements"
    else:
        return f"Low confidence in {metric_type.replace('_', ' ')} measurements - interpret with caution"

def save_site_metrics(site_metrics: Dict, global_metrics: Dict, output_path: Path) -> None:
    rows = []
    for ch, met in site_metrics.items():
        row = {"Channel": ch}
        row.update({band: power for band, power in met.items()})
        rows.append(row)
    global_row = {"Channel": "Global"}
    global_row.update(global_metrics)
    rows.append(global_row)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Clinical metrics saved to: {output_path}")

def remove_overlapping_channels(info: mne.Info, tol: float = 0.05) -> Tuple[mne.Info, List[int]]:
    ch_names = info["ch_names"]
    pos = np.array([info["chs"][i]["loc"][:3] for i in range(len(ch_names))])
    unique_idx = []
    for i in range(len(ch_names)):
        if not any(np.linalg.norm(pos[i] - pos[j]) < tol for j in unique_idx):
            unique_idx.append(i)
    info_clean = mne.pick_info(info, sel=unique_idx)
    return info_clean, unique_idx

CRITICAL_SITES = {"F3", "CZ", "O1", "FZ", "PZ", "F4", "O2", "T3", "T4"}

def plot_topomap_abs_rel(
    abs_vals: np.ndarray,
    rel_vals: np.ndarray,
    raw: mne.io.Raw,
    band_name: str,
    cond_name: str,
    output_path: Path,
    instability_vals: np.ndarray = None,
    instability_validations: Dict = None,
    site_metrics: dict = None,
    cmap: str = "plasma",
    cmap_instability: str = "magma",
    figsize: Tuple[float, float] = (15, 4),
    dpi: int = 300,
    normalize: bool = True,
    show_ch_names: bool = True
) -> None:
    """
    Plot absolute and relative power topomaps side-by-side with enhanced instability visualization.
    """
    logger_cr = logging.getLogger(__name__)
    logger_cr.debug(f"plot_topomap_abs_rel called for {cond_name} - {band_name}. Instability provided: {instability_vals is not None}")

    # Create output filename
    output_file = output_path / f"topomap_{cond_name}_{band_name}.png"
    output_path.mkdir(parents=True, exist_ok=True)

    info = raw.info
    montage = mne.channels.make_standard_montage("standard_1020")
    raw_temp = mne.io.RawArray(np.zeros((len(info["ch_names"]), 1)), info)
    raw_temp.set_montage(montage, match_case=False, on_missing="warn")

    default_pos = {
        "T7": [-0.071, 0.008, 0.038],
        "T8": [0.071, 0.008, 0.038],
        "P7": [-0.055, -0.055, 0.038],
        "P8": [0.055, -0.055, 0.038],
        "F3": [-0.038, 0.071, 0.038],
        "F4": [0.038, 0.071, 0.038],
        "CZ": [0.0, 0.0, 0.087],
        "O1": [-0.038, -0.071, 0.038],
        "O2": [0.038, -0.071, 0.038],
        "FZ": [0.0, 0.071, 0.055],
        "PZ": [0.0, -0.055, 0.071],
    }
    for ch in raw_temp.info["chs"]:
        ch_name = ch["ch_name"].upper()
        if np.all(ch["loc"][:3] == 0) and ch_name in default_pos:
            ch["loc"][:3] = default_pos[ch_name]
            print(f"Injected position for {ch_name}: {ch['loc'][:3]}")
    info = raw_temp.info
    print(f"\n=== Channels and Positions for Topomap ({band_name}, {cond_name}) ===")
    for ch in info["chs"]:
        print(f"{ch['ch_name']}: {ch['loc'][:3]}")
    pos = np.array([ch["loc"][:3] for ch in info["chs"] if ch.get("loc") is not None])
    if len(pos) == 0 or not np.any(pos):
        print(f"Warning: No valid sensor positions for topomap ({cond_name}, {band_name}). Skipping.")
        with open("missing_channels_log.txt", "a") as log:
            log.write(f"\n=== Topomap Failure ({cond_name}, {band_name}) ===\n")
            log.write("No valid sensor positions. Skipping.\n")
        return
    info_clean, sel_idx = remove_overlapping_channels(info)
    if not sel_idx:
        print(f"Warning: No unique channel positions after overlap removal ({cond_name}, {band_name}). Using all channels.")
        info_clean = info
        sel_idx = list(range(len(info["ch_names"])))
    CRITICAL_SITES = {"F3", "CZ", "O1", "FZ", "PZ", "F4", "O2", "T7", "T8", "P7", "P8"}
    cleaned_ch_names = set(info_clean["ch_names"])
    missing_critical = CRITICAL_SITES - cleaned_ch_names
    if missing_critical:
        print(f"[!] Warning: Missing electrodes {missing_critical} for {band_name} ({cond_name})")
        with open("missing_channels_log.txt", "a") as log:
            log.write(f"\n=== Topomap Warning ({cond_name}, {band_name}) ===\n")
            log.write(f"Missing electrodes: {missing_critical}\n")
    final_pos = np.array([ch["loc"][:3] for ch in info_clean["chs"] if ch.get("loc") is not None])
    if len(final_pos) == 0 or not np.any(final_pos):
        print(f"[!] Skipping topomap: info_clean has no valid sensor positions for {band_name} ({cond_name})")
        with open("missing_channels_log.txt", "a") as log:
            log.write(f"\n=== Topomap Skipped ({cond_name}, {band_name}) ===\n")
            log.write("No valid sensor positions in cleaned info. Skipping.\n")
        return
    abs_vals_subset = abs_vals[sel_idx]
    rel_vals_subset = rel_vals[sel_idx]
    if instability_vals is not None:
        instability_subset = instability_vals[sel_idx]
    else:
        instability_subset = None
    if normalize:
        abs_vals_subset = (abs_vals_subset - abs_vals_subset.min()) / (abs_vals_subset.max() - abs_vals_subset.min() + 1e-10)
        rel_vals_subset = (rel_vals_subset - rel_vals_subset.min()) / (rel_vals_subset.max() - rel_vals_subset.min() + 1e-10)
        if instability_subset is not None:
            instability_subset = (instability_subset - instability_subset.min()) / (instability_subset.max() - instability_subset.min() + 1e-10)

    # Enhanced color settings for better visibility
    plt.style.use('dark_background')
    
    n_subplots = 3 if instability_subset is not None else 2
    fig, axes = plt.subplots(1, n_subplots, figsize=figsize, facecolor="#000000")
    fig.patch.set_facecolor("#000000")

    # Enhanced absolute power plot
    ax_abs = axes[0]
    ax_abs.set_facecolor("#000000")
    vmin_abs, vmax_abs = np.percentile(abs_vals_subset, [5, 95])
    im_abs, _ = mne.viz.plot_topomap(
        abs_vals_subset, info_clean, axes=ax_abs, show=False, cmap=cmap,
        vlim=(vmin_abs, vmax_abs), names=info_clean["ch_names"] if show_ch_names else None,
        outlines="head", contours=6
    )
    ax_abs.set_title(f"{band_name} Abs Power\n({cond_name})", color="#00ffff", fontsize=12, pad=20)
    cbar_abs = plt.colorbar(im_abs, ax=ax_abs, orientation="horizontal", fraction=0.05, pad=0.08)
    cbar_abs.set_label("µV²" if not normalize else "Normalized", color="#00ffff")
    cbar_abs.ax.tick_params(colors="#00ffff")

    # Enhanced relative power plot
    ax_rel = axes[1]
    ax_rel.set_facecolor("#000000")
    vmin_rel, vmax_rel = np.percentile(rel_vals_subset, [5, 95])
    im_rel, _ = mne.viz.plot_topomap(
        rel_vals_subset, info_clean, axes=ax_rel, show=False, cmap=cmap,
        vlim=(vmin_rel, vmax_rel), names=info_clean["ch_names"] if show_ch_names else None,
        outlines="head", contours=6
    )
    ax_rel.set_title(f"{band_name} Rel Power\n({cond_name})", color="#ff00ff", fontsize=12, pad=20)
    cbar_rel = plt.colorbar(im_rel, ax=ax_rel, orientation="horizontal", fraction=0.05, pad=0.08)
    cbar_rel.set_label("%" if not normalize else "Normalized", color="#ff00ff")
    cbar_rel.ax.tick_params(colors="#ff00ff")

    if instability_subset is not None:
        ax_inst = axes[2]
        ax_inst.set_facecolor("#000000")
        
        # Create custom colormap for instability levels
        colors = [(0, 1, 0, 1),      # Green for normal
                 (1, 1, 0, 1),      # Yellow for elevated
                 (1, 0, 0, 1)]      # Red for critical
        n_bins = 100
        custom_cmap = mpl.colors.LinearSegmentedColormap.from_list("custom", colors, N=n_bins)
        
        # Plot instability topomap
        vmin_inst, vmax_inst = np.percentile(instability_subset, [5, 95])
        im_inst, _ = mne.viz.plot_topomap(
            instability_subset, info_clean, axes=ax_inst, show=False, cmap=custom_cmap,
            vlim=(vmin_inst, vmax_inst), names=info_clean["ch_names"] if show_ch_names else None,
            outlines="head", contours=6
        )
        ax_inst.set_title(f"{band_name} Instability\n({cond_name})", color="#ffff00", fontsize=12, pad=20)
        
        # Add custom colorbar with threshold markers
        cbar_inst = plt.colorbar(im_inst, ax=ax_inst, orientation="horizontal", fraction=0.05, pad=0.08)
        cbar_inst.set_label("Variance (µV²)" if not normalize else "Normalized", color="#ffff00")
        cbar_inst.ax.tick_params(colors="#ffff00")
        
        # Add threshold markers if validations are provided
        if instability_validations and band_name in INSTABILITY_THRESHOLDS:
            thresholds = INSTABILITY_THRESHOLDS[band_name]
            if not normalize:
                # Add threshold lines to colorbar
                cbar_inst.ax.axvline(thresholds["base"], color='g', linestyle='--', alpha=0.7)
                cbar_inst.ax.axvline(thresholds["high"], color='y', linestyle='--', alpha=0.7)
                cbar_inst.ax.axvline(thresholds["critical"], color='r', linestyle='--', alpha=0.7)
            
            # Add markers for channels with elevated/critical instability
            if isinstance(instability_validations, dict):  # Check if it's a dictionary
                for ch_idx, ch in enumerate(info_clean["ch_names"]):
                    if ch in instability_validations:
                        validation = instability_validations[ch]
                        if isinstance(validation, dict) and "status" in validation:  # Check if validation is a dict with status
                            if validation["status"] in ["elevated", "critical"]:
                                pos = mne.channels.layout._find_topomap_coords(info_clean, picks=[ch])[0]
                                marker_color = "#ffff00" if validation["status"] == "elevated" else "#ff0000"
                                ax_inst.plot(pos[0], pos[1], marker='*', markersize=15,
                                           color=marker_color, markeredgecolor='#ffffff',
                                           markeredgewidth=1)

    # Add findings summary if available
    if isinstance(instability_validations, dict):  # Check if it's a dictionary
        critical_channels = []
        elevated_channels = []
        for ch, validation in instability_validations.items():
            if isinstance(validation, dict) and "status" in validation:  # Check if validation is a dict with status
                if validation["status"] == "critical":
                    critical_channels.append(ch)
                elif validation["status"] == "elevated":
                    elevated_channels.append(ch)
        
        if critical_channels or elevated_channels:
            summary_text = []
            if critical_channels:
                summary_text.append(f"Critical: {', '.join(critical_channels)}")
            if elevated_channels:
                summary_text.append(f"Elevated: {', '.join(elevated_channels)}")
            fig.text(0.02, 0.02, "\n".join(summary_text), color='white', fontsize=8,
                    bbox=dict(facecolor='black', alpha=0.8, edgecolor='white'))

    fig.suptitle(f"{band_name} Activity Analysis ({cond_name})",
                 color="#ffffff", fontsize=14, y=1.05)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    fig.savefig(output_file, dpi=dpi, bbox_inches="tight", facecolor="#000000",
                edgecolor='none', pad_inches=0.1)
    plt.close(fig)
    print(f"Saved topomap for {band_name} ({cond_name}) to {output_file}")

def plot_band_psd_overlay(
    sig1: np.ndarray,
    sig2: np.ndarray,
    sfreq: float,
    band: Tuple[float, float],
    ch_name: str,
    band_name: str,
    colors: Tuple[str, str] = ("#00ffff", "#ff00ff"),  # Changed to cyan and magenta
    figsize: Tuple[float, float] = (8, 4),
) -> plt.Figure:
    """Enhanced PSD overlay plot with candy-colored gradients"""
    fmin, fmax = band
    freqs, psd1 = welch(sig1, fs=sfreq, nperseg=int(sfreq * 2), noverlap=int(sfreq))
    _, psd2 = welch(sig2, fs=sfreq, nperseg=int(sfreq * 2), noverlap=int(sfreq))
    mask = (freqs >= fmin) & (freqs <= fmax)
    freqs_band = freqs[mask]
    psd1_band = psd1[mask]
    psd2_band = psd2[mask]
    
    fig, ax = plt.subplots(figsize=figsize, facecolor="#000000")
    
    # Create gradient lines
    for i in range(len(freqs_band)-1):
        ax.fill_between(freqs_band[i:i+2], psd1_band[i:i+2], alpha=0.1, color=colors[0])
        ax.fill_between(freqs_band[i:i+2], psd2_band[i:i+2], alpha=0.1, color=colors[1])
    
    # Main lines with glow effect
    ax.plot(freqs_band, psd1_band, color=colors[0], label="EO", linewidth=2.5)
    ax.plot(freqs_band, psd2_band, color=colors[1], label="EC", linewidth=2.5)
    
    # Add subtle glow effect
    for line in ax.lines:
        line.set_path_effects([plt.matplotlib.patheffects.withSimplePatchShadow(
            offset=(0, 0), shadow_rgbFace=line.get_color(), alpha=0.3, rho=0.3
        )])
    
    ax.set_title(f"{ch_name} {band_name} PSD Overlay", color="#ffffff", fontsize=12)
    ax.set_xlabel("Frequency (Hz)", color="#ffffff", fontsize=10)
    ax.set_ylabel("Power (µV²/Hz)", color="#ffffff", fontsize=10)
    ax.legend(facecolor="#000000", edgecolor="#ffffff", labelcolor="#ffffff")
    ax.set_facecolor("#000000")
    ax.tick_params(colors="#ffffff")
    ax.grid(True, color="#333333", linestyle="--", alpha=0.3)
    
    fig.tight_layout()
    return fig

def plot_band_waveform_overlay(
    sig1: np.ndarray,
    sig2: np.ndarray,
    sfreq: float,
    band: Tuple[float, float],
    ch_name: str,
    band_name: str,
    colors: Tuple[str, str] = ("cyan", "magenta"),
    epoch_length: float = 10.0,
    figsize: Tuple[float, float] = (10, 4),
) -> plt.Figure:
    fmin, fmax = band
    sig1_filt = mne.filter.filter_data(sig1, sfreq, fmin, fmax, verbose=False)
    sig2_filt = mne.filter.filter_data(sig2, sfreq, fmin, fmax, verbose=False)
    n_samples = int(epoch_length * sfreq)
    if n_samples > len(sig1_filt):
        n_samples = len(sig1_filt)
    if n_samples > len(sig2_filt):
        n_samples = len(sig2_filt)
    t = np.arange(n_samples) / sfreq
    sig1_epoch = sig1_filt[:n_samples]
    sig2_epoch = sig2_filt[:n_samples]
    fig, ax = plt.subplots(figsize=figsize, facecolor="black")
    ax.plot(t, sig1_epoch, color=colors[0], label="EO", alpha=0.7)
    ax.plot(t, sig2_epoch, color=colors[1], label="EC", alpha=0.7)
    ax.set_title(f"{ch_name} {band_name} Waveform Overlay", color="white", fontsize=12)
    ax.set_xlabel("Time (s)", color="white")
    ax.set_ylabel("Amplitude (µV)", color="white")
    ax.legend(facecolor="black", edgecolor="white", labelcolor="white")
    ax.set_facecolor("black")
    ax.tick_params(colors="white")
    ax.grid(True, color="gray", linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig

def generate_full_site_plots(raw_eo: mne.io.Raw, raw_ec: mne.io.Raw, output_dir: Path, site_metrics: dict = None, bands: dict = None, erps: dict = None) -> None:
    """
    Generate site-specific plots (PSD, waveform, difference, ERP) with customized styles.

    Args:
        raw_eo (mne.io.Raw): Raw EEG data for eyes-open condition.
        raw_ec (mne.io.Raw): Raw EEG data for eyes-closed condition.
        output_dir (Path): Directory to save plots.
        site_metrics (dict, optional): Site-specific metrics for abnormality flagging.
        bands (dict, optional): Dictionary of frequency bands.
        erps (dict, optional): Dictionary of ERP data for each condition.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    channels = raw_eo.ch_names
    sfreq = raw_eo.info["sfreq"]
    if bands is None:
        bands = {
            'Delta': (1, 4),
            'Theta': (4, 8),
            'Alpha': (8, 13),
            'SMR': (13, 15),
            'Beta': (15, 30),
            'HighBeta': (30, 40)
        }

    for ch in channels:
        if ch not in raw_ec.ch_names or ch not in raw_eo.ch_names:
            with open("missing_channels_log.txt", "a") as f:
                f.write(f"{ch} missing in EO or EC for plotting\n")
            continue
        ch_folder = output_dir / ch
        psd_folder = ch_folder / "PSD_Overlay"
        wave_folder = ch_folder / "Waveform_Overlay"
        diff_folder = ch_folder / "Difference"
        erp_folder = ch_folder / "ERP_Overlay"
        psd_folder.mkdir(parents=True, exist_ok=True)
        wave_folder.mkdir(parents=True, exist_ok=True)
        diff_folder.mkdir(parents=True, exist_ok=True)
        erp_folder.mkdir(parents=True, exist_ok=True)

        # Check for abnormal metrics
        is_abnormal = False
        abnormal_bands = []
        if site_metrics and ch in site_metrics:
            for band in bands:
                power = site_metrics[ch].get(f"{band}_Power", np.nan)
                if power and flag_abnormal(power, band) != "Within normative range":
                    is_abnormal = True
                    abnormal_bands.append(f"{band}: {generic_interpretation(band, power)}")

        eo_sig = raw_eo.get_data(picks=[ch])[0] * 1e6
        ec_sig = raw_ec.get_data(picks=[ch])[0] * 1e6
        for band_name, band_range in bands.items():
            # PSD Overlay
            fig_psd = plot_band_psd_overlay(
                eo_sig, ec_sig, sfreq, band_range, ch, band_name,
                colors=("blue", "green") if not is_abnormal else ("red", "red")
            )
            fig_psd.set_facecolor("black")
            ax_psd = fig_psd.gca()
            ax_psd.grid(True, linestyle='--', alpha=0.7)
            ax_psd.set_facecolor("black")
            ax_psd.set_title(f"{ch} {band_name} PSD Overlay", color="white", fontsize=10)
            ax_psd.set_xlabel("Frequency (Hz)", color="white", fontsize=8)
            ax_psd.set_ylabel("Power (µV²/Hz)", color="white", fontsize=8)
            ax_psd.tick_params(colors="white")
            if is_abnormal:
                ax_psd.annotate(
                    "Abnormal: " + "; ".join(abnormal_bands[:2]),  # Limit for brevity
                    xy=(0.05, 0.95), xycoords='axes fraction', fontsize=8, color="red",
                    bbox=dict(facecolor="black", alpha=0.8)
                )
            psd_path = psd_folder / f"{ch}_PSD_{band_name}.png"
            fig_psd.savefig(psd_path, facecolor="black", dpi=300)
            plt.close(fig_psd)

            # Waveform Overlay
            fig_wave = plot_band_waveform_overlay(
                eo_sig, ec_sig, sfreq, band_range, ch, band_name,
                colors=("blue", "green") if not is_abnormal else ("red", "red"),
                epoch_length=10
            )
            fig_wave.set_facecolor("black")
            ax_wave = fig_wave.gca()
            ax_wave.grid(True, linestyle='--', alpha=0.7)
            ax_wave.set_facecolor("black")
            ax_wave.set_title(f"{ch} {band_name} Waveform Overlay", color="white", fontsize=10)
            ax_wave.set_xlabel("Time (s)", color="white", fontsize=8)
            ax_wave.set_ylabel("Amplitude (µV)", color="white", fontsize=8)
            ax_wave.tick_params(colors="white")
            if is_abnormal:
                ax_wave.annotate(
                    "Abnormal: " + "; ".join(abnormal_bands[:2]),
                    xy=(0.05, 0.95), xycoords='axes fraction', fontsize=8, color="red",
                    bbox=dict(facecolor="black", alpha=0.8)
                )
            wave_path = wave_folder / f"{ch}_Waveform_{band_name}.png"
            fig_wave.savefig(wave_path, facecolor="black", dpi=300)
            plt.close(fig_wave)

            # Difference Plot
            power_eo = compute_band_power(eo_sig, sfreq, band_range)
            power_ec = compute_band_power(ec_sig, sfreq, band_range)
            fig_diff, ax = plt.subplots(figsize=(4, 4), facecolor="black")
            ax.bar(["EO", "EC"], [power_eo, power_ec], color=["blue", "green"] if not is_abnormal else ["red", "red"])
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_facecolor("black")
            ax.set_title(f"{ch} {band_name} Power Difference", color="white", fontsize=10)
            ax.set_ylabel("Power (µV²)", color="white", fontsize=8)
            ax.tick_params(colors="white")
            if is_abnormal:
                ax.annotate(
                    "Abnormal: " + "; ".join(abnormal_bands[:2]),
                    xy=(0.05, 0.95), xycoords='axes fraction', fontsize=8, color="red",
                    bbox=dict(facecolor="black", alpha=0.8)
                )
            fig_diff.tight_layout()
            diff_path = diff_folder / f"{ch}_Difference_{band_name}.png"
            fig_diff.savefig(diff_path, facecolor="black", dpi=300)
            plt.close(fig_diff)

        # ERP Overlay
        if erps:
            erp_data = erps.get(ch)
            if erp_data is not None:
                fig_erp, ax = plt.subplots(figsize=(8, 4), facecolor="black")
                ax.plot(np.linspace(-0.1, 0.5, len(erp_data)), erp_data)
                ax.set_title(f"ERP {ch}", color="white", fontsize=10)
                ax.set_xlabel("Time (s)", color="white", fontsize=8)
                ax.set_ylabel("Amplitude (µV)", color="white", fontsize=8)
                ax.tick_params(colors="white")
                if is_abnormal:
                    ax.annotate(
                        "Abnormal: " + "; ".join(abnormal_bands[:2]),
                        xy=(0.05, 0.95), xycoords='axes fraction', fontsize=8, color="red",
                        bbox=dict(facecolor="black", alpha=0.8)
                    )
                erp_path = erp_folder / f"erp_{ch}.png"
                fig_erp.savefig(erp_path, facecolor="black", dpi=300)
                plt.close(fig_erp)

def generate_clinical_interpretations(bp_EO: Dict, bp_EC: Dict, site_metrics: Dict, global_metrics: Dict) -> str:
    """Generate detailed clinical interpretations for each site."""
    output = []
    
    # Process each site
    for site in sorted(bp_EO.keys()):
        output.append(f"\n=== Site: {site} ===")
        
        # Add site function if available
        if site in SITE_SPECIFIC_IMPLICATIONS:
            output.append(f"Function: {SITE_SPECIFIC_IMPLICATIONS[site]['function']}")
        
        # Band Powers
        output.append("\nFrequency Band Powers (µV²):")
        for band in ['Delta', 'Theta', 'Alpha', 'SMR', 'Beta', 'HighBeta']:
            if band in bp_EO[site]:
                power_eo = bp_EO[site][band]
                power_ec = bp_EC[site].get(band, 0) if site in bp_EC else 0
                status_eo = flag_abnormal(power_eo, band)
                status_ec = flag_abnormal(power_ec, band)
                
                output.append(f"\n  {band}:")
                output.append(f"    Eyes Open: {power_eo:.2f} ({status_eo})")
                output.append(f"    Eyes Closed: {power_ec:.2f} ({status_ec})")
                
                # Add band-specific interpretations
                if status_eo != "Within normative range" or status_ec != "Within normative range":
                    output.append(f"    Clinical Significance: {generic_interpretation(band, max(power_eo, power_ec))}")
                    
                    # Add site-specific implications if available
                    if site in SITE_SPECIFIC_IMPLICATIONS:
                        if power_eo > NORM_VALUES.get(band, {}).get('mean', 0) + 2 * NORM_VALUES.get(band, {}).get('sd', 1):
                            if f"{band.lower()}_high" in SITE_SPECIFIC_IMPLICATIONS[site]['clinical_relevance']:
                                output.append(f"    Site-Specific Implication: {SITE_SPECIFIC_IMPLICATIONS[site]['clinical_relevance'][f'{band.lower()}_high']}")
        
        # Site-specific metrics if available
        if site in site_metrics:
            metrics = site_metrics[site]
            output.append("\nSite-Specific Metrics:")
            
            # Alpha Change
            if 'Alpha_Change' in metrics:
                value = metrics['Alpha_Change']
                output.append(f"\n  Alpha Change (EO→EC): {value:.1f}% ({flag_abnormal(value, 'Alpha_Change')})")
                if abs(value) > THRESHOLDS.get('Alpha_Change', {}).get('high', 70):
                    if site in SITE_SPECIFIC_IMPLICATIONS:
                        output.append(f"    Clinical Significance: {SITE_SPECIFIC_IMPLICATIONS[site]['clinical_relevance'].get('alpha_high', '')}")
                        output.append("    Recommendations:")
                        output.append("      - Balance with Beta training")
                        output.append("      - Assess for cognitive disengagement")
                        output.append("      - Consider targeted interventions")
            
            # Theta/Beta Ratio
            if 'Theta_Beta_Ratio' in metrics:
                ratio = metrics['Theta_Beta_Ratio']
                output.append(f"\n  Theta/Beta Ratio: {ratio:.2f} ({flag_abnormal(ratio, 'Theta_Beta_Ratio')})")
                if ratio > THRESHOLDS['Theta_Beta_Ratio']['severe']:
                    output.append("    Clinical Significance: Severe - Indicative of attention regulation issues")
                    output.append("    Recommendations:")
                    output.append("      - Inhibit Theta (4-8 Hz)")
                    output.append("      - Enhance Beta (15-27 Hz)")
                    output.append("      - Consider behavioral interventions")
                    output.append("      - Monitor for co-occurring conditions")
            
            # Alpha Peak Frequency
            if 'Alpha_Peak_Freq' in metrics:
                freq = metrics['Alpha_Peak_Freq']
                output.append(f"\n  Alpha Peak Frequency: {freq:.2f} Hz ({flag_abnormal(freq, 'Alpha_Peak_Freq')})")
                if freq < NORM_VALUES['Alpha_Peak_Freq']['mean'] - NORM_VALUES['Alpha_Peak_Freq']['sd']:
                    output.append("    Clinical Significance: Reduced cognitive processing speed")
                    output.append("    Recommendations:")
                    output.append("      - Consider cognitive enhancement strategies")
                    output.append("      - Monitor for fatigue or medication effects")
            
            # Total Power
            if 'Total_Power_EO' in metrics and 'Total_Power_EC' in metrics:
                eo_power = metrics['Total_Power_EO']
                ec_power = metrics['Total_Power_EC']
                output.append(f"\n  Total Power:")
                output.append(f"    Eyes Open: {eo_power:.2f} µV² ({flag_abnormal(eo_power, 'Total_Power')})")
                output.append(f"    Eyes Closed: {ec_power:.2f} µV² ({flag_abnormal(ec_power, 'Total_Power')})")
                if max(eo_power, ec_power) > THRESHOLDS['Total_Amplitude']['max']:
                    output.append("    Clinical Significance: High overall power - possible hyperarousal")
                    output.append("    Recommendations:")
                    output.append("      - Assess for artifact contamination")
                    output.append("      - Consider arousal regulation strategies")
                    output.append("      - Monitor for developmental factors")
            
            # Coherence
            coherence_found = False
            for key in metrics:
                if key.startswith('Coherence_Alpha_'):
                    if not coherence_found:
                        output.append("\n  Coherence Measures:")
                        coherence_found = True
                    value = metrics[key]
                    pair = key.replace('Coherence_Alpha_', '')
                    output.append(f"    Alpha Coherence with {pair}: {value:.2f} ({flag_abnormal(value, 'Coherence_Alpha')})")
                    if value > NORM_VALUES['Coherence_Alpha']['mean'] + 2 * NORM_VALUES['Coherence_Alpha']['sd']:
                        output.append("      Clinical Significance: High coherence - possible over-synchronization")
                        output.append("      Recommendations:")
                        output.append("        - Assess for network flexibility")
                        output.append("        - Consider coherence training")
                        output.append("        - Monitor for cognitive adaptability")
    
    # Global Metrics
    if global_metrics:
        output.append("\n=== Global Metrics ===")
        if 'frontal_asymmetry' in global_metrics:
            asym = global_metrics['frontal_asymmetry']
            output.append(f"\nFrontal Asymmetry (F4/F3 Alpha, EO): {asym:.2f} ({flag_abnormal(asym, 'Frontal_Asymmetry')})")
            if abs(asym) > NORM_VALUES['Frontal_Asymmetry']['mean'] + 2 * NORM_VALUES['Frontal_Asymmetry']['sd']:
                output.append("  Clinical Significance: Significant frontal asymmetry")
                output.append("  Recommendations:")
                output.append("    - Assess for mood disorders")
                output.append("    - Consider asymmetry training")
                output.append("    - Monitor for emotional regulation")
    
    return "\n".join(output)

def _generate_clinical_report(raw_ec: mne.io.Raw, raw_eo: mne.io.Raw, output_dir: str, channels: List[str] = None) -> Dict:
    """
    Generate a clinical report with band powers, instability indices, and plots.
    Also generates clinical interpretation and pyramid model reports.
    
    Args:
        raw_ec: Raw EEG data for eyes closed condition
        raw_eo: Raw EEG data for eyes open condition
        output_dir: Directory to save output files
        channels: List of channels to analyze. If None, uses all channels.
        
    Returns:
        Dictionary containing report data and file paths
    """
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    reports_dir = output_dir / "reports"
    topomap_dir = reports_dir / "topomaps"
    
    # Create necessary directories
    for dir_path in [plots_dir, reports_dir, topomap_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Compute band powers for both conditions
    bp_EO = compute_all_band_powers(raw_eo)
    bp_EC = compute_all_band_powers(raw_ec)
    
    # Compute instability indices
    try:
        instability_results = compute_instability_index(raw_ec, raw_eo)
    except Exception as e:
        logging.error(f"Error computing instability indices: {e}")
        instability_results = {'EC': {}, 'EO': {}}
    
    # Compute site metrics and global metrics
    site_metrics, global_metrics, findings = compute_site_metrics(raw_eo, raw_ec, bp_EO, bp_EC)
    
    # Generate clinical interpretations
    clinical_interpretations = generate_clinical_interpretations(bp_EO, bp_EC, site_metrics, global_metrics)
    interpretations_path = reports_dir / "clinical_interpretations.txt"
    with open(interpretations_path, 'w', encoding='utf-8') as f:
        f.write(clinical_interpretations)
    
    # Generate plots for each condition and band
    plot_files = {}
    bands = {
        'Delta': (1, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'HighBeta': (20, 30)
    }
    
    for condition, raw in [('EC', raw_ec), ('EO', raw_eo)]:
        for band_name, band_range in bands.items():
            try:
                # Get band power data
                band_powers = {}
                if condition == 'EC':
                    band_powers = {ch: bp_EC[ch][band_name] for ch in raw.ch_names if ch in bp_EC}
                else:
                    band_powers = {ch: bp_EO[ch][band_name] for ch in raw.ch_names if ch in bp_EO}
                
                if not band_powers:
                    logging.warning(f"No valid channels found for {condition} {band_name}")
                    continue
                
                # Convert to arrays for plotting
                ch_names = list(band_powers.keys())
                abs_vals = np.array([band_powers[ch] for ch in ch_names])
                total_powers = np.array([
                    sum(bp_EC[ch].values()) if condition == 'EC' else sum(bp_EO[ch].values())
                    for ch in ch_names
                ])
                
                # Avoid division by zero
                total_powers[total_powers == 0] = 1
                rel_vals = abs_vals / total_powers
                
                # Get instability values for this condition
                instability_vals = None
                instability_validations = None
                if condition in instability_results and band_name in instability_results[condition]:
                    instability_vals = np.array([
                        instability_results[condition][band_name].get(ch, 0)
                        for ch in ch_names
                    ])
                    instability_validations = {
                        ch: instability_results[condition].get(band_name, {}).get(ch, {})
                        for ch in ch_names
                    }
                
                # Generate and save plot
                plot_topomap_abs_rel(
                    abs_vals=abs_vals,
                    rel_vals=rel_vals,
                    raw=raw,
                    band_name=band_name,
                    cond_name=condition,
                    output_path=plots_dir,
                    instability_vals=instability_vals,
                    instability_validations=instability_validations,
                    site_metrics=site_metrics
                )
                
                # Copy plot to topomaps directory for PDF report
                plot_file = plots_dir / f"topomap_{condition}_{band_name}.png"
                if plot_file.exists():
                    topomap_dest = topomap_dir / plot_file.name
                    shutil.copy2(plot_file, topomap_dest)
                    plot_files[f"{condition}_{band_name}"] = str(topomap_dest)
                
            except Exception as e:
                logging.error(f"Error generating plot for {condition} {band_name}: {str(e)}")
                continue
    
    # Generate main clinical interpretation report
    clinical_report_path = reports_dir / "clinical_interpretation.txt"
    site_report_path = reports_dir / "site_by_site_interpretation.txt"
    
    # Generate main clinical interpretation report
    with open(clinical_report_path, 'w', encoding='utf-8') as f:
        f.write("CLINICAL INTERPRETATION REPORT\n")
        f.write("============================\n\n")
        
        # Data Quality and Validation
        f.write("1. DATA QUALITY VALIDATION\n")
        f.write("-------------------------\n")
        validator = GunkelmanValidator()
        ec_validation = validator.check_emg_contamination(raw_ec.get_data(), raw_ec.info['sfreq'])
        eo_validation = validator.check_emg_contamination(raw_eo.get_data(), raw_eo.info['sfreq'])
        f.write(f"EMG Contamination (EC): {ec_validation['emg_ratio']:.2f}\n")
        f.write(f"EMG Contamination (EO): {eo_validation['emg_ratio']:.2f}\n\n")
        
        # Reference Validation - only if reference channels exist
        ref_channels = [ch for ch in raw_ec.ch_names if ch in ['A1', 'A2', 'M1', 'M2']]
        if len(ref_channels) >= 2:
            ref_validation = validator.validate_reference(raw_ec, ref_channels)
            f.write("Reference Validation Results:\n")
            for key, value in ref_validation.items():
                f.write(f"- {key}: {'Valid' if value else 'Invalid'}\n")
        else:
            f.write("Reference Validation: No standard reference channels (A1/A2/M1/M2) found\n")
        f.write("\n")
        
        # Alpha Reactivity
        alpha_reactivity = validator.check_alpha_reactivity(raw_ec, raw_eo, ['O1', 'O2', 'P3', 'P4'])
        f.write("2. ALPHA REACTIVITY ANALYSIS\n")
        f.write("--------------------------\n")
        for ch, results in alpha_reactivity.items():
            if isinstance(results, dict):
                suppression = results.get('suppression', 0.0)
                stability = results.get('stability', 0.0)
                valid = results.get('valid_reactivity', False)
                f.write(f"{ch}:\n")
                f.write(f"  Suppression: {suppression:.2f}\n")
                f.write(f"  Stability: {stability:.2f}\n")
                f.write(f"  Valid: {valid}\n")
            else:
                f.write(f"{ch}: No valid reactivity data\n")
        f.write("\n")
        
        # Clinical Findings
        f.write("3. CLINICAL FINDINGS\n")
        f.write("------------------\n")
        for finding in findings:
            f.write(f"- {finding}\n")
        f.write("\n")
        
        # Site-Specific Findings
        f.write("4. SITE-SPECIFIC FINDINGS\n")
        f.write("-----------------------\n")
        for ch, metrics in site_metrics.items():
            f.write(f"\n{ch}:\n")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    f.write(f"  {metric}: {value:.2f}\n")
                else:
                    f.write(f"  {metric}: {value}\n")
        f.write("\n")
        
        # Global Metrics
        f.write("5. GLOBAL METRICS\n")
        f.write("---------------\n")
        for metric, value in global_metrics.items():
            if isinstance(value, (int, float)):
                f.write(f"{metric}: {value:.2f}\n")
            else:
                f.write(f"{metric}: {value}\n")
        f.write("\n")
        
        # Connectivity Analysis
        f.write("6. CONNECTIVITY ANALYSIS\n")
        f.write("----------------------\n")
        coherence_pairs = [('F3', 'F4'), ('P3', 'P4')]
        for ch1, ch2 in coherence_pairs:
            if ch1 in raw_ec.ch_names and ch2 in raw_ec.ch_names:
                for band_name, band_range in bands.items():
                    try:
                        coh = compute_coherence(raw_ec, ch1, ch2, band_range, raw_ec.info['sfreq'])
                        f.write(f"{ch1}-{ch2} {band_name} Coherence: {coh:.3f}\n")
                    except Exception as e:
                        logging.warning(f"Could not compute coherence for {ch1}-{ch2} {band_name}: {e}")
        f.write("\n")
    
    # Generate detailed site-by-site interpretation report
    with open(site_report_path, 'w', encoding='utf-8') as f:
        f.write("SITE-BY-SITE CLINICAL INTERPRETATION REPORT\n")
        f.write("========================================\n\n")
        
        for ch in raw_ec.ch_names:
            if ch not in SITE_SPECIFIC_IMPLICATIONS:
                continue
                
            f.write(f"\n{ch} - {SITE_SPECIFIC_IMPLICATIONS[ch]['function']}\n")
            f.write("=" * (len(ch) + len(SITE_SPECIFIC_IMPLICATIONS[ch]['function']) + 3) + "\n\n")
            
            # Write site functions
            f.write("Clinical Relevance:\n")
            for key, value in SITE_SPECIFIC_IMPLICATIONS[ch]['clinical_relevance'].items():
                f.write(f"- {value}\n")
            f.write("\n")
            
            # Analyze band powers
            f.write("Band Power Analysis:\n")
            for band in ['Delta', 'Theta', 'Alpha', 'Beta', 'HighBeta']:
                ec_power = bp_EC.get(ch, {}).get(band, 0)
                eo_power = bp_EO.get(ch, {}).get(band, 0)
                
                if ec_power > 0 or eo_power > 0:
                    f.write(f"\n{band} Band:\n")
                    f.write(f"- EC Power: {ec_power:.2f} µV²\n")
                    f.write(f"- EO Power: {eo_power:.2f} µV²\n")
                    
                    # Add band-specific clinical implications
                    if band.lower() + "_high" in SITE_SPECIFIC_IMPLICATIONS[ch]['clinical_relevance']:
                        if eo_power > ec_power * 1.5:  # Significant elevation
                            f.write(f"Clinical Relevance: {SITE_SPECIFIC_IMPLICATIONS[ch]['clinical_relevance'][band.lower() + '_high']}\n")
            
            # Add instability analysis
            f.write("\nInstability Analysis:\n")
            for band in ['Delta', 'Theta', 'Alpha', 'Beta', 'HighBeta']:
                if ch in instability_results.get('EC', {}).get(band, {}):
                    inst_val = instability_results['EC'][band][ch]
                    validation = validate_band_instability(inst_val, band)
                    f.write(f"{band}: {validation['message']} (Value: {inst_val:.2f})\n")
            
            # Add site-specific metrics if available
            if ch in site_metrics:
                f.write("\nSite-Specific Metrics:\n")
                for metric, value in site_metrics[ch].items():
                    if isinstance(value, (int, float)):
                        f.write(f"- {metric}: {value:.2f}\n")
                        # Add clinical interpretation based on thresholds
                        if metric in THRESHOLDS:
                            thresh = THRESHOLDS[metric]
                            if 'clinical_implications' in thresh:
                                if value > thresh.get('high', float('inf')):
                                    f.write(f"  Implication: {thresh['clinical_implications']['high']}\n")
                                elif value < thresh.get('low', float('-inf')):
                                    f.write(f"  Implication: {thresh['clinical_implications']['low']}\n")
                                else:
                                    f.write(f"  Implication: {thresh['clinical_implications']['optimal']}\n")
            
            # Add recommendations based on findings
            f.write("\nRecommendations:\n")
            if ch in DETAILED_SITES:
                # Add specific recommendations based on the site's role and findings
                if ch in ['F3', 'F4']:
                    f.write("- Consider frontal alpha asymmetry training if indicated\n")
                elif ch in ['C3', 'C4']:
                    f.write("- Consider SMR training for sensorimotor integration\n")
                elif ch in ['P3', 'P4']:
                    f.write("- Monitor parietal alpha for cognitive processing\n")
                elif ch in ['O1', 'O2']:
                    f.write("- Assess visual processing and alpha blocking\n")
            
            f.write("\n" + "-"*50 + "\n")
    
    return {
        'band_powers_EO': bp_EO,
        'band_powers_EC': bp_EC,
        'instability_results': instability_results,
        'site_metrics': site_metrics,
        'global_metrics': global_metrics,
        'findings': findings,
        'plot_files': plot_files,
        'clinical_report': str(clinical_report_path),
        'site_report': str(site_report_path),
        'interpretations_report': str(interpretations_path)
    }

def main():
    parser = argparse.ArgumentParser(description="Generate clinical EEG report from EDF files.")
    parser.add_argument("data_dir", type=str, help="Directory containing EO and EC EDF files.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for reports and plots.")
    parser.add_argument("--csd", action="store_true", help="Apply surface Laplacian (CSD) transform.")
    parser.add_argument("--filter", type=float, nargs=2, default=None, help="Low and high cutoff frequencies for bandpass filter (Hz).")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        sys.exit(f"Data directory {data_dir} does not exist.")
    project_root = get_project_root()
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / "outputs" / "report"
    files = find_edf_files(data_dir)
    if not files["EO"] or not files["EC"]:
        sys.exit("Could not find both EO and EC EDF files in the specified directory.")
    raw_eo = load_data(files["EO"])
    raw_ec = load_data(files["EC"])
    for raw in (raw_eo, raw_ec):
        raw.set_eeg_reference("average", projection=True)
        if args.filter:
            raw.filter(l_freq=args.filter[0], h_freq=args.filter[1], verbose=False)
        if args.csd:
            raw = mne.preprocessing.compute_current_source_density(raw)
        raw.rename_channels({ch: clean_channel_name_dynamic(ch) for ch in raw.ch_names})
    _generate_clinical_report(raw_eo, raw_ec, output_dir)

if __name__ == "__main__":
    main()
