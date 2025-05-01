#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
data_to_csv.py - Enhanced EDF Data Export Module

This module provides comprehensive data export functionality supporting multiple clinical 
and research formats. It handles:
- Standard epoch-based metrics
- Clinical summary formats (QEEG style)
- Research-grade exports with detailed feature sets
- Multiple standardized formats (EDF+, European Data Format, etc.)
"""

import argparse
import numpy as np
import pandas as pd
import mne
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from modules import io_utils

# Enhanced frequency bands including clinical and research definitions
BANDS = {
    # Clinical bands
    "Delta": (1, 4),
    "Theta": (4, 8),
    "Alpha": (8, 12),
    "SMR": (12, 15),
    "Beta": (15, 27),
    "HighBeta": (28, 38),
    # Research bands
    "Low_Delta": (0.5, 2),
    "High_Delta": (2, 4),
    "Low_Theta": (4, 6),
    "High_Theta": (6, 8),
    "Low_Alpha": (8, 10),
    "High_Alpha": (10, 12),
    "Low_Beta": (12, 15),
    "Mid_Beta": (15, 18),
    "High_Beta": (18, 25),
    "Gamma": (35, 45),
    "High_Gamma": (45, 80),
    # Additional specialized bands
    "Mu": (8, 13),        # Motor cortex rhythm
    "Sigma": (12, 16),    # Sleep spindles
    "SCP": (0.1, 1),      # Slow cortical potentials
    "HFO": (80, 200)      # High-frequency oscillations
}

# Clinical feature sets
CLINICAL_FEATURES = {
    "basic": ["Delta", "Theta", "Alpha", "SMR", "Beta", "HighBeta"],
    "extended": ["Low_Delta", "High_Delta", "Low_Theta", "High_Theta", 
                "Low_Alpha", "High_Alpha", "Low_Beta", "Mid_Beta", "High_Beta"],
    "full": list(BANDS.keys()),
    "sleep": ["Delta", "Theta", "Alpha", "Sigma", "Beta"],
    "motor": ["Delta", "Theta", "Mu", "Beta", "Gamma"],
    "cognitive": ["Theta", "Alpha", "Beta", "Gamma"]
}

# Export format specifications
EXPORT_FORMATS = {
    "standard": {
        "description": "Basic epoch-based metrics",
        "features": CLINICAL_FEATURES["basic"],
        "include_ratios": False
    },
    "clinical": {
        "description": "Clinical QEEG format with ratios",
        "features": CLINICAL_FEATURES["basic"],
        "include_ratios": True,
        "ratios": [
            ("Theta", "Beta", "Theta/Beta"),
            ("Alpha", "Theta", "Alpha/Theta"),
            ("Beta", "Alpha", "Beta/Alpha"),
            ("Delta", "Alpha", "Delta/Alpha"),
            ("Theta", "Alpha", "Theta/Alpha"),
            ("Beta", "Theta", "Beta/Theta")
        ]
    },
    "research": {
        "description": "Comprehensive research format",
        "features": CLINICAL_FEATURES["full"],
        "include_ratios": True,
        "include_connectivity": True,
        "include_complexity": True,
        "include_advanced_stats": True
    },
    "minimal": {
        "description": "Minimal clinical format",
        "features": ["Delta", "Theta", "Alpha", "Beta"],
        "include_ratios": False
    },
    "sleep": {
        "description": "Sleep analysis format",
        "features": CLINICAL_FEATURES["sleep"],
        "include_ratios": True,
        "include_spindles": True,
        "ratios": [
            ("Delta", "Beta", "Delta/Beta"),
            ("Theta", "Beta", "Theta/Beta"),
            ("Sigma", "Beta", "Sigma/Beta")
        ]
    },
    "motor": {
        "description": "Motor analysis format",
        "features": CLINICAL_FEATURES["motor"],
        "include_ratios": True,
        "include_mu_rhythm": True,
        "ratios": [
            ("Mu", "Beta", "Mu/Beta"),
            ("Beta", "Gamma", "Beta/Gamma")
        ]
    },
    "cognitive": {
        "description": "Cognitive analysis format",
        "features": CLINICAL_FEATURES["cognitive"],
        "include_ratios": True,
        "include_complexity": True,
        "ratios": [
            ("Theta", "Alpha", "Theta/Alpha"),
            ("Alpha", "Beta", "Alpha/Beta"),
            ("Theta", "Beta", "Theta/Beta")
        ]
    },
    "connectivity": {
        "description": "Connectivity-focused format",
        "features": CLINICAL_FEATURES["basic"],
        "include_connectivity": True,
        "connectivity_metrics": ["wpli", "plv", "pli", "dpli", "imcoh"],
        "include_network_metrics": True
    }
}

def compute_band_power(data: np.ndarray, sfreq: float, band: Tuple[float, float]) -> float:
    """Enhanced band power computation with better error handling and normalization."""
    try:
        fmin, fmax = band
        filtered = mne.filter.filter_data(data.astype(np.float64), sfreq, fmin, fmax, verbose=False)
        # Normalize by frequency range to make bands comparable
        power = np.mean(filtered ** 2) / (fmax - fmin)
        return float(power)
    except Exception as e:
        print(f"Error computing band power for band {band}: {e}")
        return np.nan

def compute_connectivity_metrics(epoch_data: np.ndarray, sfreq: float, band: Tuple[float, float]) -> Dict[str, float]:
    """Compute connectivity metrics between channels."""
    try:
        from mne.connectivity import spectral_connectivity
        conn = spectral_connectivity(
            epoch_data[np.newaxis, :, :],
            method='wpli',
            sfreq=sfreq,
            fmin=band[0],
            fmax=band[1],
            verbose=False
        )
        return {
            'wpli_mean': float(np.mean(conn[0])),
            'wpli_std': float(np.std(conn[0]))
        }
    except Exception as e:
        print(f"Error computing connectivity for band {band}: {e}")
        return {'wpli_mean': np.nan, 'wpli_std': np.nan}

def compute_complexity_metrics(data: np.ndarray) -> Dict[str, float]:
    """Compute signal complexity metrics."""
    try:
        from antropy import sample_entropy, perm_entropy
        return {
            'sample_entropy': float(sample_entropy(data)),
            'perm_entropy': float(perm_entropy(data))
        }
    except Exception as e:
        print(f"Error computing complexity metrics: {e}")
        return {'sample_entropy': np.nan, 'perm_entropy': np.nan}

def compute_advanced_stats(values: np.ndarray) -> Dict[str, float]:
    """Compute advanced statistical measures."""
    try:
        from scipy import stats
        from numpy import ma
        
        # Handle potential NaN values
        masked_values = ma.masked_invalid(values)
        
        return {
            # Central tendency
            'mean': float(np.mean(masked_values)),
            'median': float(np.median(masked_values)),
            'mode': float(stats.mode(masked_values, keepdims=True)[0][0]),
            'trimmed_mean': float(stats.trim_mean(masked_values, 0.1)),
            
            # Dispersion
            'std': float(np.std(masked_values)),
            'var': float(np.var(masked_values)),
            'mad': float(stats.median_abs_deviation(masked_values)),
            'iqr': float(stats.iqr(masked_values)),
            'range': float(np.ptp(masked_values)),
            
            # Shape
            'skew': float(stats.skew(masked_values)),
            'kurtosis': float(stats.kurtosis(masked_values)),
            
            # Distribution
            'shapiro_stat': float(stats.shapiro(masked_values)[0]),
            'shapiro_p': float(stats.shapiro(masked_values)[1]),
            
            # Robust statistics
            'winsorized_mean': float(stats.winsorize(masked_values, limits=0.05).mean()),
            'huber_mean': float(stats.huber(masked_values)[0])
        }
    except Exception as e:
        print(f"Error computing advanced stats: {e}")
        return {stat: np.nan for stat in [
            'mean', 'median', 'mode', 'trimmed_mean', 'std', 'var', 'mad', 'iqr',
            'range', 'skew', 'kurtosis', 'shapiro_stat', 'shapiro_p',
            'winsorized_mean', 'huber_mean'
        ]}

def compute_network_metrics(connectivity_matrix: np.ndarray) -> Dict[str, float]:
    """Compute graph theory metrics from connectivity matrix."""
    try:
        import networkx as nx
        
        # Create weighted graph from connectivity matrix
        G = nx.from_numpy_array(np.abs(connectivity_matrix))
        
        return {
            'density': float(nx.density(G)),
            'avg_clustering': float(nx.average_clustering(G, weight='weight')),
            'avg_path_length': float(nx.average_shortest_path_length(G, weight='weight')),
            'global_efficiency': float(nx.global_efficiency(G)),
            'modularity': float(nx.community.modularity_max(G)[0]),
            'assortativity': float(nx.degree_assortativity_coefficient(G, weight='weight')),
            'small_worldness': float(nx.sigma(G)) if nx.is_connected(G) else np.nan
        }
    except Exception as e:
        print(f"Error computing network metrics: {e}")
        return {metric: np.nan for metric in [
            'density', 'avg_clustering', 'avg_path_length', 'global_efficiency',
            'modularity', 'assortativity', 'small_worldness'
        ]}

def compute_enhanced_connectivity_metrics(epoch_data: np.ndarray, sfreq: float, band: Tuple[float, float]) -> Dict[str, float]:
    """Compute enhanced connectivity metrics between channels."""
    try:
        from mne.connectivity import spectral_connectivity
        
        methods = ['wpli', 'plv', 'pli', 'dpli', 'imcoh']
        results = {}
        
        for method in methods:
            conn = spectral_connectivity(
                epoch_data[np.newaxis, :, :],
                method=method,
                sfreq=sfreq,
                fmin=band[0],
                fmax=band[1],
                verbose=False
            )
            results[f'{method}_mean'] = float(np.mean(conn[0]))
            results[f'{method}_std'] = float(np.std(conn[0]))
            
            # Add network metrics if matrix is available
            if conn[0].shape[-1] > 2:  # If we have a connectivity matrix
                network_metrics = compute_network_metrics(conn[0])
                results.update({f'{method}_{k}': v for k, v in network_metrics.items()})
        
        return results
    except Exception as e:
        print(f"Error computing enhanced connectivity for band {band}: {e}")
        return {f'{method}_{metric}': np.nan 
                for method in ['wpli', 'plv', 'pli', 'dpli', 'imcoh']
                for metric in ['mean', 'std']}

def process_edf_to_csv(edf_path: str, 
                      epoch_length: float,
                      output_dir: str,
                      export_format: str = "standard",
                      conditions: List[str] = ["EO", "EC"],
                      overwrite: bool = False) -> None:
    """
    Enhanced EDF processing with multiple export formats and conditions.
    
    Args:
        edf_path: Path to the EDF file
        epoch_length: Duration of epochs in seconds
        output_dir: Directory for output files
        export_format: One of EXPORT_FORMATS keys
        conditions: List of conditions to process
        overwrite: Whether to overwrite existing files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    raw = io_utils.load_eeg_data(edf_path, use_csd=False, for_source=False, apply_filter=True)
    sfreq = raw.info["sfreq"]
    
    # Get format specification
    format_spec = EXPORT_FORMATS.get(export_format, EXPORT_FORMATS["standard"])
    
    for condition in conditions:
        # Process each condition
        output_files = {
            "epochs": output_dir / f"{condition}_epochs.csv",
            "summary": output_dir / f"{condition}_summary.csv",
            "connectivity": output_dir / f"{condition}_connectivity.csv",
            "advanced_stats": output_dir / f"{condition}_advanced_stats.csv",
            "network": output_dir / f"{condition}_network_metrics.csv"
        }
        
        # Skip if files exist and not overwriting
        if not overwrite and all(f.exists() for f in output_files.values()):
            print(f"Files already exist for condition {condition}. Skipping...")
            continue

        # Create epochs
        events = mne.make_fixed_length_events(raw, duration=epoch_length, verbose=False)
        epochs = mne.Epochs(raw, events, tmin=0, tmax=epoch_length, baseline=None, preload=True, verbose=False)
        
        # Process epochs
        epoch_rows = []
        summary_data = {ch: {band: [] for band in format_spec["features"]} for ch in epochs.ch_names}
        
        for i, epoch in enumerate(epochs.get_data()):
            epoch_start = events[i, 0] / sfreq
            
            # Process each channel
            for ch_idx, ch in enumerate(epochs.ch_names):
                row = {
                    "Condition": condition,
                    "Channel": ch,
                    "Epoch": i,
                    "Start_Time": epoch_start,
                    "End_Time": epoch_start + epoch_length
                }
                
                # Compute band powers
                for band_name in format_spec["features"]:
                    power = compute_band_power(epoch[ch_idx], sfreq, BANDS[band_name])
                    row[band_name] = power
                    summary_data[ch][band_name].append(power)
                
                # Add ratios if specified
                if format_spec.get("include_ratios"):
                    for band1, band2, ratio_name in format_spec.get("ratios", []):
                        if row[band1] != 0:
                            row[ratio_name] = row[band2] / row[band1]
                        else:
                            row[ratio_name] = np.nan
                
                # Add complexity metrics for research format
                if format_spec.get("include_complexity"):
                    complexity = compute_complexity_metrics(epoch[ch_idx])
                    row.update(complexity)
                
                epoch_rows.append(row)
            
            # Add connectivity metrics if specified
            if format_spec.get("include_connectivity"):
                for band_name in format_spec["features"]:
                    conn_metrics = compute_enhanced_connectivity_metrics(
                        epoch, sfreq, BANDS[band_name])
                    for metric, value in conn_metrics.items():
                        row[f"{band_name}_{metric}"] = value
        
        # Save epoch-level data
        pd.DataFrame(epoch_rows).to_csv(output_files["epochs"], index=False)
        
        # Create and save summary statistics
        summary_rows = []
        for ch in epochs.ch_names:
            row = {"Channel": ch, "Condition": condition}
            for band in format_spec["features"]:
                values = summary_data[ch][band]
                row.update({
                    f"{band}_Mean": np.mean(values),
                    f"{band}_Std": np.std(values),
                    f"{band}_Median": np.median(values)
                })
            summary_rows.append(row)
        
        pd.DataFrame(summary_rows).to_csv(output_files["summary"], index=False)
        
        # Add advanced statistics if specified
        if format_spec.get("include_advanced_stats"):
            advanced_stats_rows = []
            for ch in epochs.ch_names:
                row = {"Channel": ch, "Condition": condition}
                for band in format_spec["features"]:
                    values = summary_data[ch][band]
                    stats = compute_advanced_stats(np.array(values))
                    row.update({f"{band}_{stat}": value 
                              for stat, value in stats.items()})
                advanced_stats_rows.append(row)
            pd.DataFrame(advanced_stats_rows).to_csv(output_files["advanced_stats"], index=False)
        
        # Add enhanced connectivity metrics if specified
        if format_spec.get("include_connectivity"):
            connectivity_rows = []
            for i, epoch in enumerate(epochs.get_data()):
                row = {"Epoch": i, "Condition": condition}
                for band_name in format_spec["features"]:
                    conn_metrics = compute_enhanced_connectivity_metrics(
                        epoch, sfreq, BANDS[band_name])
                    row.update({f"{band_name}_{k}": v 
                              for k, v in conn_metrics.items()})
                connectivity_rows.append(row)
            pd.DataFrame(connectivity_rows).to_csv(output_files["connectivity"], index=False)
        
        print(f"Processed {condition}. Files saved to {output_dir}")

def save_computed_features_to_csv(features: Dict[str, Any], info: Dict[str, Any], output_path: str) -> None:
    """
    Save computed EEG features to a CSV file.
    
    Args:
        features: Dictionary containing computed features
        info: Dictionary containing metadata and information about the recording
        output_path: Path where to save the CSV file
    """
    # Create a flat dictionary for DataFrame
    flat_dict = {}
    
    # Add metadata
    for key, value in info.items():
        if isinstance(value, (str, int, float)):
            flat_dict[f"meta_{key}"] = [value]
    
    # Add features
    for feature_type, feature_dict in features.items():
        if isinstance(feature_dict, dict):
            for metric_name, value in feature_dict.items():
                if isinstance(value, (int, float, str, np.number)):
                    flat_dict[f"{feature_type}_{metric_name}"] = [value]
                elif isinstance(value, np.ndarray) and value.size == 1:
                    flat_dict[f"{feature_type}_{metric_name}"] = [float(value)]
    
    # Create DataFrame and save
    df = pd.DataFrame(flat_dict)
    df.to_csv(output_path, index=False)

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced EDF processing with multiple export formats"
    )
    parser.add_argument("--edf", required=True, help="Path to the EDF file")
    parser.add_argument("--epoch_length", type=float, default=2.0, help="Epoch length in seconds")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--format", choices=list(EXPORT_FORMATS.keys()), default="standard",
                      help="Export format specification")
    parser.add_argument("--conditions", nargs="+", default=["EO", "EC"],
                      help="Conditions to process")
    parser.add_argument("--overwrite", action="store_true",
                      help="Overwrite existing files")
    
    args = parser.parse_args()
    process_edf_to_csv(args.edf, args.epoch_length, args.output_dir,
                      args.format, args.conditions, args.overwrite)

if __name__ == "__main__":
    main()
