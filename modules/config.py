import argparse
import sys
import logging

logger = logging.getLogger(__name__)

# Directory structure for output files
OUTPUT_FOLDERS = {
    "detailed": "detailed_site_plots",
    "topomaps_eo": "topomaps/EO",
    "topomaps_ec": "topomaps/EC",
    "waveforms_eo": "waveforms/EO",
    "waveforms_ec": "waveforms/EC",
    "coherence_eo": "coherence/EO",
    "coherence_ec": "coherence/EC",
    "zscore_eo": "zscore/EO",
    "zscore_ec": "zscore/EC",
    "variance_eo": "variance/EO",
    "variance_ec": "variance/EC",
    "tfr_eo": "tfr/EO",
    "tfr_ec": "tfr/EC",
    "ica_eo": "ica/EO",
    "ica_ec": "ica/EC",
    "erp": "erp",
    "source": "source",
    "vigilance": "vigilance"
}

# EEG channel constants
CRITICAL_SITES = {"F3", "F4", "CZ", "PZ", "O1", "O2", "T7", "T8", "FZ"}

# Frequency bands for analysis
BANDS = {
    "Delta": (1, 4),
    "Theta": (4, 8),
    "Alpha": (8, 12),
    "SMR": (12, 15),
    "Beta": (15, 27),
    "HighBeta": (28, 38),
}

# Channels prioritized for vigilance processing
VIGILANCE_CHANNELS = ['OZ', 'O1', 'O2', 'PZ']

# Frequency band shortcuts for vigilance processing
ALPHA_BAND = BANDS["Alpha"]  # (8, 12)
THETA_BAND = BANDS["Theta"]  # (4, 8)

# Colors for vigilance state visualizations
VIGILANCE_COLORS = {
    'A1': 'cyan',
    'A2': 'blue',
    'A3': 'green',
    'B1': 'yellow',
    'B2': 'orange',
    'B3': 'red',
    'C': 'gray'
}

# Output directory structure
OUTPUT_FOLDERS = {
    "topomaps_eo": "topomaps/EO",
    "topomaps_ec": "topomaps/EC",
    "waveforms_eo": "waveforms/EO",
    "erp": "erp",
    "coherence_eo": "coherence/EO",
    "coherence_ec": "coherence/EC",
    "zscore_eo": "zscore/EO",
    "zscore_ec": "zscore/EC",
    "tfr_eo": "tfr/EO",
    "tfr_ec": "tfr/EC",
    "ica_eo": "ica/EO",
    "source": "source_localization",
    "detailed": "detailed_site_plots",
    "vigilance": "vigilance",
    "variance_eo": "variance/EO",
    "variance_ec": "variance/EC"
}

# Z-score normalization methods
ZSCORE_METHODS = {
    "1": "standard",
    "2": "robust_mad",
    "3": "robust_iqr",
    "4": "published_norms"
}

# Normative statistics for z-score normalization
DEFAULT_NORM_STATS = {
    "Alpha": {"median": 18.0, "mad": 6.0},
    "Theta": {"median": 15.0, "mad": 5.0},
    "Delta": {"median": 20.0, "mad": 7.0},
    "SMR": {"median": 6.0, "mad": 2.0},
    "Beta": {"median": 5.0, "mad": 2.0},
    "HighBeta": {"median": 3.5, "mad": 1.5}
}

PUBLISHED_NORM_STATS = {
    "Alpha": {"median": 20.0, "mad": 4.0},
    "Theta": {"median": 16.0, "mad": 3.5},
    "Delta": {"median": 22.0, "mad": 5.0},
    "SMR": {"median": 7.0, "mad": 1.5},
    "Beta": {"median": 6.0, "mad": 1.8},
    "HighBeta": {"median": 4.0, "mad": 1.2}
}

# Normative values for flagging abnormalities (mean and standard deviation)
NORM_VALUES = {
    "Alpha": {"mean": 18.0, "sd": 6.0},
    "Theta": {"mean": 15.0, "sd": 5.0},
    "Delta": {"mean": 20.0, "sd": 7.0},
    "SMR": {"mean": 6.0, "sd": 2.0},
    "Beta": {"mean": 5.0, "sd": 2.0},
    "HighBeta": {"mean": 3.5, "sd": 1.5},
    "Alpha_Change": {"mean": 50.0, "sd": 20.0},  # Percentage change EO to EC
    "Theta_Beta_Ratio": {"mean": 1.5, "sd": 0.5},
    "Alpha_Peak_Freq": {"mean": 10.0, "sd": 1.0},  # Hz
    "Delta_Power": {"mean": 20.0, "sd": 7.0},
    "SMR_Power": {"mean": 6.0, "sd": 2.0},
    "Total_Power": {"mean": 100.0, "sd": 30.0},  # Total power EO or EC
    "Frontal_Asymmetry": {"mean": 0.0, "sd": 0.5},  # Log ratio F4/F3
    "Coherence_Alpha": {"mean": 0.7, "sd": 0.2}
}

# Thresholds for specific metrics and sites
THRESHOLDS = {
    "CZ_Alpha_Percent": {"low": 20.0, "high": 80.0},  # Alpha change percentage
    "O1_Alpha_EC": {"low": 10.0, "high": 90.0},  # Alpha change for O1
    "Theta_Beta_Ratio": {"threshold": 2.0, "severe": 3.0},
    "F3F4_Theta_Beta_Ratio": {"threshold": 2.0},
    "O1_Theta_Beta_Ratio": {"threshold": 1.0},
    "FZ_Delta": {"min": 25.0},
    "Delta_Power": {"high": 30.0},
    "SMR_Power": {"low": 4.0},
    "Total_Power": {"high": 150.0},
    "Total_Amplitude": {"max": 50.0},  # Alpha EO amplitude
    "Alpha_Peak_Freq": {"low": 8.0, "high": 12.0},
    "Coherence_Alpha": {"low": 0.4, "high": 0.9},
    "Frontal_Asymmetry": {"low": -1.0, "high": 1.0}
}

# Sites for detailed clinical interpretations
DETAILED_SITES = {"CZ", "O1", "O2", "PZ", "F3", "F4", "FZ", "T3", "T4"}

# Plotting configuration
PLOT_CONFIG = {
    "topomap_figsize": (10, 4),
    "waveform_figsize": (15, 3),  # Adjusted for better layout
    "psd_figsize": (6, 4),
    "coherence_figsize": (8, 6),
    "tfr_figsize": (8, 6),
    "bar_figsize": (10, 6),
    "source_figsize": (8, 6),
    "topomap_cmap": "viridis",
    "zscore_cmap": "coolwarm",
    "difference_cmap": "RdBu_r",
    "waveform_colors": ("cyan", "magenta"),
    "zscore_clim": (-3, 3),
    "epoch_length": 10,
    "n_cols_waveform": 3,  # Reduced for readability
    "dpi": 100
}

def load_zscore_stats(method_choice):
    """
    Load z-score normative statistics based on method choice.
    
    Args:
        method_choice (str): Z-score method ID ('1', '2', '3', '4').
    
    Returns:
        dict: Normative statistics for z-score normalization.
    """
    if method_choice == "4":
        logger.info("Using published normative values for adult EEG.")
        return PUBLISHED_NORM_STATS
    logger.info("Using default normative values for z-score normalization.")
    return DEFAULT_NORM_STATS

def parse_arguments():
    """
    Parse command-line arguments or prompt for input to configure the EEG analysis.
    
    Returns:
        dict: Configuration settings.
    """
    parser = argparse.ArgumentParser(
        prog='The Squiggle Interpreter',
        description='Comprehensive EEG Analysis & Clinical Report Generation'
    )
    parser.add_argument('--csd', help="Use current source density (CSD) for graphs? (y/n)")
    parser.add_argument('--zscore',
                        help="Z-score normalization method: 1: Standard, 2: Robust (MAD), 3: Robust (IQR), 4: Published Norms")
    parser.add_argument('--report', help="Generate full clinical report? (y/n)")
    parser.add_argument('--phenotype', help="Generate phenotype classification? (y/n)")
    parser.add_argument('--csv', action='store_true', help="Export EDF data metrics to CSV")
    parser.add_argument('--edf', help="Path to an EDF file for CSV export")
    parser.add_argument('--epoch_length', type=float, default=2.0,
                        help="Epoch length (in seconds) for CSV export (default: 2.0)")
    parser.add_argument('--output_csv', help="Output CSV file path for CSV export")

    args = parser.parse_args()
    config = {'csv': args.csv}

    if config['csv']:
        if not args.edf or not args.output_csv:
            logger.error("For CSV export, please provide both --edf and --output_csv arguments.")
            sys.exit(1)
        config['edf_path'] = args.edf
        config['epoch_length'] = args.epoch_length
        config['output_csv'] = args.output_csv
        return config

    config['csd'] = (args.csd or input(
        "Use current source density (CSD) for graphs? (y/n, default: n): ") or "n").lower() == "y"
    logger.info(f"Using CSD for graphs: {config['csd']}")

    if args.zscore is None:
        logger.info("Choose z-score normalization method:")
        logger.info("  1: Standard (mean/std)")
        logger.info("  2: Robust (MAD-based)")
        logger.info("  3: Robust (IQR-based)")
        logger.info("  4: Published Norms (adult norms)")
        config['zscore'] = input("Enter choice (default: 1): ") or "1"
    else:
        config['zscore'] = args.zscore
    logger.info(f"Selected z-score method: {ZSCORE_METHODS.get(config['zscore'], 'unknown')}")

    config['report'] = (args.report or input(
        "Generate full clinical report? (y/n, default: y): ") or "y").lower() == "y"
    config['phenotype'] = (args.phenotype or input(
        "Generate phenotype classification? (y/n, default: y): ") or "y").lower() == "y"

    return config