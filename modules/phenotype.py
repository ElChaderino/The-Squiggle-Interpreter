import numpy as np
import logging
from pathlib import Path
from .feature_extraction import extract_classification_features # Assuming feature_extraction is in the same 'modules' dir
import time

# Setup logger for this module
logger = logging.getLogger(__name__)


def classify_eeg_profile(features, verbose=False):
    """
    Classify EEG phenotype based on input features.

    Parameters:
        features (dict): EEG metrics including:
            'theta_beta', 'paf', 'high_beta', 'vigilance_sequence',
            'coherence_global', 'delta_power', 'alpha_power', 'smr',
            'low_alpha', 'high_alpha', 'beta', 'theta', 'delta', 'hi_beta',
            ratios: 'hi_alpha_to_lo_alpha', 'hi_beta_to_beta'
            optional: 'csd_frontal_smr', 'csd_parietal_alpha',
                      'sloreta_frontal_hibeta', etc.
        verbose (bool): If True, return reasoning for classification.

    Returns:
        dict: {
            'best_match': str,
            'confidence': float,
            'matches': list of (phenotype, score, recommendations),
            'zscore_summary': dict,
            'vigilance_pattern': list,
            'explanations': list (if verbose)
        }
    """
    # Basic feature assignments with default values
    theta_beta = features.get('theta_beta', 0)
    paf = features.get('paf', 10.0)
    high_beta = features.get('high_beta', 0)
    vigilance = features.get('vigilance_sequence', [])
    delta_power = features.get('delta_power', 0)
    alpha_power = features.get('alpha_power', 0)
    smr = features.get('smr', 0)

    # Swingle-style features (may be used by additional rules in the future)
    theta = features.get('theta', 0)
    delta = features.get('delta', 0)
    low_alpha = features.get('low_alpha', 0)
    high_alpha = features.get('high_alpha', 0)
    beta = features.get('beta', 0)
    hi_beta = features.get('hi_beta', 0)
    hi_alpha_to_lo_alpha = features.get('hi_alpha_to_lo_alpha', 0)
    hi_beta_to_beta = features.get('hi_beta_to_beta', 0)

    # CSD-based (spatially localized power)
    csd_frontal_smr = features.get('csd_frontal_smr', None)
    csd_parietal_alpha = features.get('csd_parietal_alpha', None)

    # sLORETA-based source metrics
    sloreta_frontal_hibeta = features.get('sloreta_frontal_hibeta', None)
    sloreta_parietal_alpha = features.get('sloreta_parietal_alpha', None)

    # Define the profile rules
    profile_rules = [
        {
            "name": "OCD_frontal_loop_localized",
            "condition": lambda: sloreta_frontal_hibeta is not None and sloreta_frontal_hibeta > 2.5,
            "confidence": 0.91,
            "recommendations": ["Downtrain sLORETA Hi-Beta in OFC", "HRV integration"],
            "explanation": "Elevated Hi-Beta source in orbitofrontal region points to OCD-like loop."
        },
        {
            "name": "Trauma_freeze_csd",
            "condition": lambda: (csd_parietal_alpha is not None and csd_parietal_alpha < 1.0) and ("C" in vigilance),
            "confidence": 0.87,
            "recommendations": ["Restore parietal alpha via entrainment or uptraining"],
            "explanation": "Suppressed parietal alpha with freeze stage suggests trauma freeze subtype."
        },
        {
            "name": "Underarousal_smr_deficit",
            "condition": lambda: csd_frontal_smr is not None and csd_frontal_smr < 0.8,
            "confidence": 0.85,
            "recommendations": ["Increase SMR at frontal-central sites"],
            "explanation": "Low CSD-SMR frontally may indicate poor arousal regulation."
        },
        # Additional rules can be added here as needed.
    ]

    matches = []
    explanations = []

    # Evaluate each rule
    for rule in profile_rules:
        if rule["condition"]():
            matches.append((rule["name"], rule["confidence"], rule["recommendations"]))
            if verbose:
                explanations.append(rule["explanation"])

    # Select best match based on evaluation order
    best_match, confidence, recommendations = ("Unclassified", 0.5, []) if not matches else matches[0]

    return {
        "best_match": best_match,
        "confidence": confidence,
        "matches": matches,
        "vigilance_pattern": vigilance,
        "zscore_summary": {
            "Theta/Beta": theta_beta,
            "High Beta": high_beta,
            "Delta Power": delta_power,
            "SMR": smr,
            "PAF": paf,
            "HiAlpha/LoAlpha": hi_alpha_to_lo_alpha,
            "HiBeta/Beta": hi_beta_to_beta,
            "CSD Frontal SMR": csd_frontal_smr,
            "CSD Parietal Alpha": csd_parietal_alpha,
            "sLORETA Frontal Hi-Beta": sloreta_frontal_hibeta,
        },
        "recommendations": recommendations,
        "explanations": explanations if verbose else None
    }

# --- New Orchestration Function --- 
def run_phenotype_analysis(raw_eo, subject_folder: Path, subject: str, 
                           raw_ec=None, csd_raw_eo=None, 
                           sloreta_data=None, vigilance_states=None) -> dict | None:
    """Extracts features, classifies EEG phenotype, and saves results.

    Args:
        raw_eo (mne.io.Raw): Raw EEG data for Eyes Open condition.
        subject_folder (Path): Path object for the subject's output directory.
        subject (str): Subject identifier.
        raw_ec (mne.io.Raw, optional): Raw EEG data for Eyes Closed condition.
        csd_raw_eo (mne.io.Raw, optional): CSD-transformed EO data.
        sloreta_data (dict, optional): Dictionary containing sLORETA source data.
        vigilance_states (list, optional): List of computed vigilance states.

    Returns:
        dict | None: Dictionary containing phenotype classification results, 
                     or None if processing fails.
    """
    logger.info(f"--- Starting Phenotype Analysis for Subject: {subject} ---")
    start_time = time.time() # Need to import time

    if raw_eo is None:
        logger.warning("Skipping phenotype analysis: EO data is None.")
        return None

    phenotype_results = None
    try:
        # 1. Extract Features
        logger.info("  Extracting features for classification...")
        features = extract_classification_features(
            raw=raw_eo,
            vigilance_sequence=vigilance_states, # Pass if available
            eyes_open_raw=raw_eo,
            eyes_closed_raw=raw_ec,
            csd_raw=csd_raw_eo, # Pass CSD EO if available
            sloreta_data=sloreta_data # Pass sLORETA if available
        )
        logger.debug(f"  Extracted features: {list(features.keys())}")

        # 2. Classify Profile
        logger.info("  Classifying EEG profile...")
        # Pass verbose=True if detailed explanations are desired in the output dict
        phenotype_results = classify_eeg_profile(features, verbose=True)
        logger.info(f"  Phenotype classification completed. Best match: {phenotype_results.get('best_match')}")

        # 3. Save Results to File
        # Use the dedicated phenotype plot/data directory if available, otherwise subject_folder
        # Assuming 'folders' dict passed to process_subject has 'plots_phenotype'
        # This function only receives subject_folder, so we save there directly.
        phenotype_report_path = subject_folder / f"{subject}_phenotype_results.txt"
        try:
            with open(phenotype_report_path, "w", encoding="utf-8") as f:
                f.write(f"Phenotype Classification Results for Subject: {subject}\n")
                f.write("==================================================\n")
                if phenotype_results:
                    f.write(f"Best Match: {phenotype_results.get('best_match', 'N/A')}\n")
                    f.write(f"Confidence: {phenotype_results.get('confidence', 'N/A'):.2f}\n\n")
                    if phenotype_results.get('recommendations'):
                        f.write("Recommendations:\n")
                        for rec in phenotype_results['recommendations']:
                            f.write(f"- {rec}\n")
                        f.write("\n")
                    if phenotype_results.get('explanations'):
                         f.write("Explanations:\n")
                         for exp in phenotype_results['explanations']:
                              f.write(f"- {exp}\n")
                         f.write("\n")
                    if phenotype_results.get('zscore_summary'):
                         f.write("Z-Score Summary:\n")
                         for key, value in phenotype_results['zscore_summary'].items():
                              # Format value nicely, handle None
                              f_val = f"{value:.2f}" if isinstance(value, (int, float)) else str(value)
                              f.write(f"- {key}: {f_val}\n")
                else:
                    f.write("No classification results generated.\n")
            logger.info(f"  Saved phenotype results to: {phenotype_report_path}")
        except IOError as e_io:
            logger.error(f"  Error saving phenotype results file to {phenotype_report_path}: {e_io}")
        except Exception as e_save:
             logger.error(f"  Unexpected error saving phenotype results: {e_save}", exc_info=True)


    except ImportError:
         logger.error("  Skipping phenotype analysis: Missing dependency for feature_extraction.")
         return None
    except Exception as e:
        logger.error(f"  Error during phenotype analysis for subject {subject}: {e}", exc_info=True)
        return None # Return None on failure

    end_time = time.time()
    logger.info(f"--- Phenotype Analysis for Subject: {subject} finished in {end_time - start_time:.2f}s ---")
    return phenotype_results # Return computed results dict
