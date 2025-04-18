# modules/phenotype.py
import numpy as np
import logging
from pathlib import Path

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

def process_phenotype(raw_eo, subject_folder, subject):
    """
    Placeholder for processing EEG phenotype classification.
    
    Args:
        raw_eo (mne.io.Raw): Eyes-open raw EEG data.
        subject_folder (str): Directory to save phenotype results.
        subject (str): Subject identifier.
    
    Returns:
        dict: Placeholder phenotype results.
    """
    logger.warning("Phenotype processing not fully implemented: Missing feature_extraction.py")
    output_path = Path(subject_folder) / "phenotype.txt"
    try:
        # Placeholder: Simulate feature extraction (requires feature_extraction.py)
        features = {
            'theta_beta': 0.0,
            'paf': 10.0,
            'high_beta': 0.0,
            'vigilance_sequence': [],
            'delta_power': 0.0,
            'alpha_power': 0.0,
            'smr': 0.0
        }
        result = classify_eeg_profile(features, verbose=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"Subject: {subject}\n")
            f.write(f"Best Match: {result['best_match']}\n")
            f.write(f"Confidence: {result['confidence']:.2f}\n")
            f.write("Recommendations:\n")
            for rec in result['recommendations']:
                f.write(f"- {rec}\n")
            if result['explanations']:
                f.write("Explanations:\n")
                for exp in result['explanations']:
                    f.write(f"- {exp}\n")
        logger.info(f"Saved phenotype results to {output_path}")
        return result
    except Exception as e:
        logger.error(f"Failed to process phenotype for {subject}: {e}")
        return {}
