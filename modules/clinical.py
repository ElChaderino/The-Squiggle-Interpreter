#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
clinical_report.py

This module generates clinical reports (HTML and PDF) for EEG data analysis in The Squiggle Interpreter.
It includes site-by-site analysis, global metrics, vigilance plots, source localization, and integrates
CSV metrics from data_to_csv.py for enhanced reporting.

Key Features:
- Generates an interactive HTML report with embedded plots.
- Generates a PDF report summarizing key findings.
- Integrates detailed metrics from CSV files (e.g., alpha reactivity, vigilance states).
- Handles edge cases to prevent crashes during report generation.
"""

import mne
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import matplotlib.pyplot as plt
from .config import BANDS, NORM_VALUES, THRESHOLDS, DETAILED_SITES, CRITICAL_SITES, OUTPUT_FOLDERS, PLOT_CONFIG
from .io_utils import load_eeg_data, find_subject_edf_files
from .processing import (compute_all_band_powers, compute_alpha_peak_frequency, compute_frontal_asymmetry,
                        compute_instability_index, compute_coherence, compute_theta_beta_ratio,
                        compute_percentage_change, compute_all_zscore_maps)
from .plotting import plot_difference_topomap, plot_difference_bar, generate_full_site_plots, plot_topomap_abs_rel
from .pdf_report_builder import build_html_report, build_pdf_report
from .pyramid_model import map_to_pyramid
from .phenotype import classify_eeg_profile
from .feature_extraction import extract_classification_features
import os

logger = logging.getLogger(__name__)

def compute_site_metrics(raw_eo: mne.io.Raw, raw_ec: mne.io.Raw, bp_EO: dict, bp_EC: dict, subject_folder: str) -> tuple[dict, dict]:
    """
    Compute site-specific and global metrics for EEG data.
    
    Args:
        raw_eo, raw_ec (mne.io.Raw): Raw EEG data for EO and EC.
        bp_EO, bp_EC (dict): Band powers for EO and EC.
        subject_folder (str): Path to the subject's output folder.
    
    Returns:
        tuple: (site_metrics, global_metrics)
    """
    if raw_eo is None and raw_ec is None:
        logger.warning("Cannot compute site metrics: No EO or EC data available.")
        return {}, {}
    site_metrics = {}
    global_metrics = {}
    sfreq = raw_eo.info["sfreq"] if raw_eo else raw_ec.info["sfreq"]
    missing_log = Path(subject_folder) / "missing_channels_log.txt"
    
    if raw_eo and raw_ec:
        missing_ec_channels = set(bp_EO.keys()) - set(bp_EC.keys())
        if missing_ec_channels:
            with open(missing_log, "a") as f:
                for ch in missing_ec_channels:
                    f.write(f"{ch} missing from EC\n")
                    logger.warning(f"{ch} missing from EC")
    
    channel_pairs = [("F3", "F4"), ("O1", "O2"), ("T3", "T4")]
    for ch in bp_EO.keys():
        if ch not in bp_EC:
            with open(missing_log, "a") as f:
                f.write(f"{ch} missing from EC\n")
                logger.warning(f"{ch} missing from EC")
            continue
        site_metrics[ch] = {}
        alpha_EO = bp_EO[ch].get("Alpha", 0)
        alpha_EC = bp_EC[ch].get("Alpha", 0)
        site_metrics[ch]["Alpha_Change"] = compute_percentage_change(alpha_EO, alpha_EC)
        site_metrics[ch]["Theta_Beta_Ratio"] = compute_theta_beta_ratio(raw_eo, ch) if raw_eo else np.nan
        ec_sig = raw_ec.get_data(picks=[ch])[0] * 1e6 if raw_ec else np.zeros(1)
        site_metrics[ch]["Alpha_Peak_Freq"] = compute_alpha_peak_frequency(ec_sig, sfreq, BANDS["Alpha"]) if raw_ec else np.nan
        site_metrics[ch]["Delta_Power"] = bp_EO[ch].get("Delta", np.nan)
        site_metrics[ch]["SMR_Power"] = bp_EO[ch].get("SMR", np.nan)
        site_metrics[ch]["Total_Power_EO"] = sum(bp_EO[ch].values())
        site_metrics[ch]["Total_Power_EC"] = sum(bp_EC[ch].values())
        site_metrics[ch]["Total_Amplitude"] = alpha_EO
        for ch1, ch2 in channel_pairs:
            if ch == ch1 and raw_eo and ch2 in raw_eo.ch_names:
                coherence_val = compute_coherence(raw_eo, ch1, ch2, BANDS["Alpha"], sfreq)
                site_metrics[ch][f"Coherence_Alpha_{ch1}_{ch2}"] = coherence_val
    global_metrics["Frontal_Asymmetry"] = compute_frontal_asymmetry(bp_EO) if bp_EO else np.nan
    logger.info("Computed site and global metrics.")
    return site_metrics, global_metrics

def flag_abnormal(value: float, metric: str) -> str:
    """
    Flag metrics outside normative ranges.
    
    Args:
        value (float): Metric value.
        metric (str): Metric name.
    
    Returns:
        str: Abnormality flag.
    """
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
    """
    Provide generic interpretations for abnormal band powers.
    
    Args:
        band (str): Frequency band name.
        value (float): Band power value.
    
    Returns:
        str: Interpretation text.
    """
    norm_mean, norm_sd = NORM_VALUES[band]["mean"], NORM_VALUES[band]["sd"]
    flag = flag_abnormal(value, band)
    if flag == "Within normative range":
        return "Activity is within normative limits."
    interpretations = {
        "Below normative range": {
            "Delta": "Reduced slow-wave activity, possibly impacting recovery processes.",
            "Theta": "Low Theta may indicate hyperarousal, difficulty relaxing, or poor emotional regulation.",
            "Alpha": "Low Alpha may suggest stress, anxiety, or impaired visual processing.",
            "SMR": "Low SMR may reflect reduced calm focus, increased impulsivity, or motor control issues.",
            "Beta": "Low Beta may indicate diminished cognitive processing, alertness, or sustained attention.",
            "HighBeta": "Lower HighBeta is generally not concerning but may suggest reduced stress response.",
        },
        "Above normative range": {
            "Delta": "High Delta may indicate cognitive impairment, attentional deficits, or brain injury.",
            "Theta": "Elevated Theta may be linked to drowsiness, inattention, or daydreaming tendencies.",
            "Alpha": "Excessive Alpha may suggest over-relaxation, disengagement, or lack of focus.",
            "SMR": "High SMR might reflect overcompensation in motor control or hyperfocus.",
            "Beta": "Elevated Beta may relate to anxiety, stress, hyperarousal, or overthinking.",
            "HighBeta": "High HighBeta may indicate excessive stress, cortical overactivation, or agitation.",
        },
    }
    return interpretations.get(flag, {}).get(band, "")

def interpret_metrics(site_metrics: dict, global_metrics: dict, bp_EO: dict, bp_EC: dict) -> list[str]:
    """
    Generate clinical interpretations for site and global metrics.
    
    Args:
        site_metrics (dict): Site-specific metrics.
        global_metrics (dict): Global metrics.
        bp_EO, bp_EC (dict): Band powers for EO and EC.
    
    Returns:
        list: Interpretation strings.
    """
    interpretations = []
    for ch in bp_EO:
        interpretations.append(f"\n=== Site: {ch} ===")
        site_upper = ch.upper()
        interpretations.append("Frequency Band Powers (µV²):")
        for band, power in bp_EO[ch].items():
            flag = flag_abnormal(power, band)
            interpretations.append(f"  {band}: {power:.2f} ({flag})")
            if flag != "Within normative range":
                interpretations.append(f"    {generic_interpretation(band, power)}")
        if site_upper in DETAILED_SITES:
            metrics = site_metrics[ch]
            alpha_change = metrics["Alpha_Change"]
            flag = flag_abnormal(alpha_change, "Alpha_Change")
            interpretations.append(f"Alpha Change (EO→EC): {alpha_change:.1f}% ({flag})")
            if site_upper == "CZ":
                if alpha_change < THRESHOLDS["CZ_Alpha_Percent"]["low"]:
                    interpretations.extend([
                        "  Implications: Deficits in visual processing, memory retention, and recall.",
                        "                Poor cortical engagement during state transitions.",
                        "                May affect learning and cognitive flexibility.",
                        "  Recommendations: Enhance Alpha during EC for memory consolidation.",
                        "                  Optimize modulation for cognitive flexibility.",
                        "                  Consider neurofeedback to improve state transitions.",
                    ])
                elif alpha_change > THRESHOLDS["CZ_Alpha_Percent"]["high"]:
                    interpretations.extend([
                        "  Implications: Cognitive fog or inefficiencies in cortical regulation.",
                        "                May indicate over-relaxation or lack of engagement.",
                        "                Potential for reduced cognitive performance under stress.",
                        "  Recommendations: Enhance Beta (15-18 Hz) and SMR (12-15 Hz), reduce Theta/Delta.",
                        "                  Consider coherence training between frontal and central regions.",
                        "                  Monitor for signs of cognitive overload.",
                    ])
            elif site_upper == "O1":
                if alpha_change < THRESHOLDS["O1_Alpha_EC"]["low"]:
                    interpretations.extend([
                        "  Implications: May indicate traumatic stress or unresolved psychological trauma.",
                        "                Potential for visual processing deficits.",
                        "                May affect emotional regulation and stress response.",
                        "  Recommendations: Enhance Alpha (8-12 Hz), inhibit Theta (4-7 Hz) during EC.",
                        "                  Consider trauma-focused interventions.",
                        "                  Monitor for signs of anxiety or hyperarousal.",
                    ])
                elif alpha_change > THRESHOLDS["O1_Alpha_EC"]["high"]:
                    interpretations.extend([
                        "  Implications: Suggests enhanced artistic interest or introspection.",
                        "                May indicate overactive visual imagination.",
                        "                Potential for dissociation or daydreaming tendencies.",
                        "  Recommendations: Use Alpha training to balance creativity with emotional stability.",
                        "                  Assess for dissociative tendencies.",
                        "                  Consider grounding techniques to improve focus.",
                    ])
            elif site_upper == "PZ":
                if alpha_change < THRESHOLDS["CZ_Alpha_Percent"]["low"]:
                    interpretations.extend([
                        "  Implications: Issues with sensory integration and attention load management.",
                        "                May affect spatial awareness and sensory processing.",
                        "                Potential for difficulties in multitasking.",
                        "  Recommendations: Enhance Alpha modulation, cross-check with FZ and O1.",
                        "                  Consider sensory integration training.",
                        "                  Monitor for attention deficits.",
                    ])
                elif alpha_change > THRESHOLDS["CZ_Alpha_Percent"]["high"]:
                    interpretations.extend([
                        "  Implications: Over-relaxation or reduced sensory processing efficiency.",
                        "                May indicate disengagement from sensory input.",
                        "                Potential for reduced situational awareness.",
                        "  Recommendations: Balance with Beta training.",
                        "                  Enhance sensory engagement activities.",
                        "                  Assess for signs of sensory avoidance.",
                    ])
            elif site_upper == "O2":
                if alpha_change < THRESHOLDS["O1_Alpha_EC"]["low"]:
                    interpretations.extend([
                        "  Implications: Visual processing issues or stress-related disturbances.",
                        "                May affect right-hemisphere visual processing.",
                        "                Potential for emotional dysregulation.",
                        "  Recommendations: Enhance Alpha, assess with O1 and PZ.",
                        "                  Consider stress management techniques.",
                        "                  Monitor for visual processing deficits.",
                    ])
                elif alpha_change > THRESHOLDS["O1_Alpha_EC"]["high"]:
                    interpretations.extend([
                        "  Implications: Possible overcompensation in visual processing.",
                        "                May indicate hyperfocus on visual stimuli.",
                        "                Potential for visual overstimulation.",
                        "  Recommendations: Stabilize Alpha, reduce excessive activity.",
                        "                  Assess for visual overstimulation.",
                        "                  Consider relaxation techniques.",
                    ])
            elif site_upper == "F3":
                if alpha_change < THRESHOLDS["CZ_Alpha_Percent"]["low"]:
                    interpretations.extend([
                        "  Implications: Reduced left frontal Alpha, may indicate hyperarousal.",
                        "                Potential for anxiety or overthinking.",
                        "                May affect executive function and decision-making.",
                        "  Recommendations: Enhance Alpha to promote calm focus.",
                        "                  Consider stress reduction techniques.",
                        "                  Monitor for signs of anxiety.",
                    ])
                elif alpha_change > THRESHOLDS["CZ_Alpha_Percent"]["high"]:
                    interpretations.extend([
                        "  Implications: Excessive left frontal Alpha, may indicate disengagement.",
                        "                Potential for reduced motivation or withdrawal.",
                        "                May affect emotional regulation.",
                        "  Recommendations: Balance with Beta training.",
                        "                  Assess for depressive tendencies.",
                        "                  Consider motivational strategies.",
                    ])
            elif site_upper == "F4":
                if alpha_change < THRESHOLDS["CZ_Alpha_Percent"]["low"]:
                    interpretations.extend([
                        "  Implications: Reduced right frontal Alpha, may indicate emotional suppression.",
                        "                Potential for withdrawal behavior.",
                        "                May affect emotional expression.",
                        "  Recommendations: Enhance Alpha to promote emotional balance.",
                        "                  Consider emotional regulation training.",
                        "                  Monitor for signs of withdrawal.",
                    ])
                elif alpha_change > THRESHOLDS["CZ_Alpha_Percent"]["high"]:
                    interpretations.extend([
                        "  Implications: Excessive right frontal Alpha, may indicate over-relaxation.",
                        "                Potential for reduced emotional engagement.",
                        "                May affect social interaction.",
                        "  Recommendations: Balance with Beta training.",
                        "                  Assess for social withdrawal.",
                        "                  Consider social engagement activities.",
                    ])
            elif site_upper == "FZ":
                if alpha_change < THRESHOLDS["CZ_Alpha_Percent"]["low"]:
                    interpretations.extend([
                        "  Implications: Reduced midline frontal Alpha, may indicate hyperarousal.",
                        "                Potential for difficulties in sustained attention.",
                        "                May affect cognitive control.",
                        "  Recommendations: Enhance Alpha to improve focus.",
                        "                  Consider attention training.",
                        "                  Monitor for signs of inattention.",
                    ])
                elif alpha_change > THRESHOLDS["CZ_Alpha_Percent"]["high"]:
                    interpretations.extend([
                        "  Implications: Excessive midline frontal Alpha, may indicate disengagement.",
                        "                Potential for reduced cognitive effort.",
                        "                May affect executive function.",
                        "  Recommendations: Balance with Beta training.",
                        "                  Assess for cognitive disengagement.",
                        "                  Consider cognitive stimulation activities.",
                    ])
            elif site_upper == "T3":
                if alpha_change < THRESHOLDS["CZ_Alpha_Percent"]["low"]:
                    interpretations.extend([
                        "  Implications: Reduced left temporal Alpha, may indicate auditory processing issues.",
                        "                Potential for difficulties in language comprehension.",
                        "                May affect emotional regulation.",
                        "  Recommendations: Enhance Alpha to improve auditory processing.",
                        "                  Consider language-based interventions.",
                        "                  Monitor for emotional dysregulation.",
                    ])
                elif alpha_change > THRESHOLDS["CZ_Alpha_Percent"]["high"]:
                    interpretations.extend([
                        "  Implications: Excessive left temporal Alpha, may indicate over-relaxation.",
                        "                Potential for reduced auditory engagement.",
                        "                May affect verbal memory.",
                        "  Recommendations: Balance with Beta training.",
                        "                  Assess for auditory disengagement.",
                        "                  Consider auditory stimulation activities.",
                    ])
            elif site_upper == "T4":
                if alpha_change < THRESHOLDS["CZ_Alpha_Percent"]["low"]:
                    interpretations.extend([
                        "  Implications: Reduced right temporal Alpha, may indicate emotional processing issues.",
                        "                Potential for difficulties in emotional recognition.",
                        "                May affect social interaction.",
                        "  Recommendations: Enhance Alpha to improve emotional processing.",
                        "                  Consider emotional recognition training.",
                        "                  Monitor for social difficulties.",
                    ])
                elif alpha_change > THRESHOLDS["CZ_Alpha_Percent"]["high"]:
                    interpretations.extend([
                        "  Implications: Excessive right temporal Alpha, may indicate emotional disengagement.",
                        "                Potential for reduced emotional awareness.",
                        "                May affect empathy.",
                        "  Recommendations: Balance with Beta training.",
                        "                  Assess for emotional disengagement.",
                        "                  Consider empathy-building activities.",
                    ])
            tb_ratio = metrics["Theta_Beta_Ratio"]
            flag = flag_abnormal(tb_ratio, "Theta_Beta_Ratio")
            interpretations.append(f"Theta/Beta Ratio (EO): {tb_ratio:.2f} ({flag})")
            if site_upper in {"CZ", "O1", "F3", "F4", "T3", "T4"}:
                if tb_ratio > THRESHOLDS["Theta_Beta_Ratio"]["severe"]:
                    interpretations.extend([
                        "  Severe: Indicative of ADHD-like symptoms (hyperactivity, impulsivity).",
                        "          May significantly impact attention and impulse control.",
                        "          Potential for academic or occupational challenges.",
                        "  Recommendation: Inhibit Theta (4-8 Hz), enhance Beta (15-27 Hz).",
                        "                 Consider behavioral interventions for ADHD.",
                        "                 Monitor for co-occurring conditions like anxiety.",
                    ])
                elif tb_ratio > THRESHOLDS["Theta_Beta_Ratio"]["threshold"]:
                    interpretations.extend([
                        "  Suggestive of attention regulation challenges.",
                        "  May indicate mild inattention or distractibility.",
                        "  Potential for difficulties in sustained tasks.",
                        "  Recommendation: Monitor and consider Theta/Beta training.",
                        "                 Assess for environmental factors affecting attention.",
                        "                 Consider cognitive training for focus.",
                    ])
            if site_upper == "O1" and tb_ratio < THRESHOLDS["O1_Theta_Beta_Ratio"]["threshold"]:
                interpretations.extend([
                    "  Implications: Reflects poor stress tolerance or heightened anxiety.",
                    "                May indicate overactivation of visual processing.",
                    "                Potential for stress-related visual disturbances.",
                    "  Recommendations: Promote Theta stabilization, inhibit excessive Beta.",
                    "                  Consider stress management techniques.",
                    "                  Monitor for visual stress symptoms.",
                ])
            elif site_upper in {"F3", "F4"} and tb_ratio > THRESHOLDS["F3F4_Theta_Beta_Ratio"]["threshold"]:
                interpretations.extend([
                    "  Implications: Cognitive deficiencies, emotional volatility, or poor impulse control.",
                    "                May affect executive function and decision-making.",
                    "                Potential for emotional outbursts or impulsivity.",
                    "  Recommendations: Inhibit Theta, enhance Alpha for calm alertness and executive function.",
                    "                  Consider emotional regulation training.",
                    "                  Monitor for signs of impulsivity.",
                ])
            elif site_upper in {"T3", "T4"} and tb_ratio > THRESHOLDS["Theta_Beta_Ratio"]["threshold"]:
                interpretations.extend([
                    "  Implications: Auditory processing deficits or emotional dysregulation.",
                    "                May affect language processing or emotional recognition.",
                    "                Potential for difficulties in social communication.",
                    "  Recommendations: Balance Theta/Beta, target auditory processing training.",
                    "                  Consider social skills training.",
                    "                  Monitor for emotional dysregulation.",
                ])
            apf = metrics["Alpha_Peak_Freq"]
            flag = flag_abnormal(apf, "Alpha_Peak_Freq")
            interpretations.append(f"Alpha Peak Frequency (EC): {apf:.2f} Hz ({flag})")
            if apf < THRESHOLDS["Alpha_Peak_Freq"]["low"]:
                interpretations.extend([
                    "  Implication: Slowed Alpha peak, may indicate hypoarousal or cognitive slowing.",
                    "               Potential for reduced cognitive processing speed.",
                    "               May affect learning and memory.",
                    "  Recommendation: Enhance Alpha frequency through training.",
                    "                 Consider cognitive stimulation activities.",
                    "                 Monitor for signs of cognitive fatigue.",
                ])
            elif apf > THRESHOLDS["Alpha_Peak_Freq"]["high"]:
                interpretations.extend([
                    "  Implication: Fast Alpha peak, may indicate hyperarousal or anxiety.",
                    "               Potential for overactivation and stress.",
                    "               May affect relaxation and sleep quality.",
                    "  Recommendation: Stabilize Alpha frequency, reduce stress.",
                    "                 Consider relaxation techniques.",
                    "                 Monitor for signs of anxiety.",
                ])
            delta = metrics["Delta_Power"]
            flag = flag_abnormal(delta, "Delta_Power")
            interpretations.append(f"Delta Power (EO): {delta:.2f} µV² ({flag})")
            if site_upper == "FZ" and delta > THRESHOLDS["FZ_Delta"]["min"]:
                interpretations.extend([
                    "  Implications: Suggests cognitive deficits, poor concentration, or delayed development.",
                    "                May affect sustained attention and working memory.",
                    "                Potential for academic or occupational challenges.",
                    "  Recommendations: Inhibit Delta, enhance SMR for cognitive clarity.",
                    "                  Consider cognitive training for attention.",
                    "                  Monitor for developmental delays.",
                ])
            elif delta > THRESHOLDS["Delta_Power"]["high"]:
                interpretations.extend([
                    "  Implication: Excessive slow-wave activity, possible cognitive deficits.",
                    "               May indicate brain injury or neurological issues.",
                    "               Potential for reduced cognitive performance.",
                    "  Recommendation: Inhibit Delta (1-4 Hz), enhance SMR/Beta.",
                    "                 Assess for neurological conditions.",
                    "                 Monitor for cognitive impairments.",
                ])
            smr = metrics["SMR_Power"]
            flag = flag_abnormal(smr, "SMR_Power")
            interpretations.append(f"SMR Power (EO): {smr:.2f} µV² ({flag})")
            if smr < THRESHOLDS["SMR_Power"]["low"]:
                interpretations.extend([
                    "  Implication: Low SMR, may reflect reduced calm focus or motor control.",
                    "               Potential for impulsivity or hyperactivity.",
                    "               May affect physical coordination.",
                    "  Recommendation: Enhance SMR (12-15 Hz) training.",
                    "                 Consider motor control exercises.",
                    "                 Monitor for signs of impulsivity.",
                ])
            total_eo = metrics["Total_Power_EO"]
            total_ec = metrics["Total_Power_EC"]
            flag_eo = flag_abnormal(total_eo, "Total_Power")
            flag_ec = flag_abnormal(total_ec, "Total_Power")
            interpretations.extend([
                f"Total Power (EO): {total_eo:.2f} µV² ({flag_eo})",
                f"Total Power (EC): {total_ec:.2f} µV² ({flag_ec})",
            ])
            if total_eo > THRESHOLDS["Total_Power"]["high"]:
                interpretations.extend([
                    "  Implication: High overall power (EO), possible developmental delays.",
                    "               May indicate excessive slow-wave activity.",
                    "               Potential for cognitive or developmental issues.",
                    "  Recommendation: Inhibit slow-wave activity, assess for artifacts.",
                    "                 Consider neurological evaluation.",
                    "                 Monitor for developmental progress.",
                ])
            total_amplitude = metrics["Total_Amplitude"]
            if site_upper == "CZ" and total_amplitude > THRESHOLDS["Total_Amplitude"]["max"]:
                interpretations.extend([
                    f"Total Amplitude (Alpha EO): {total_amplitude:.2f} µV²",
                    "  Implications: Potential developmental delays or cognitive deficits.",
                    "                May indicate overactivation or artifacts.",
                    "                Potential for reduced cognitive efficiency.",
                    "  Recommendations: Inhibit slow-wave activity, enhance SMR/Beta.",
                    "                  Assess for artifacts in data.",
                    "                  Monitor for cognitive performance.",
                ])
            for metric_name, value in metrics.items():
                if metric_name.startswith("Coherence_Alpha"):
                    flag = flag_abnormal(value, "Coherence_Alpha")
                    interpretations.append(f"{metric_name}: {value:.2f} ({flag})")
                    if value < THRESHOLDS["Coherence_Alpha"]["low"]:
                        interpretations.extend([
                            "  Implication: Low coherence, may indicate poor connectivity.",
                            "               Potential for difficulties in network integration.",
                            "               May affect cognitive coordination.",
                            "  Recommendation: Enhance coherence through training.",
                            "                 Consider connectivity-focused neurofeedback.",
                            "                 Monitor for cognitive integration issues.",
                        ])
                    elif value > THRESHOLDS["Coherence_Alpha"]["high"]:
                        interpretations.extend([
                            "  Implication: High coherence, may indicate over-synchronization.",
                            "               Potential for reduced flexibility in neural networks.",
                            "               May affect adaptability.",
                            "  Recommendation: Balance coherence levels.",
                            "                 Assess for rigidity in cognitive processing.",
                            "                 Consider flexibility training.",
                        ])
    interpretations.append("\n=== Global Metrics ===")
    fa = global_metrics["Frontal_Asymmetry"]
    flag = flag_abnormal(fa, "Frontal_Asymmetry")
    interpretations.append(f"Frontal Asymmetry (F4/F3 Alpha, EO): {fa:.2f} ({flag})")
    if fa < THRESHOLDS["Frontal_Asymmetry"]["low"]:
        interpretations.extend([
            "  Implication: Left-dominant asymmetry, may indicate depressive tendencies.",
            "               Potential for reduced positive affect.",
            "               May affect emotional well-being.",
            "  Recommendation: Enhance right frontal Alpha, monitor emotional regulation.",
            "                 Consider mood-focused interventions.",
            "                 Monitor for signs of depression.",
        ])
    elif fa > THRESHOLDS["Frontal_Asymmetry"]["high"]:
        interpretations.extend([
            "  Implication: Right-dominant asymmetry, may indicate withdrawal behavior.",
            "               Potential for reduced approach motivation.",
            "               May affect social engagement.",
            "  Recommendation: Enhance left frontal Alpha, assess emotional state.",
            "                 Consider social engagement strategies.",
            "                 Monitor for signs of withdrawal.",
        ])
    logger.info("Generated clinical interpretations.")
    return interpretations

def save_site_metrics(site_metrics: dict, global_metrics: dict, output_path: Path) -> None:
    """
    Save site and global metrics to a CSV file.
    
    Args:
        site_metrics (dict): Site-specific metrics.
        global_metrics (dict): Global metrics.
        output_path (Path): Output CSV file path.
    
    Returns:
        None
    """
    try:
        rows = []
        for ch, met in site_metrics.items():
            row = {"Channel": ch}
            row.update({key: val for key, val in met.items()})
            rows.append(row)
        global_row = {"Channel": "Global"}
        global_row.update(global_metrics)
        rows.append(global_row)
        df = pd.DataFrame(rows)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Clinical metrics saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save clinical metrics to {output_path}: {e}")

def generate_reports(raw_eo, raw_ec, folders, subject_folder, subject, band_list, config, skip_html=False, **results):
    """
    Generate clinical reports and visualizations for EEG data.
    
    Args:
        raw_eo, raw_ec (mne.io.Raw): Raw EEG data for EO and EC (non-CSD).
        folders (dict): Dictionary of output folders.
        subject_folder (str): Path to the subject's output folder.
        subject (str): Subject identifier.
        band_list (list): List of frequency bands.
        config (dict): Configuration dictionary.
        skip_html (bool): If True, skip HTML report generation.
        **results: Visualization results (topomaps, waveforms, etc.).
    """
    if raw_eo is None and raw_ec is None:
        logger.warning(f"Skipping report generation for subject {subject}: No EO or EC data available.")
        return

    try:
        subject_folder = Path(subject_folder)
        output_dir = subject_folder / OUTPUT_FOLDERS["detailed"]

        # Determine whether to use CSD-transformed data
        raw_eo_csd = results.get('raw_eo_csd', raw_eo)
        raw_ec_csd = results.get('raw_ec_csd', raw_ec)

        # Use CSD-transformed data for band powers if available
        bp_raw_eo = raw_eo_csd if config['csd'] and raw_eo_csd else raw_eo
        bp_raw_ec = raw_ec_csd if config['csd'] and raw_ec_csd else raw_ec
        bp_eo = compute_all_band_powers(bp_raw_eo) if bp_raw_eo else {}
        bp_ec = compute_all_band_powers(bp_raw_ec) if bp_raw_ec else {}
        logger.info(f"Subject {subject} - Computed band powers for EO channels: {list(bp_eo.keys()) if bp_eo else 'None'}")
        logger.info(f"Subject {subject} - Computed band powers for EC channels: {list(bp_ec.keys()) if bp_ec else 'None'}")

        # Generate text reports
        site_metrics, global_metrics = compute_site_metrics(raw_eo, raw_ec, bp_eo, bp_ec, subject_folder)
        save_site_metrics(site_metrics, global_metrics, subject_folder / "clinical_metrics.csv")
        interpretations = interpret_metrics(site_metrics, global_metrics, bp_eo, bp_ec)
        with open(subject_folder / "clinical_interpretations.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(interpretations))
        logger.info(f"Clinical interpretations saved to: {subject_folder / 'clinical_interpretations.txt'}")

        pyramid_mappings = map_to_pyramid(bp_eo, bp_ec, site_metrics, global_metrics)
        with open(subject_folder / "pyramid_mappings.txt", "w", encoding="utf-8") as f:
            for mapping in pyramid_mappings:
                f.write(f"{mapping}\n")
        logger.info(f"Pyramid mappings saved to: {subject_folder / 'pyramid_mappings.txt'}")

        # Skip site plots, as handled in pipeline.py
        site_dict = results.get("site_dict", {})

        # Compute instability indices
        instability_eo = compute_instability_index(bp_raw_eo, BANDS) if bp_raw_eo else {}
        instability_ec = compute_instability_index(bp_raw_ec, BANDS) if bp_raw_ec else {}

        # Generate global topomaps
        global_diff_images = {}
        for band_name in band_list:
            if raw_eo:
                abs_powers_eo = np.array([bp_eo[ch][band_name] for ch in raw_eo.ch_names])
                total_powers_eo = np.array([sum(bp_eo[ch].values()) for ch in raw_eo.ch_names])
                rel_powers_eo = np.zeros_like(abs_powers_eo)
                mask_eo = total_powers_eo > 0
                rel_powers_eo[mask_eo] = (abs_powers_eo[mask_eo] / total_powers_eo[mask_eo]) * 100
                instability_vals_eo = np.array([instability_eo[band_name][ch] for ch in raw_eo.ch_names]) if instability_eo else None
                fig_eo = plot_topomap_abs_rel(
                    abs_powers_eo, rel_powers_eo, raw_eo, band_name, "EO", instability_vals=instability_vals_eo
                )
                if fig_eo:
                    topo_path_eo = Path(folders["topomaps_eo"]) / f"topomap_{band_name}_EO.png"
                    topo_path_eo.parent.mkdir(parents=True, exist_ok=True)
                    fig_eo.savefig(topo_path_eo, dpi=PLOT_CONFIG["dpi"], facecolor="black")
                    plt.close(fig_eo)
                    logger.info(f"Saved topomap for {band_name} (EO) to {topo_path_eo}")
            if raw_ec:
                abs_powers_ec = np.array([bp_ec[ch][band_name] for ch in raw_ec.ch_names])
                total_powers_ec = np.array([sum(bp_ec[ch].values()) for ch in raw_ec.ch_names])
                rel_powers_ec = np.zeros_like(abs_powers_ec)
                mask_ec = total_powers_ec > 0
                rel_powers_ec[mask_ec] = (abs_powers_ec[mask_ec] / total_powers_ec[mask_ec]) * 100
                instability_vals_ec = np.array([instability_ec[band_name][ch] for ch in raw_ec.ch_names]) if instability_ec else None
                fig_ec = plot_topomap_abs_rel(
                    abs_powers_ec, rel_powers_ec, raw_ec, band_name, "EC", instability_vals=instability_vals_ec
                )
                if fig_ec:
                    topo_path_ec = Path(folders["topomaps_ec"]) / f"topomap_{band_name}_EC.png"
                    topo_path_ec.parent.mkdir(parents=True, exist_ok=True)
                    fig_ec.savefig(topo_path_ec, dpi=PLOT_CONFIG["dpi"], facecolor="black")
                    plt.close(fig_ec)
                    logger.info(f"Saved topomap for {band_name} (EC) to {topo_path_ec}")

        # Generate difference plots
        if raw_eo and raw_ec:
            for b in band_list:
                try:
                    diff_vals = [bp_eo[ch][b] - bp_ec[ch][b] for ch in raw_eo.ch_names]
                    diff_topo_fig = plot_difference_topomap(diff_vals, raw_eo.info, b)
                    diff_bar_fig = plot_difference_bar(diff_vals, raw_eo.ch_names, b)
                    diff_topo_path = output_dir / f"DifferenceTopomap_{b}.png"
                    diff_bar_path = output_dir / f"DifferenceBar_{b}.png"
                    diff_topo_fig.savefig(diff_topo_path, facecolor='black')
                    diff_bar_fig.savefig(diff_bar_path, facecolor='black')
                    plt.close(diff_topo_fig)
                    plt.close(diff_bar_fig)
                    global_diff_images[b] = {
                        "diff_topo": os.path.basename(diff_topo_path),
                        "diff_bar": os.path.basename(diff_bar_path)
                    }
                    logger.info(f"Generated global difference images for {b}: Topomap={diff_topo_path}, Bar={diff_bar_path}")
                except Exception as e:
                    logger.warning(f"Failed to generate difference images for {b}: {e}")

        site_list = raw_eo.ch_names if raw_eo else (raw_ec.ch_names if raw_ec else [])
        if not site_dict:
            logger.warning("Site dictionary is empty; site plots may not have been generated correctly.")

        # Load phenotype data
        phenotype_data = {}
        phenotype_report_path = subject_folder / f"{subject}_phenotype.txt"
        if phenotype_report_path.exists():
            try:
                with open(phenotype_report_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for line in lines[2:]:
                        if ":" in line:
                            key, value = line.split(":", 1)
                            phenotype_data[key.strip()] = value.strip()
            except Exception as e:
                logger.warning(f"Failed to read phenotype data from {phenotype_report_path}: {e}")

        # Prepare hypnograms
        hypnograms = {
            "EO": {},
            "EC": {},
            "EO_CSD": {},
            "EC_CSD": {}
        }
        available_channels = raw_eo.ch_names if raw_eo else (raw_ec.ch_names if raw_ec else [])
        for ch_name in available_channels:
            for condition in ["EO", "EC", "EO_CSD", "EC_CSD"]:
                hypnogram_file = f"vigilance_hypnogram_{condition}_{ch_name}.png"
                hypnogram_path = subject_folder / OUTPUT_FOLDERS["vigilance"] / hypnogram_file
                if hypnogram_path.exists():
                    hypnograms[condition][ch_name] = hypnogram_file

        # Prepare report data
        logger.debug("Preparing report data for HTML generation...")
        report_data = {
            "global_topomaps": {
                "EO": results.get("topomaps", {}).get("EO", {}),
                "EC": results.get("topomaps", {}).get("EC", {})
            },
            "global_waveforms": results.get("waveforms", {}).get("EO", {}),
            "coherence": {
                "EO": results.get("coherence", {}).get("EO", {}),
                "EC": results.get("coherence", {}).get("EC", {})
            },
            "global_erp": {
                "EO": str(os.path.basename(results.get("erp", {}).get("EO", ""))) if results.get("erp", {}).get("EO") else "",
                "EC": str(os.path.basename(results.get("erp", {}).get("EC", ""))) if results.get("erp", {}).get("EC") else ""
            },
            "zscore": {
                "EO": results.get("zscores", {}).get("EO", {}),
                "EC": results.get("zscores", {}).get("EC", {})
            },
            "variance": {
                "EO": results.get("variance", {}).get("EO", {}),
                "EC": results.get("variance", {}).get("EC", {})
            },
            "tfr": {
                "EO": results.get("tfr", {}).get("EO", {}),
                "EC": results.get("tfr", {}).get("EC", {})
            },
            "ica": {
                "EO": str(os.path.basename(results.get("ica", {}).get("EO", ""))) if results.get("ica", {}).get("EO") else "",
                "EC": ""
            },
            "source_localization": results.get("source_localization", {}),
            "site_list": site_list,
            "band_list": band_list,
            "site_dict": site_dict,
            "global_topomaps_path": OUTPUT_FOLDERS["topomaps_eo"],
            "global_waveforms_path": OUTPUT_FOLDERS["waveforms_eo"],
            "coherence_path": OUTPUT_FOLDERS["coherence_eo"],
            "global_erp_path": OUTPUT_FOLDERS["erp"],
            "tfr_path": OUTPUT_FOLDERS["tfr_eo"],
            "ica_path": OUTPUT_FOLDERS["ica_eo"],
            "sites_path": OUTPUT_FOLDERS["detailed"],
            "source_path": OUTPUT_FOLDERS["source"],
            "phenotype": phenotype_data,
            "hypnograms": hypnograms
        }
        logger.debug(f"Report data prepared: {report_data}")

        # Generate reports
        logger.info("Attempting to generate reports...")
        if config["report"] and not skip_html:
            logger.info("Report generation is enabled (config['report'] = True).")
            try:
                subject_report_path = subject_folder / "eeg_report.html"
                logger.info(f"Generating HTML report at {subject_report_path}...")
                build_html_report(report_data, subject_report_path)
                logger.info(f"Generated interactive HTML report at {subject_report_path}")
            except Exception as e:
                logger.error(f"Failed to generate HTML report: {e}")
                raise

        try:
            logger.info("Generating PDF report...")
            build_pdf_report(
                report_output_dir=subject_folder,
                band_powers={"EO": bp_eo, "EC": bp_ec},
                instability_indices={"EO": instability_eo, "EC": instability_ec},
                source_localization=results.get("source_localization", {}),
                vigilance_plots=hypnograms,
                channels=site_list
            )
            logger.info(f"Generated PDF report at {subject_folder / 'clinical_report.pdf'}")
        except Exception as e:
            logger.error(f"Failed to generate PDF report: {e}")
            raise

        if config["phenotype"] and raw_eo:
            process_phenotype(raw_eo, subject_folder, subject)

    except Exception as e:
        logger.error(f"Failed to generate reports for subject {subject}: {e}")
        raise

def process_phenotype(raw_eo, subject_folder, subject):
    """
    Process EEG phenotype classification.
    
    Args:
        raw_eo (mne.io.Raw): Raw EEG data for EO.
        subject_folder (str): Subject directory.
        subject (str): Subject identifier.
    
    Returns:
        None
    """
    if raw_eo is None:
        logger.warning("Skipping phenotype processing: EO data is None.")
        return
    try:
        features = extract_classification_features(raw_eo, [])
        phenotype_results = classify_eeg_profile(features)

        phenotype_report_path = Path(subject_folder) / f"{subject}_phenotype.txt"
        with open(phenotype_report_path, "w", encoding="utf-8") as f:
            f.write("Phenotype Classification Results\n")
            f.write("===============================\n")
            for k, v in phenotype_results.items():
                f.write(f"{k}: {v}\n")
        logger.info(f"Phenotype results saved to: {phenotype_report_path}")

        clinical_txt_path = Path(subject_folder) / f"{subject}_clinical_report.txt"
        with open(clinical_txt_path, "a", encoding="utf-8") as f:
            f.write("\n\nPhenotype Classification Results\n")
            f.write("===============================\n")
            for k, v in phenotype_results.items():
                f.write(f"{k}: {v}\n")
        logger.info(f"Appended phenotype results to: {clinical_txt_path}")
    except Exception as e:
        logger.error(f"Failed to process phenotype for subject {subject}: {e}")
