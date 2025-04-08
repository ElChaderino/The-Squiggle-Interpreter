#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
from typing import Dict, Tuple, List
import mne
from . import pdf_report_builder
import re
from . import pyramid_model
from modules.io_utils import load_eeg_data as load_data

BANDS = {
    "Delta": (1, 4),
    "Theta": (4, 8),
    "Alpha": (8, 12),
    "SMR": (12, 15),
    "Beta": (16, 27),
    "HighBeta": (28, 38),
}

NORM_VALUES = {
    "Delta": {"mean": 20.0, "sd": 10.0},
    "Theta": {"mean": 15.0, "sd": 7.0},
    "Alpha": {"mean": 18.0, "sd": 6.0},
    "SMR": {"mean": 6.0, "sd": 2.5},
    "Beta": {"mean": 5.0, "sd": 2.0},
    "HighBeta": {"mean": 3.5, "sd": 1.5},
    "Alpha_Change": {"mean": 50.0, "sd": 15.0},
    "Theta_Beta_Ratio": {"mean": 1.5, "sd": 0.5},
    "Alpha_Peak_Freq": {"mean": 10.0, "sd": 1.0},
    "Delta_Power": {"mean": 20.0, "sd": 10.0},
    "SMR_Power": {"mean": 6.0, "sd": 2.5},
    "Frontal_Asymmetry": {"mean": 0.0, "sd": 0.1},
    "Total_Power": {"mean": 70.0, "sd": 20.0},
    "Coherence_Alpha": {"mean": 0.7, "sd": 0.1},
}

THRESHOLDS = {
    "CZ_Alpha_Percent": {"low": 30, "high": 25},
    "Theta_Beta_Ratio": {"threshold": 2.2, "severe": 3.0},
    "Total_Amplitude": {"max": 60},
    "O1_Alpha_EC": {"low": 50, "high": 150},
    "O1_Theta_Beta_Ratio": {"threshold": 1.8},
    "F3F4_Theta_Beta_Ratio": {"threshold": 2.2},
    "FZ_Delta": {"min": 9.0},
    "Alpha_Change": {"low": 30, "high": 70},
    "Alpha_Peak_Freq": {"low": 8.0, "high": 12.0},
    "Delta_Power": {"high": 30.0},
    "SMR_Power": {"low": 3.0},
    "Frontal_Asymmetry": {"low": -0.2, "high": 0.2},
    "Total_Power": {"high": 100.0},
    "Coherence_Alpha": {"low": 0.5, "high": 0.9},
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
    fmin, fmax = band
    freqs, psd = welch(data, fs=sfreq, nperseg=int(sfreq * 2), noverlap=int(sfreq))
    band_mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band_mask):
        return 0.0
    band_psd = psd[band_mask]
    freq_res = freqs[1] - freqs[0]
    power = np.sum(band_psd) * freq_res
    return float(power)

def compute_instability_index(raw: mne.io.Raw, bands: Dict[str, Tuple[float, float]]) -> Dict[str, Dict[str, float]]:
    sfreq = raw.info["sfreq"]
    data = raw.get_data() * 1e6
    instability = {}
    for band_name, (fmin, fmax) in bands.items():
        data_filt = mne.filter.filter_data(data, sfreq, fmin, fmax, verbose=False)
        variance = np.var(data_filt, axis=1)  # Shape: (n_channels,)
        # Convert the NumPy array to a dictionary mapping channel names to variances
        instability[band_name] = {ch: float(var) for ch, var in zip(raw.ch_names, variance)}
        print(f"\n=== Instability Index (Variance) for {band_name} ===")
        for ch, var in instability[band_name].items():
            print(f"{ch}: {var:.2f} µV²")
    return instability

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

def compute_site_metrics(raw_eo: mne.io.Raw, raw_ec: mne.io.Raw, bp_EO: Dict, bp_EC: Dict) -> Tuple[Dict, Dict]:
    site_metrics = {}
    global_metrics = {}
    sfreq = raw_eo.info["sfreq"]
    missing_ec_channels = set(bp_EO.keys()) - set(bp_EC.keys())
    if missing_ec_channels:
        with open("missing_channels_log.txt", "a") as f:
            for ch in missing_ec_channels:
                f.write(f"{ch} missing from EC\n")
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
    global_metrics["Frontal_Asymmetry"] = compute_frontal_asymmetry(bp_EO, "F3", "F4")
    return site_metrics, global_metrics

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

def interpret_metrics(site_metrics: Dict, global_metrics: Dict, bp_EO: Dict, bp_EC: Dict) -> List[str]:
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
                        "  Recommendations: Enhance Alpha, assess with O1 and Pz.",
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
    return interpretations

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
    cmap: str = "viridis",
    cmap_instability: str = "hot",
    figsize: Tuple[float, float] = (15, 4),
    dpi: int = 300,
    normalize: bool = False,
    show_ch_names: bool = True
) -> None:
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
        ch_names = raw.ch_names
        instability_reordered = np.zeros(len(ch_names))
        for i, ch in enumerate(ch_names):
            idx = raw.ch_names.index(ch)
            instability_reordered[i] = instability_vals[idx]
        instability_subset = instability_reordered[sel_idx]
    else:
        instability_subset = None
    if normalize:
        abs_vals_subset = (abs_vals_subset - abs_vals_subset.min()) / (abs_vals_subset.max() - abs_vals_subset.min() + 1e-10)
        rel_vals_subset = (rel_vals_subset - rel_vals_subset.min()) / (rel_vals_subset.max() - rel_vals_subset.min() + 1e-10)
        if instability_subset is not None:
            instability_subset = (instability_subset - instability_subset.min()) / (instability_subset.max() - instability_subset.min() + 1e-10)
    n_subplots = 3 if instability_subset is not None else 2
    fig, axes = plt.subplots(1, n_subplots, figsize=figsize, facecolor="black")
    fig.patch.set_facecolor("black")
    ax_abs = axes[0]
    ax_abs.set_facecolor("black")
    vmin_abs, vmax_abs = np.min(abs_vals_subset), np.max(abs_vals_subset)
    im_abs, _ = mne.viz.plot_topomap(
        abs_vals_subset, info_clean, axes=ax_abs, show=False, cmap=cmap,
        vlim=(vmin_abs, vmax_abs), names=info_clean["ch_names"] if show_ch_names else None
    )
    ax_abs.set_title(f"{band_name} Abs Power ({cond_name})", color="white", fontsize=10)
    cbar_abs = plt.colorbar(im_abs, ax=ax_abs, orientation="horizontal", fraction=0.05, pad=0.08)
    cbar_abs.set_label("µV²" if not normalize else "Normalized", color="white")
    cbar_abs.ax.tick_params(colors="white")
    ax_rel = axes[1]
    ax_rel.set_facecolor("black")
    vmin_rel, vmax_rel = np.min(rel_vals_subset), np.max(rel_vals_subset)
    im_rel, _ = mne.viz.plot_topomap(
        rel_vals_subset, info_clean, axes=ax_rel, show=False, cmap=cmap,
        vlim=(vmin_rel, vmax_rel), names=info_clean["ch_names"] if show_ch_names else None
    )
    ax_rel.set_title(f"{band_name} Rel Power ({cond_name})", color="white", fontsize=10)
    cbar_rel = plt.colorbar(im_rel, ax=ax_rel, orientation="horizontal", fraction=0.05, pad=0.08)
    cbar_rel.set_label("%" if not normalize else "Normalized", color="white")
    cbar_rel.ax.tick_params(colors="white")
    if instability_subset is not None:
        ax_inst = axes[2]
        ax_inst.set_facecolor("black")
        vmin_inst, vmax_inst = np.min(instability_subset), np.max(instability_subset)
        im_inst, _ = mne.viz.plot_topomap(
            instability_subset, info_clean, axes=ax_inst, show=False, cmap=cmap_instability,
            vlim=(vmin_inst, vmax_inst), names=info_clean["ch_names"] if show_ch_names else None
        )
        ax_inst.set_title(f"{band_name} Instability ({cond_name})", color="white", fontsize=10)
        cbar_inst = plt.colorbar(im_inst, ax=ax_inst, orientation="horizontal", fraction=0.05, pad=0.08)
        cbar_inst.set_label("Variance (µV²)" if not normalize else "Normalized", color="white")
        cbar_inst.ax.tick_params(colors="white")
    fig.suptitle(f"Topomaps (Abs, Rel, Instability) - {band_name} ({cond_name})", color="white", fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"Saved topomap for {band_name} ({cond_name}) to {output_path}")

def plot_band_psd_overlay(
    sig1: np.ndarray,
    sig2: np.ndarray,
    sfreq: float,
    band: Tuple[float, float],
    ch_name: str,
    band_name: str,
    colors: Tuple[str, str] = ("cyan", "magenta"),
    figsize: Tuple[float, float] = (8, 4),
) -> plt.Figure:
    fmin, fmax = band
    freqs, psd1 = welch(sig1, fs=sfreq, nperseg=int(sfreq * 2), noverlap=int(sfreq))
    _, psd2 = welch(sig2, fs=sfreq, nperseg=int(sfreq * 2), noverlap=int(sfreq))
    mask = (freqs >= fmin) & (freqs <= fmax)
    freqs_band = freqs[mask]
    psd1_band = psd1[mask]
    psd2_band = psd2[mask]
    fig, ax = plt.subplots(figsize=figsize, facecolor="black")
    ax.plot(freqs_band, psd1_band, color=colors[0], label="EO")
    ax.plot(freqs_band, psd2_band, color=colors[1], label="EC")
    ax.set_title(f"{ch_name} {band_name} PSD Overlay", color="white", fontsize=12)
    ax.set_xlabel("Frequency (Hz)", color="white")
    ax.set_ylabel("Power (µV²/Hz)", color="white")
    ax.legend(facecolor="black", edgecolor="white", labelcolor="white")
    ax.set_facecolor("black")
    ax.tick_params(colors="white")
    ax.grid(True, color="gray", linestyle="--", alpha=0.5)
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

def generate_full_site_plots(raw_eo: mne.io.Raw, raw_ec: mne.io.Raw, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    channels = raw_eo.ch_names
    sfreq = raw_eo.info["sfreq"]
    for ch in channels:
        if ch not in raw_ec.ch_names or ch not in raw_eo.ch_names:
            with open("missing_channels_log.txt", "a") as f:
                f.write(f"{ch} missing in EO or EC for plotting\n")
            continue
        ch_folder = output_dir / ch
        psd_folder = ch_folder / "PSD_Overlay"
        wave_folder = ch_folder / "Waveform_Overlay"
        diff_folder = ch_folder / "Difference"
        psd_folder.mkdir(parents=True, exist_ok=True)
        wave_folder.mkdir(parents=True, exist_ok=True)
        diff_folder.mkdir(parents=True, exist_ok=True)
        eo_sig = raw_eo.get_data(picks=[ch])[0] * 1e6
        ec_sig = raw_ec.get_data(picks=[ch])[0] * 1e6
        for band_name, band_range in BANDS.items():
            fig_psd = plot_band_psd_overlay(eo_sig, ec_sig, sfreq, band_range, ch, band_name, colors=("cyan", "magenta"))
            psd_path = psd_folder / f"{ch}_PSD_{band_name}.png"
            fig_psd.savefig(psd_path, facecolor="black")
            plt.close(fig_psd)
            fig_wave = plot_band_waveform_overlay(eo_sig, ec_sig, sfreq, band_range, ch, band_name, colors=("cyan", "magenta"), epoch_length=10)
            wave_path = wave_folder / f"{ch}_Waveform_{band_name}.png"
            fig_wave.savefig(wave_path, facecolor="black")
            plt.close(fig_wave)
            power_eo = compute_band_power(eo_sig, sfreq, band_range)
            power_ec = compute_band_power(ec_sig, sfreq, band_range)
            fig_diff, ax = plt.subplots(figsize=(4, 4), facecolor="black")
            ax.bar(["EO", "EC"], [power_eo, power_ec], color=["cyan", "magenta"])
            ax.set_title(f"{ch} {band_name} Difference", color="white", fontsize=10)
            ax.set_ylabel("Power (µV²)", color="white")
            ax.tick_params(colors="white")
            ax.set_facecolor("black")
            fig_diff.tight_layout()
            diff_path = diff_folder / f"{ch}_Difference_{band_name}.png"
            fig_diff.savefig(diff_path, facecolor="black")
            plt.close(fig_diff)

def _generate_clinical_report(raw_eo: mne.io.Raw, raw_ec: mne.io.Raw, output_dir: Path, channels: List[str] = None, source_localization: dict = None, coherence: dict = None, zscores: dict = None, tfr: dict = None, ica: dict = None) -> None:
    """
    Generate a clinical report for EEG data.

    Args:
        raw_eo (mne.io.Raw): Raw EEG data for eyes-open condition
        raw_ec (mne.io.Raw): Raw EEG data for eyes-closed condition
        output_dir (Path): Directory to save the report files
        channels (List[str], optional): List of channel names
        source_localization (dict, optional): Dictionary of source localization results
        coherence (dict, optional): Dictionary of coherence results
        zscores (dict, optional): Dictionary of z-scores
        tfr (dict, optional): Dictionary of time-frequency results
        ica (dict, optional): Dictionary of ICA results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    bp_EO = compute_all_band_powers(raw_eo)
    bp_EC = compute_all_band_powers(raw_ec)
    site_metrics, global_metrics = compute_site_metrics(raw_eo, raw_ec, bp_EO, bp_EC)
    save_site_metrics(site_metrics, global_metrics, output_dir / "clinical_metrics.csv")
    interpretations = interpret_metrics(site_metrics, global_metrics, bp_EO, bp_EC)
    with open(output_dir / "clinical_interpretations.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(interpretations))
    print(f"Clinical interpretations saved to: {output_dir / 'clinical_interpretations.txt'}")
    pyramid_mappings = pyramid_model.map_to_pyramid(bp_EO, bp_EC, site_metrics, global_metrics)
    with open(output_dir / "pyramid_mappings.txt", "w", encoding="utf-8") as f:
        for mapping in pyramid_mappings:
            f.write(f"{mapping}\n")
    print(f"Pyramid mappings saved to: {output_dir / 'pyramid_mappings.txt'}")
    instability_EO = compute_instability_index(raw_eo, BANDS)
    instability_EC = compute_instability_index(raw_ec, BANDS)
    for band_name, band_range in BANDS.items():
        abs_powers_EO = np.array([bp_EO[ch][band_name] for ch in raw_eo.ch_names])
        total_powers_EO = np.array([sum(bp_EO[ch].values()) for ch in raw_eo.ch_names])
        rel_powers_EO = np.zeros_like(abs_powers_EO)
        mask_EO = total_powers_EO > 0
        rel_powers_EO[mask_EO] = (abs_powers_EO[mask_EO] / total_powers_EO[mask_EO]) * 100
        abs_powers_EC = np.array([bp_EC[ch][band_name] for ch in raw_eo.ch_names])
        total_powers_EC = np.array([sum(bp_EC[ch].values()) for ch in raw_eo.ch_names])
        rel_powers_EC = np.zeros_like(abs_powers_EC)
        mask_EC = total_powers_EC > 0
        rel_powers_EC[mask_EC] = (abs_powers_EC[mask_EC] / total_powers_EC[mask_EC]) * 100
        plot_topomap_abs_rel(
            abs_powers_EO, rel_powers_EO, raw_eo, band_name, "EO",
            output_dir / "topomaps" / f"topomap_{band_name}_EO.png",
            instability_vals=np.array([instability_EO[band_name][ch] for ch in raw_eo.ch_names])
        )
        plot_topomap_abs_rel(
            abs_powers_EC, rel_powers_EC, raw_ec, band_name, "EC",
            output_dir / "topomaps" / f"topomap_{band_name}_EC.png",
            instability_vals=np.array([instability_EC[band_name][ch] for ch in raw_eo.ch_names])
        )
    generate_full_site_plots(raw_eo, raw_ec, output_dir / "per_site_plots")
    band_powers = {"EO": bp_EO, "EC": bp_EC}
    instability_indices = {"EO": instability_EO, "EC": instability_EC}
    vigilance_plots = {}
    if raw_eo:
        vigilance_plots["EO"] = {
            "hypnogram": output_dir / "vigilance_hypnogram_EO.png",
            "strip": output_dir / "vigilance_strip_EO.png"
        }
    if raw_ec:
        vigilance_plots["EC"] = {
            "hypnogram": output_dir / "vigilance_hypnogram_EC.png",
            "strip": output_dir / "vigilance_strip_EC.png"
        }
    pdf_report_builder.build_pdf_report(
        report_output_dir=output_dir,
        band_powers=band_powers,
        instability_indices=instability_indices,
        source_localization=source_localization,
        vigilance_plots=vigilance_plots,
        channels=channels  # Pass the channels list
    )
    print(f"PDF report generated at: {output_dir / 'clinical_report.pdf'}")

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
