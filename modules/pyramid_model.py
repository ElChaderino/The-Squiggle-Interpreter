#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pyramid Model — Differentiated & Vigilance-Enriched Clinical Mapping Module

This module contains comprehensive mappings for:
  • Refined Clinical Mapping (Pyramid Levels 1–5)
  • EEG Connectivity Mapping (with differentiators, cognitive/behavioral signatures, vigilance stage, and neurofeedback targets)
  • Vigilance Transition Logic (diagnostic utility)
  • EEG-Based Condition Differentiation (e.g., AuDHD, ADHD, ASD, Anxiety, Sleep Problems, Depression, Schizophrenia, OCD, TBI, PTSD)
  • Optional Live Use / Integration Suggestions

In addition to the static dictionaries below, dynamic helper functions are provided.
For example, the function suggest_pyramid_level() accepts a dictionary of computed metrics
(from your EDF data processing) and returns a recommended pyramid level along with a summary.

Ensure this file is saved with UTF-8 encoding so that emojis (🟢, 🟡, 🟠, 🔴, ⚫, etc.) display correctly.
"""

# -------------------- Section 1: Refined Clinical Mapping --------------------
REFINED_CLINICAL_MAPPING = {
    1: {
        "level_name": "🟢 Level 1: Optimal",
        "eeg_patterns": [
            "✅ PAF 9.5–10.5 Hz",
            "✅ Theta/Beta ≈ 1.5–2.0",
            "✅ Coherence: Normative Range"
        ],
        "cognitive_behavior": "Efficient working memory, executive function, stress resilience",
        "protocols": "Maintenance only, or peak performance (SMR training at Cz; alpha uptraining at Pz)"
    },
    2: {
        "level_name": "🟡 Level 2: Mild Deviation",
        "eeg_patterns": [
            "🔼 Theta/Beta 2.2–2.9",
            "🔼 Alpha/Theta imbalance"
        ],
        "cognitive_behavior": "Light attentional drift, mild anxiety, early insomnia signs",
        "protocols": "SMR training, Alpha/theta balance (Pz), heart rate variability integration"
    },
    3: {
        "level_name": "🟠 Level 3: Moderate Disruption",
        "eeg_patterns": [
            "🔼 Theta/Beta >3.0",
            "🔽 Hypocoherence interhemispheric",
            "🔼 High Beta"
        ],
        "cognitive_behavior": "ADHD symptoms, anxiety/PTSD profile, rumination",
        "protocols": "Focused down-training (Hi-Beta at Fz/Cz), Interhemispheric coherence training (e.g., F3–F4)"
    },
    4: {
        "level_name": "🔴 Level 4: Severe Dysregulation",
        "eeg_patterns": [
            "🔼 Theta/Beta >3.5",
            "🔽 PAF <8 Hz",
            "❌ Fronto-parietal disconnectivity"
        ],
        "cognitive_behavior": "Severe executive dysfunction, cognitive decline (early dementia), emotional instability",
        "protocols": "Pz Alpha reinforcement, inter-lobe coherence repair, memory training protocols"
    },
    5: {
        "level_name": "⚫ Level 5: Pathological",
        "eeg_patterns": [
            "🔼 Delta in wake",
            "❌ Z-scores ≥ ±3.0 globally",
            "❌ Multiband desynchronization"
        ],
        "cognitive_behavior": "Stroke recovery, severe dementia, TBI, epilepsy, vegetative states",
        "protocols": "Passive neurostimulation (pEMF, tDCS adjunct), delta suppression biofeedback (if trainable), caregiver-assisted interventions"
    }
}

# -------------------- Section 2: EEG Connectivity Mapping --------------------
EEG_CONNECTIVITY_MAPPING = {
    1: {
        "level_name": "🟢 Level 1: Optimal Connectivity",
        "eeg_patterns": [
            "✅ PAF 9.5–10.5 Hz",
            "✅ Theta/Beta ≈ 1.5–2.0",
            "✅ Normative coherence + phase synchrony"
        ],
        "differentiators": "High posterior alpha power, dominant alpha rhythm with EO stability, minimal frontal theta",
        "cognition_behavior": ("Flow state, deep focus, executive capacity, creativity, rapid task switching; "
                               "calm but alert, fluid language output, high adaptability under cognitive load"),
        "vigilance_stage": "🟦 A1 – Posterior Alpha Dominant (stable eyes-open rhythm)",
        "neurofeedback_targets": [
            "SMR training at Cz",
            "Alpha uptraining at Pz/POz",
            "Optional: Peak states with Alpha/Theta crossover at Pz"
        ]
    },
    2: {
        "level_name": "🟡 Level 2: Mild EEG Deviation",
        "eeg_patterns": [
            "🔼 Theta/Beta: 2.2–2.9",
            "🔼 Mild Alpha/Theta imbalance",
            "📉 Slight anterior drift of alpha"
        ],
        "differentiators": "Less stable alpha, more frontal drift, occasional vigilance stage slipping, frontal Theta spikes",
        "cognition_behavior": "Daydreaming, momentary attention drops, mild anxiety, shallow sleep complaints",
        "vigilance_stage": "🟩 A2/A3 – Anterior alpha emerging or dominant",
        "neurofeedback_targets": [
            "SMR uptraining at Cz",
            "Alpha stabilization at Pz",
            "HRV or GSR feedback integration",
            "Breath pacing / mindfulness overlay"
        ]
    },
    3: {
        "level_name": "🟠 Level 3: Moderate Disruption",
        "eeg_patterns": [
            "🔼 Theta/Beta >3.0",
            "🔽 Hypocoherence (e.g., F3↔F4)",
            "🔼 Frontal/central High Beta >22 Hz"
        ],
        "differentiators": "Dominant anterior theta with high beta co-activation, alpha dropout, dyssynchronous hemispheres",
        "cognition_behavior": "ADHD-type inattention, PTSD-like hypervigilance, poor sleep onset, low working memory",
        "vigilance_stage": "🟨 B1/B2 – Alpha dropouts; frontal theta rising",
        "neurofeedback_targets": [
            "HiBeta downtraining at Fz/Cz",
            "Interhemispheric coherence training (F3↔F4, C3↔C4)",
            "Theta downtraining at Fz if excessive",
            "Mindfulness-based focus training",
            "Optional adjunct: Flash biofeedback, Audio entrainment (10 Hz posterior)"
        ]
    },
    4: {
        "level_name": "🔴 Level 4: Severe Dysregulation",
        "eeg_patterns": [
            "🔼 Theta/Beta >3.5",
            "🔽 PAF <8 Hz",
            "❌ Fronto-parietal hypocoherence"
        ],
        "differentiators": "Loss of alpha peak, flattening of rhythm, emergence of slow delta, cognitive slowing",
        "cognition_behavior": "Executive dysfunction, emotional lability, early dementia signs, planning errors",
        "vigilance_stage": "🟥 B3 – Prominent theta, little alpha, start of sleep onset indicators",
        "neurofeedback_targets": [
            "Alpha restoration at Pz (train up 9–10 Hz)",
            "PAF nudging via auditory/visual entrainment",
            "Fronto-parietal coherence training (Fz↔Pz)",
            "Memory task-linked training",
            "Adjunct: Nootropics, tACS/tDCS paired with protocol"
        ]
    },
    5: {
        "level_name": "⚫ Level 5: Pathological / Degenerative",
        "eeg_patterns": [
            "🔼 Delta (1–4 Hz) dominant during wake",
            "❌ Z-scores ≥ ±3.0 globally",
            "❌ Loss of synchrony across all bands"
        ],
        "differentiators": "Global slowing, high variability, mixed amplitude disturbances, suppressed alpha",
        "cognition_behavior": "Severe impairment (e.g., late-stage dementia, vegetative state, TBI coma), loss of responsiveness",
        "vigilance_stage": "🟪 C – Sleep onset / cortical disengagement",
        "neurofeedback_targets": [
            "Delta suppression (if volitional)",
            "Alpha induction (posterior or auditory cues)",
            "Caregiver-assisted feedback loops",
            "Passive protocols: pEMF, vibrotactile entrainment",
            "Adjunct: Photobiomodulation (810 nm), rhythmic music therapy, very slow HRV entrainment"
        ]
    }
}

# -------------------- Section 3: Vigilance Transition Logic --------------------
VIGILANCE_TRANSITION_LOGIC = {
    "A1": {
        "eeg_signature": "Posterior alpha (Pz, O1/O2), stable EO rhythm",
        "clinical_meaning": "High vigilance, calm alert",
        "possible_response": "None needed or peak training"
    },
    "A2": {
        "eeg_signature": "Anterior alpha creeps forward",
        "clinical_meaning": "Early vigilance drop",
        "possible_response": "Rebalance alpha at Pz"
    },
    "A3": {
        "eeg_signature": "Anterior alpha dominant",
        "clinical_meaning": "Reduction in alertness",
        "possible_response": "Boost posterior rhythm, SMR training"
    },
    "B1": {
        "eeg_signature": "Alpha dropout begins",
        "clinical_meaning": "Stress emerging, reduced top-down control",
        "possible_response": "Introduce calming training, breathwork"
    },
    "B2": {
        "eeg_signature": "Frontal theta appears",
        "clinical_meaning": "Loss of attention, emotional lability",
        "possible_response": "Theta suppression, enhance frontal coherence"
    },
    "B3": {
        "eeg_signature": "Dominant theta, near sleep onset",
        "clinical_meaning": "Deep hypoarousal or trauma looping",
        "possible_response": "Downtrain theta, re-engage with SMR"
    },
    "C": {
        "eeg_signature": "Spindles/delta or disengagement",
        "clinical_meaning": "Sleep onset or pathological disconnection",
        "possible_response": "Monitor; consider low-frequency neuromodulation"
    }
}

# -------------------- Section 4: EEG-Based Condition Differentiation --------------------
CONDITION_DIFFERENTIATION = {
    "AuDHD": {
        "theta_frequencies": {"low": "1.6 Hz", "mid": "5.6-5.8 Hz"},
        "alpha_frequencies": "Persistent low alpha at 7.5 Hz",
        "interpretation": (
            "Fluctuation between low theta (1.6 Hz) and mid-theta (5.6-5.8 Hz) with persistent low alpha "
            "suggests combined inattentiveness and internal disengagement typical of AuDHD.")
    },
    "ADHD": {
        "theta_frequencies": {"low": "1.6 Hz", "mid": "5.6-5.8 Hz"},
        "alpha_frequencies": "Unstable, lacking a dominant frequency",
        "interpretation": "Dominant mid-theta with absent or unstable alpha indicates classic ADHD inattention."
    },
    "ASD": {
        "theta_frequencies": "Approximately 7.5 Hz",
        "alpha_frequencies": "Persistent low alpha (7.5 Hz)",
        "interpretation": ("A consistent 7.5 Hz in both theta and alpha ranges points to internal over-processing "
                           "and difficulty shifting focus, common in ASD.")
    },
    "Anxiety": {
        "theta_frequencies": "Decreased theta (4–7 Hz)",
        "alpha_frequencies": "Suppressed alpha (8–12 Hz)",
        "beta_frequencies": "Elevated high beta (20–30 Hz)",
        "interpretation": "Suppressed alpha with elevated beta signals hyper-arousal and chronic anxiety."
    },
    "Sleep_Problems": {
        "theta_frequencies": "Disrupted theta patterns",
        "alpha_frequencies": "Persistent alpha during eyes-closed rest",
        "beta_frequencies": "Elevated low beta (12–18 Hz)",
        "interpretation": "Inability to downshift from active brain states impairs sleep onset and continuity."
    },
    "Depression": {
        "theta_frequencies": "Elevated low theta (4–6 Hz)",
        "alpha_frequencies": "Frontal alpha asymmetry (left > right)",
        "beta_frequencies": "Decreased low beta (12–15 Hz)",
        "interpretation": "Elevated theta with frontal alpha imbalance indicates cognitive slowing and withdrawal."
    },
    "Schizophrenia": {
        "theta_frequencies": "Elevated low and mid-theta (3–6 Hz)",
        "alpha_frequencies": "Fragmented, unstable alpha",
        "beta_frequencies": "Erratic beta activity",
        "interpretation": "Combined theta elevation and disrupted alpha reflect cognitive disorganization."
    },
    "OCD": {
        "theta_frequencies": "Elevated high theta (6–8 Hz)",
        "alpha_frequencies": "Reduced alpha",
        "beta_frequencies": "Elevated high beta (20–30 Hz)",
        "interpretation": "High theta and beta with low alpha indicate repetitive, rigid thought patterns."
    },
    "TBI": {
        "theta_frequencies": "Excessive slow theta (3–5 Hz)",
        "alpha_frequencies": "Suppressed alpha",
        "beta_frequencies": "Reduced low beta (12–18 Hz) with occasional bursts",
        "interpretation": "Excessive slow theta with reduced beta points to post-traumatic cognitive slowing."
    },
    "PTSD": {
        "theta_frequencies": "Elevated high theta (5–7 Hz)",
        "alpha_frequencies": "Suppressed alpha",
        "beta_frequencies": "Elevated high beta (20–30 Hz)",
        "interpretation": "High theta and beta with suppressed alpha are markers of hypervigilance and trauma."
    }
}

# -------------------- Section 5: Live Use / Integration Suggestions --------------------
LIVE_USE_SUGGESTIONS = {
    "color_badge_system": "Implement a color-coded badge system for vigilance (e.g., green = A1, purple = C)",
    "pop_up_interventions": "Trigger pop-up interventions during vigilance dips (e.g., HRV cue, sound cue)"
}


# -------------------- Dynamic Helper Functions --------------------
def suggest_pyramid_level(metrics):
    """
    Evaluate computed metrics from EDF data and suggest a pyramid level.
    The 'metrics' dictionary is expected to include keys like:
       - 'theta_beta_ratio': average Theta/Beta ratio (float)
       - 'alpha_change': percentage change in Alpha power (EO→EC) (float)
       - 'vigilance_index': an optional measure of vigilance (float)

    This function uses example threshold logic:
       - If theta_beta_ratio is between 1.5 and 2.0 and alpha_change is moderate, suggest Level 1.
       - If theta_beta_ratio is higher, suggest higher levels accordingly.

    Returns a tuple: (level, mapping, summary)
    """
    tb = metrics.get("theta_beta_ratio")
    alpha_change = metrics.get("alpha_change")
    summary = "Metrics: "
    if tb is not None:
        summary += f"Theta/Beta Ratio = {tb:.2f}. "
    if alpha_change is not None:
        summary += f"Alpha Change = {alpha_change:.1f}%. "

    # Example logic – adjust thresholds as needed:
    if tb is None or alpha_change is None:
        level = 1
    elif 1.5 <= tb <= 2.0 and 20 <= alpha_change <= 40:
        level = 1
    elif 2.0 < tb <= 2.9:
        level = 2
    elif 3.0 <= tb <= 3.5:
        level = 3
    elif 3.5 < tb <= 4.5:
        level = 4
    elif tb > 4.5:
        level = 5
    else:
        level = 1  # default

    mapping = REFINED_CLINICAL_MAPPING.get(level, {})
    summary += f"Suggested Level: {level} ({mapping.get('level_name', 'N/A')})."
    return level, mapping, summary


def get_refined_mapping(level):
    """Retrieve refined clinical mapping for a given pyramid level (1–5)."""
    return REFINED_CLINICAL_MAPPING.get(level)


def get_connectivity_mapping(level):
    """Retrieve EEG connectivity mapping for a given pyramid level (1–5)."""
    return EEG_CONNECTIVITY_MAPPING.get(level)


def get_vigilance_logic(stage):
    """Retrieve vigilance transition logic for a given stage (e.g., 'A1', 'B2')."""
    return VIGILANCE_TRANSITION_LOGIC.get(stage.upper())


def get_condition_differentiation(condition):
    """Retrieve EEG condition differentiation details for a given condition (e.g., 'ADHD', 'ASD')."""
    return CONDITION_DIFFERENTIATION.get(condition)


def list_all_refined_mappings():
    """Return all refined clinical mappings as a sorted list of tuples (level, mapping)."""
    return sorted(REFINED_CLINICAL_MAPPING.items())


def list_all_connectivity_mappings():
    """Return all EEG connectivity mappings as a sorted list of tuples (level, mapping)."""
    return sorted(EEG_CONNECTIVITY_MAPPING.items())


def map_to_pyramid(bp_EO: dict, bp_EC: dict, site_metrics: dict, global_metrics: dict) -> list[str]:
    """
    Maps EEG metrics to a pyramid model for clinical interpretation using refined clinical and connectivity mappings.

    Args:
        bp_EO (dict): Band powers for eyes-open condition (channel -> band -> power)
        bp_EC (dict): Band powers for eyes-closed condition (channel -> band -> power)
        site_metrics (dict): Site-specific metrics (channel -> metric -> value)
        global_metrics (dict): Global metrics (metric -> value)

    Returns:
        list[str]: List of strings representing pyramid model mappings
    """
    mappings = []

    # Compute average Theta/Beta Ratio and Alpha Change for pyramid level suggestion
    tb_ratios = [metrics["Theta_Beta_Ratio"] for metrics in site_metrics.values() if "Theta_Beta_Ratio" in metrics]
    avg_tb_ratio = sum(tb_ratios) / len(tb_ratios) if tb_ratios else None

    alpha_changes = []
    for channel in bp_EO.keys():
        if channel in bp_EC:
            alpha_EO = bp_EO[channel].get("Alpha", 0)
            alpha_EC = bp_EC[channel].get("Alpha", 0)
            alpha_change = ((alpha_EC - alpha_EO) / alpha_EO * 100) if alpha_EO != 0 else 0
            alpha_changes.append(alpha_change)
    avg_alpha_change = sum(alpha_changes) / len(alpha_changes) if alpha_changes else None

    # Suggest pyramid level based on metrics
    metrics = {
        "theta_beta_ratio": avg_tb_ratio,
        "alpha_change": avg_alpha_change
    }
    level, mapping, summary = suggest_pyramid_level(metrics)
    mappings.append(summary)

    # Add refined clinical mapping details
    refined_mapping = get_refined_mapping(level)
    if refined_mapping:
        mappings.append(f"Refined Clinical Mapping: {refined_mapping['level_name']}")
        mappings.append("  EEG Patterns: " + ", ".join(refined_mapping["eeg_patterns"]))
        mappings.append("  Cognitive/Behavioral Implications: " + refined_mapping["cognitive_behavior"])
        mappings.append("  Protocols: " + refined_mapping["protocols"])

    # Add connectivity mapping details
    connectivity_mapping = get_connectivity_mapping(level)
    if connectivity_mapping:
        mappings.append(f"EEG Connectivity Mapping: {connectivity_mapping['level_name']}")
        mappings.append("  EEG Patterns: " + ", ".join(connectivity_mapping["eeg_patterns"]))
        mappings.append("  Differentiators: " + connectivity_mapping["differentiators"])
        mappings.append("  Cognitive/Behavioral Implications: " + connectivity_mapping["cognition_behavior"])
        mappings.append("  Vigilance Stage: " + connectivity_mapping["vigilance_stage"])
        mappings.append("  Neurofeedback Targets: " + "; ".join(connectivity_mapping["neurofeedback_targets"]))

    # Add specific metric mappings
    for channel in bp_EO.keys():
        if channel in bp_EC:
            alpha_EO = bp_EO[channel].get("Alpha", 0)
            alpha_EC = bp_EC[channel].get("Alpha", 0)
            alpha_change = ((alpha_EC - alpha_EO) / alpha_EO * 100) if alpha_EO != 0 else 0
            mappings.append(f"Channel {channel}: Alpha Change (EO->EC) = {alpha_change:.2f}%")

    if "Frontal_Asymmetry" in global_metrics:
        fa = global_metrics["Frontal_Asymmetry"]
        mappings.append(f"Global: Frontal Asymmetry (F4/F3 Alpha, EO) = {fa:.2f}")

    for channel, metrics in site_metrics.items():
        if "Theta_Beta_Ratio" in metrics:
            tbr = metrics["Theta_Beta_Ratio"]
            mappings.append(f"Channel {channel}: Theta/Beta Ratio (EO) = {tbr:.2f}")

    # If no mappings are generated, add a default message
    if not mappings:
        mappings.append("No pyramid mappings generated: Insufficient data")

    return mappings


if __name__ == "__main__":
    # Example dynamic usage: simulate metrics from EDF data.
    sample_metrics = {
        "theta_beta_ratio": 2.5,
        "alpha_change": 30.0
    }
    level, mapping, summary = suggest_pyramid_level(sample_metrics)
    print("Dynamic Pyramid Level Evaluation:")
    print(summary)

    print("\n=== Refined Clinical Mapping ===")
    for lvl, mapping in list_all_refined_mappings():
        print(f"{mapping['level_name']}")
        print("  EEG Patterns:", ", ".join(mapping["eeg_patterns"]))
        print("  Cognitive/Behavioral Implications:", mapping["cognitive_behavior"])
        print("  Protocols:", mapping["protocols"])
        print()

    print("\n=== EEG Connectivity Mapping ===")
    for lvl, mapping in list_all_connectivity_mappings():
        print(f"{mapping['level_name']}")
        print("  EEG Patterns:", ", ".join(mapping["eeg_patterns"]))
        print("  Differentiators:", mapping["differentiators"])
        print("  Cognitive/Behavioral Implications:", mapping["cognition_behavior"])
        print("  Vigilance Stage:", mapping["vigilance_stage"])
        print("  Neurofeedback Targets:", "; ".join(mapping["neurofeedback_targets"]))
        print()

    # Example: Retrieve and display vigilance transition logic for stage A2.
    stage = "A2"
    logic = get_vigilance_logic(stage)
    if logic:
        print(f"Vigilance Transition Logic for Stage {stage}:")
        print("  EEG Signature:", logic["eeg_signature"])
        print("  Clinical Meaning:", logic["clinical_meaning"])
        print("  Possible Response:", logic["possible_response"])

    # Example: Retrieve condition differentiation for ADHD.
    condition = "ADHD"
    cond_info = get_condition_differentiation(condition)
    if cond_info:
        print(f"\nCondition Differentiation for {condition}:")
        print("  Theta Frequencies:", cond_info.get("theta_frequencies", "N/A"))
        print("  Alpha Frequencies:", cond_info.get("alpha_frequencies", "N/A"))
        print("  Beta Frequencies:", cond_info.get("beta_frequencies", "N/A"))
        print("  Interpretation:", cond_info.get("interpretation", ""))

    # Print live use suggestions.
    print("\nLive Use / Integration Suggestions:")
    for key, suggestion in LIVE_USE_SUGGESTIONS.items():
        print(f"  {key}: {suggestion}")
