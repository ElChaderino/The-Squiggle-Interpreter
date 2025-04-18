#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pyramid Model — Differentiated & Vigilance-Enriched Clinical Mapping Module

This module provides a comprehensive framework for mapping EEG data to clinical interpretations, 
pyramid levels, and neurofeedback protocols. It includes:

  • Refined Clinical Mapping (Pyramid Levels 1–5): Detailed EEG patterns, cognitive/behavioral implications, and protocols.
  • EEG Connectivity Mapping: Includes differentiators, cognitive/behavioral signatures, vigilance stages, and neurofeedback targets.
  • Vigilance Transition Logic: Diagnostic utility with transition probabilities and actionable responses.
  • EEG-Based Condition Differentiation: Covers a wide range of conditions (e.g., AuDHD, ADHD, ASD, Anxiety, etc.) with detailed EEG signatures.
  • Live Use / Integration Suggestions: Real-time feedback mechanisms and neurofeedback system integrations.

Dynamic helper functions are provided to map EEG metrics to pyramid levels, suggest interventions, and support live clinical use.
For example, `suggest_pyramid_level()` uses computed metrics to recommend a pyramid level with a detailed summary.

Ensure this file is saved with UTF-8 encoding to display emojis (🟢, 🟡, 🟠, 🔴, ⚫, etc.) correctly.
"""

import logging
from typing import Dict, List, Tuple, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)

# -------------------- Section 1: Refined Clinical Mapping --------------------
REFINED_CLINICAL_MAPPING = {
    1: {
        "level_name": "🟢 Level 1: Optimal",
        "eeg_patterns": [
            "✅ PAF: 9.5–10.5 Hz",
            "✅ Theta/Beta Ratio: 1.5–2.0",
            "✅ Coherence: Normative (within 1 SD)",
            "✅ Alpha Symmetry: Balanced (F3/F4 ratio 0.9–1.1)",
            "✅ Delta Power: Low during wake (<10% total power)"
        ],
        "cognitive_behavior": (
            "Efficient working memory, strong executive function, high stress resilience, "
            "optimal emotional regulation, and peak cognitive performance"
        ),
        "clinical_implications": "No clinical intervention required; suitable for peak performance training",
        "protocols": [
            "Maintenance: SMR training at Cz (12–15 Hz)",
            "Peak Performance: Alpha uptraining at Pz (8–12 Hz)",
            "Optional: Alpha/Theta crossover training at Pz for creativity enhancement"
        ]
    },
    2: {
        "level_name": "🟡 Level 2: Mild Deviation",
        "eeg_patterns": [
            "🔼 Theta/Beta Ratio: 2.2–2.9",
            "🔼 Alpha/Theta Imbalance: Theta > Alpha at Pz",
            "📉 Alpha Peak: Slight anterior drift (Fz/Pz ratio > 1.2)",
            "🔼 Low Beta: Mild elevation (12–18 Hz, >15% total power)",
            "✅ Coherence: Still within normative range"
        ],
        "cognitive_behavior": (
            "Mild attentional drift, occasional daydreaming, early signs of anxiety, "
            "shallow sleep onset difficulties, and slight emotional reactivity"
        ),
        "clinical_implications": "Early intervention recommended to prevent progression",
        "protocols": [
            "SMR training at Cz (12–15 Hz) to enhance focus",
            "Alpha/Theta balance training at Pz (downtrain Theta, uptrain Alpha)",
            "Heart rate variability (HRV) integration for stress reduction",
            "Mindfulness-based biofeedback (e.g., breath pacing)"
        ]
    },
    3: {
        "level_name": "🟠 Level 3: Moderate Disruption",
        "eeg_patterns": [
            "🔼 Theta/Beta Ratio: 3.0–3.5",
            "🔽 Interhemispheric Coherence: Hypocoherence (F3↔F4, <0.6)",
            "🔼 High Beta: Elevated (20–30 Hz, >20% total power)",
            "📉 Alpha Peak: Reduced (<8 Hz) or unstable",
            "🔼 Frontal Theta: Increased (4–8 Hz, >25% total power)"
        ],
        "cognitive_behavior": (
            "ADHD-like symptoms (inattention, impulsivity), anxiety/PTSD profile, "
            "rumination, poor sleep maintenance, and reduced working memory capacity"
        ),
        "clinical_implications": "Moderate intervention required; potential for psychiatric comorbidities",
        "protocols": [
            "High Beta downtraining at Fz/Cz (20–30 Hz)",
            "Interhemispheric coherence training (F3↔F4, C3↔C4)",
            "Theta downtraining at Fz if excessive",
            "Mindfulness-based focus training",
            "Adjunct: Audio-visual entrainment (10 Hz posterior), flash biofeedback"
        ]
    },
    4: {
        "level_name": "🔴 Level 4: Severe Dysregulation",
        "eeg_patterns": [
            "🔼 Theta/Beta Ratio: >3.5",
            "🔽 PAF: <8 Hz",
            "❌ Fronto-Parietal Coherence: Severe hypocoherence (<0.5)",
            "🔼 Delta Power: Increased during wake (>20% total power)",
            "📉 Alpha: Suppressed or absent (<5% total power)"
        ],
        "cognitive_behavior": (
            "Severe executive dysfunction, emotional instability, cognitive decline, "
            "memory impairment, potential early dementia signs, and difficulty with daily tasks"
        ),
        "clinical_implications": "Significant intervention required; neurological evaluation recommended",
        "protocols": [
            "Alpha restoration at Pz (train up 9–10 Hz)",
            "PAF nudging via auditory/visual entrainment (8–10 Hz)",
            "Fronto-parietal coherence training (Fz↔Pz)",
            "Memory task-linked neurofeedback",
            "Adjunct: tACS/tDCS paired with protocol, nootropic support"
        ]
    },
    5: {
        "level_name": "⚫ Level 5: Pathological",
        "eeg_patterns": [
            "🔼 Delta: Dominant during wake (>30% total power)",
            "❌ Z-scores: ≥ ±3.0 globally across multiple bands",
            "❌ Multiband Desynchronization: Loss of coherence across all bands",
            "📉 Alpha: Completely absent",
            "🔼 High Variability: Erratic amplitude across channels"
        ],
        "cognitive_behavior": (
            "Severe cognitive impairment (e.g., late-stage dementia, vegetative state, TBI coma), "
            "loss of responsiveness, minimal executive function, and significant neurological deficits"
        ),
        "clinical_implications": "Critical intervention required; potential for irreversible damage",
        "protocols": [
            "Delta suppression (if volitional) at Fz/Cz",
            "Alpha induction via auditory cues (posterior, 8–10 Hz)",
            "Passive neurostimulation: pEMF, tDCS adjunct",
            "Caregiver-assisted feedback loops",
            "Adjunct: Photobiomodulation (810 nm), rhythmic music therapy, slow HRV entrainment"
        ]
    }
}

# -------------------- Section 2: EEG Connectivity Mapping --------------------
EEG_CONNECTIVITY_MAPPING = {
    1: {
        "level_name": "🟢 Level 1: Optimal Connectivity",
        "eeg_patterns": [
            "✅ PAF: 9.5–10.5 Hz",
            "✅ Theta/Beta Ratio: 1.5–2.0",
            "✅ Coherence: Normative (within 1 SD), phase synchrony high",
            "✅ Alpha Symmetry: Balanced (F3/F4 ratio 0.9–1.1)",
            "✅ Delta Power: Low during wake (<10% total power)"
        ],
        "differentiators": (
            "High posterior alpha power, stable alpha rhythm with EO, minimal frontal theta, "
            "strong interhemispheric coherence (F3↔F4, C3↔C4)"
        ),
        "cognition_behavior": (
            "Flow state, deep focus, excellent executive capacity, creativity, rapid task switching; "
            "calm but alert, fluid language output, high adaptability under cognitive load"
        ),
        "clinical_implications": "Optimal brain function; suitable for peak performance training",
        "vigilance_stage": "🟦 A1 – Posterior Alpha Dominant (stable eyes-open rhythm)",
        "neurofeedback_targets": [
            "SMR training at Cz (12–15 Hz)",
            "Alpha uptraining at Pz/POz (8–12 Hz)",
            "Optional: Alpha/Theta crossover at Pz for creativity",
            "Optional: Coherence training to maintain optimal connectivity"
        ]
    },
    2: {
        "level_name": "🟡 Level 2: Mild EEG Deviation",
        "eeg_patterns": [
            "🔼 Theta/Beta Ratio: 2.2–2.9",
            "🔼 Alpha/Theta Imbalance: Theta > Alpha at Pz",
            "📉 Alpha Peak: Anterior drift (Fz/Pz ratio > 1.2)",
            "🔼 Low Beta: Mild elevation (12–18 Hz, >15% total power)",
            "✅ Coherence: Slightly reduced but within normative range"
        ],
        "differentiators": (
            "Less stable alpha, increased frontal theta spikes, occasional vigilance stage slipping, "
            "mild reduction in interhemispheric coherence"
        ),
        "cognition_behavior": (
            "Daydreaming, momentary attention drops, mild anxiety, shallow sleep complaints, "
            "slight emotional reactivity under stress"
        ),
        "clinical_implications": "Early signs of dysregulation; monitor for progression",
        "vigilance_stage": "🟩 A2/A3 – Anterior alpha emerging or dominant",
        "neurofeedback_targets": [
            "SMR uptraining at Cz (12–15 Hz) to enhance focus",
            "Alpha stabilization at Pz (8–12 Hz)",
            "HRV or GSR feedback integration for stress management",
            "Breath pacing/mindfulness overlay",
            "Optional: Coherence training (F3↔F4) to prevent further decline"
        ]
    },
    3: {
        "level_name": "🟠 Level 3: Moderate Disruption",
        "eeg_patterns": [
            "🔼 Theta/Beta Ratio: 3.0–3.5",
            "🔽 Interhemispheric Coherence: Hypocoherence (F3↔F4, <0.6)",
            "🔼 High Beta: Elevated (20–30 Hz, >20% total power)",
            "📉 Alpha Peak: Reduced (<8 Hz) or unstable",
            "🔼 Frontal Theta: Increased (4–8 Hz, >25% total power)"
        ],
        "differentiators": (
            "Dominant anterior theta with high beta co-activation, alpha dropout, "
            "dyssynchronous hemispheres, reduced fronto-parietal coherence"
        ),
        "cognition_behavior": (
            "ADHD-type inattention, impulsivity, PTSD-like hypervigilance, poor sleep onset, "
            "low working memory, emotional dysregulation"
        ),
        "clinical_implications": "Moderate dysregulation; potential for psychiatric comorbidities",
        "vigilance_stage": "🟨 B1/B2 – Alpha dropouts; frontal theta rising",
        "neurofeedback_targets": [
            "High Beta downtraining at Fz/Cz (20–30 Hz)",
            "Interhemispheric coherence training (F3↔F4, C3↔C4)",
            "Theta downtraining at Fz if excessive",
            "Mindfulness-based focus training",
            "Audio-visual entrainment (10 Hz posterior)",
            "Optional: Flash biofeedback, slow HRV entrainment"
        ]
    },
    4: {
        "level_name": "🔴 Level 4: Severe Dysregulation",
        "eeg_patterns": [
            "🔼 Theta/Beta Ratio: >3.5",
            "🔽 PAF: <8 Hz",
            "❌ Fronto-Parietal Coherence: Severe hypocoherence (<0.5)",
            "🔼 Delta Power: Increased during wake (>20% total power)",
            "📉 Alpha: Suppressed or absent (<5% total power)"
        ],
        "differentiators": (
            "Loss of alpha peak, flattening of rhythm, emergence of slow delta, "
            "severe reduction in inter-lobe coherence, cognitive slowing"
        ),
        "cognition_behavior": (
            "Severe executive dysfunction, emotional lability, early dementia signs, "
            "memory impairment, difficulty with planning and daily tasks"
        ),
        "clinical_implications": "Significant dysregulation; neurological evaluation recommended",
        "vigilance_stage": "🟥 B3 – Prominent theta, little alpha, start of sleep onset indicators",
        "neurofeedback_targets": [
            "Alpha restoration at Pz (train up 9–10 Hz)",
            "PAF nudging via auditory/visual entrainment (8–10 Hz)",
            "Fronto-parietal coherence training (Fz↔Pz)",
            "Memory task-linked neurofeedback",
            "Adjunct: tACS/tDCS paired with protocol, nootropic support"
        ]
    },
    5: {
        "level_name": "⚫ Level 5: Pathological / Degenerative",
        "eeg_patterns": [
            "🔼 Delta: Dominant during wake (>30% total power)",
            "❌ Z-scores: ≥ ±3.0 globally across multiple bands",
            "❌ Multiband Desynchronization: Loss of coherence across all bands",
            "📉 Alpha: Completely absent",
            "🔼 High Variability: Erratic amplitude across channels"
        ],
        "differentiators": (
            "Global slowing, high variability, mixed amplitude disturbances, "
            "suppressed alpha, loss of rhythmic activity"
        ),
        "cognition_behavior": (
            "Severe cognitive impairment (e.g., late-stage dementia, vegetative state, TBI coma), "
            "loss of responsiveness, minimal executive function"
        ),
        "clinical_implications": "Critical condition; potential for irreversible damage",
        "vigilance_stage": "🟪 C – Sleep onset / cortical disengagement",
        "neurofeedback_targets": [
            "Delta suppression (if volitional) at Fz/Cz",
            "Alpha induction via auditory cues (posterior, 8–10 Hz)",
            "Passive neurostimulation: pEMF, tDCS adjunct",
            "Caregiver-assisted feedback loops",
            "Adjunct: Photobiomodulation (810 nm), rhythmic music therapy, slow HRV entrainment"
        ]
    }
}

# -------------------- Section 3: Vigilance Transition Logic --------------------
VIGILANCE_TRANSITION_LOGIC = {
    "A1": {
        "eeg_signature": "Posterior alpha dominant (Pz, O1/O2), stable EO rhythm",
        "clinical_meaning": "High vigilance, calm alert state",
        "possible_response": "None needed; optional peak performance training",
        "transition_probabilities": {
            "A2": 0.2,  # 20% chance of mild vigilance drop
            "A3": 0.05, # 5% chance of anterior alpha dominance
            "B1": 0.01  # 1% chance of alpha dropout
        }
    },
    "A2": {
        "eeg_signature": "Anterior alpha creeps forward (Fz/Pz ratio > 1.2)",
        "clinical_meaning": "Early vigilance drop, slight hypoarousal",
        "possible_response": "Rebalance alpha at Pz, introduce calming exercises",
        "transition_probabilities": {
            "A1": 0.3,  # 30% chance of returning to optimal state
            "A3": 0.4,  # 40% chance of further vigilance drop
            "B1": 0.2   # 20% chance of alpha dropout
        }
    },
    "A3": {
        "eeg_signature": "Anterior alpha dominant (Fz > Pz)",
        "clinical_meaning": "Reduction in alertness, potential for inattention",
        "possible_response": "Boost posterior rhythm (Pz alpha uptraining), SMR training at Cz",
        "transition_probabilities": {
            "A2": 0.3,  # 30% chance of returning to A2
            "B1": 0.5,  # 50% chance of alpha dropout
            "B2": 0.1   # 10% chance of frontal theta emergence
        }
    },
    "B1": {
        "eeg_signature": "Alpha dropout begins (<5% total power)",
        "clinical_meaning": "Stress emerging, reduced top-down control",
        "possible_response": "Introduce calming training, breathwork, alpha restoration",
        "transition_probabilities": {
            "A3": 0.2,  # 20% chance of returning to A3
            "B2": 0.6,  # 60% chance of frontal theta emergence
            "B3": 0.1   # 10% chance of deep hypoarousal
        }
    },
    "B2": {
        "eeg_signature": "Frontal theta appears (4–8 Hz, >25% total power)",
        "clinical_meaning": "Loss of attention, emotional lability, hypoarousal",
        "possible_response": "Theta suppression at Fz, enhance frontal coherence (F3↔F4)",
        "transition_probabilities": {
            "B1": 0.3,  # 30% chance of returning to B1
            "B3": 0.5,  # 50% chance of deeper hypoarousal
            "C": 0.1    # 10% chance of sleep onset
        }
    },
    "B3": {
        "eeg_signature": "Dominant theta (4–8 Hz, >30% total power), near sleep onset",
        "clinical_meaning": "Deep hypoarousal, potential trauma looping",
        "possible_response": "Downtrain theta at Fz, re-engage with SMR at Cz, consider entrainment",
        "transition_probabilities": {
            "B2": 0.2,  # 20% chance of returning to B2
            "C": 0.6    # 60% chance of sleep onset
        }
    },
    "C": {
        "eeg_signature": "Spindles/delta dominant or cortical disengagement",
        "clinical_meaning": "Sleep onset or pathological disconnection",
        "possible_response": "Monitor; consider low-frequency neuromodulation (e.g., tDCS, pEMF)",
        "transition_probabilities": {
            "B3": 0.1   # 10% chance of returning to B3 (if arousal increases)
        }
    }
}

# -------------------- Section 4: EEG-Based Condition Differentiation --------------------
CONDITION_DIFFERENTIATION = {
    "AuDHD": {
        "theta_frequencies": {"low": "1.6 Hz", "mid": "5.6–5.8 Hz"},
        "alpha_frequencies": "Persistent low alpha at 7.5 Hz",
        "beta_frequencies": "Mild elevation in low beta (12–15 Hz)",
        "coherence": "Variable interhemispheric coherence (F3↔F4, fluctuating)",
        "interpretation": (
            "Fluctuation between low theta (1.6 Hz) and mid-theta (5.6–5.8 Hz) with persistent low alpha "
            "suggests combined inattentiveness and internal disengagement typical of AuDHD."
        ),
        "clinical_implications": "Monitor for sensory processing issues; combined ADHD and ASD interventions"
    },
    "ADHD": {
        "theta_frequencies": {"low": "1.6 Hz", "mid": "5.6–5.8 Hz"},
        "alpha_frequencies": "Unstable, lacking a dominant frequency",
        "beta_frequencies": "Reduced low beta (12–15 Hz)",
        "coherence": "Hypocoherence in frontal regions (F3↔F4, <0.6)",
        "interpretation": (
            "Dominant mid-theta with absent or unstable alpha indicates classic ADHD inattention, "
            "often with reduced frontal coherence."
        ),
        "clinical_implications": "Focus on attention and impulse control; potential for stimulant response"
    },
    "ASD": {
        "theta_frequencies": "Approximately 7.5 Hz",
        "alpha_frequencies": "Persistent low alpha (7.5 Hz)",
        "beta_frequencies": "Normal to slightly elevated low beta (12–15 Hz)",
        "coherence": "Reduced interhemispheric coherence (C3↔C4, <0.6)",
        "interpretation": (
            "A consistent 7.5 Hz in both theta and alpha ranges points to internal over-processing "
            "and difficulty shifting focus, common in ASD."
        ),
        "clinical_implications": "Address sensory sensitivities and social engagement; monitor for anxiety"
    },
    "Anxiety": {
        "theta_frequencies": "Decreased theta (4–7 Hz, <15% total power)",
        "alpha_frequencies": "Suppressed alpha (8–12 Hz, <5% total power)",
        "beta_frequencies": "Elevated high beta (20–30 Hz, >20% total power)",
        "coherence": "Hypercoherence in frontal regions (F3↔F4, >0.8)",
        "interpretation": (
            "Suppressed alpha with elevated high beta and frontal hypercoherence signals "
            "hyper-arousal and chronic anxiety."
        ),
        "clinical_implications": "Focus on relaxation and stress reduction; monitor for panic attacks"
    },
    "Sleep_Problems": {
        "theta_frequencies": "Disrupted theta patterns (variable 4–8 Hz)",
        "alpha_frequencies": "Persistent alpha during eyes-closed rest (>10% total power)",
        "beta_frequencies": "Elevated low beta (12–18 Hz, >15% total power)",
        "coherence": "Normal to slightly reduced coherence",
        "interpretation": (
            "Inability to downshift from active brain states (persistent alpha and low beta) "
            "impairs sleep onset and continuity."
        ),
        "clinical_implications": "Address sleep hygiene; consider alpha downtraining for relaxation"
    },
    "Depression": {
        "theta_frequencies": "Elevated low theta (4–6 Hz, >20% total power)",
        "alpha_frequencies": "Frontal alpha asymmetry (left > right, F3/F4 ratio > 1.2)",
        "beta_frequencies": "Decreased low beta (12–15 Hz, <10% total power)",
        "coherence": "Reduced interhemispheric coherence (F3↔F4, <0.6)",
        "interpretation": (
            "Elevated low theta with frontal alpha imbalance indicates cognitive slowing, "
            "withdrawal, and emotional dysregulation typical of depression."
        ),
        "clinical_implications": "Monitor for suicidal ideation; consider SSRI response"
    },
    "Schizophrenia": {
        "theta_frequencies": "Elevated low and mid-theta (3–6 Hz, >25% total power)",
        "alpha_frequencies": "Fragmented, unstable alpha (<5% total power)",
        "beta_frequencies": "Erratic beta activity (variable 12–30 Hz)",
        "coherence": "Severely reduced coherence globally (<0.5)",
        "interpretation": (
            "Combined theta elevation, disrupted alpha, and erratic beta with global desynchronization "
            "reflect cognitive disorganization and perceptual disturbances."
        ),
        "clinical_implications": "Neurological evaluation required; monitor for psychosis"
    },
    "OCD": {
        "theta_frequencies": "Elevated high theta (6–8 Hz, >20% total power)",
        "alpha_frequencies": "Reduced alpha (8–12 Hz, <5% total power)",
        "beta_frequencies": "Elevated high beta (20–30 Hz, >20% total power)",
        "coherence": "Hypercoherence in frontal regions (F3↔F4, >0.8)",
        "interpretation": (
            "High theta and beta with low alpha and frontal hypercoherence indicate "
            "repetitive, rigid thought patterns typical of OCD."
        ),
        "clinical_implications": "Address obsessive thoughts; consider CBT and SSRIs"
    },
    "TBI": {
        "theta_frequencies": "Excessive slow theta (3–5 Hz, >30% total power)",
        "alpha_frequencies": "Suppressed alpha (<5% total power)",
        "beta_frequencies": "Reduced low beta (12–18 Hz) with occasional bursts",
        "coherence": "Severely reduced fronto-parietal coherence (<0.5)",
        "interpretation": (
            "Excessive slow theta, suppressed alpha, and reduced coherence point to "
            "post-traumatic cognitive slowing and connectivity deficits."
        ),
        "clinical_implications": "Neurological evaluation; monitor for cognitive decline"
    },
    "PTSD": {
        "theta_frequencies": "Elevated high theta (5–7 Hz, >20% total power)",
        "alpha_frequencies": "Suppressed alpha (<5% total power)",
        "beta_frequencies": "Elevated high beta (20–30 Hz, >20% total power)",
        "coherence": "Hypercoherence in frontal regions (F3↔F4, >0.8)",
        "interpretation": (
            "High theta and beta with suppressed alpha and frontal hypercoherence are markers "
            "of hypervigilance, trauma looping, and emotional dysregulation."
        ),
        "clinical_implications": "Trauma-focused therapy; monitor for flashbacks"
    },
    "Bipolar_Disorder": {
        "theta_frequencies": "Variable theta (4–8 Hz, fluctuating with mood state)",
        "alpha_frequencies": "Unstable alpha, often suppressed during mania",
        "beta_frequencies": "Elevated high beta during mania (20–30 Hz, >25% total power)",
        "coherence": "Variable coherence, often reduced during depressive phases",
        "interpretation": (
            "Fluctuating theta and beta with unstable alpha reflect mood instability, "
            "with high beta during mania and suppressed alpha during depression."
        ),
        "clinical_implications": "Mood stabilization; monitor for manic/depressive episodes"
    },
    "Epilepsy": {
        "theta_frequencies": "Elevated slow theta (3–5 Hz, >25% total power)",
        "alpha_frequencies": "Disrupted alpha, often absent during seizures",
        "beta_frequencies": "Erratic high beta bursts during seizures",
        "coherence": "Severely reduced during seizures (<0.4)",
        "interpretation": (
            "Slow theta and erratic beta bursts with disrupted coherence indicate "
            "epileptiform activity and cortical instability."
        ),
        "clinical_implications": "Neurological evaluation; monitor for seizure activity"
    }
}

# -------------------- Section 5: Live Use / Integration Suggestions --------------------
LIVE_USE_SUGGESTIONS = {
    "color_badge_system": (
        "Implement a color-coded badge system for vigilance in real-time (e.g., green=A1, purple=C). "
        "Display on a clinician dashboard or wearable device for immediate feedback."
    ),
    "pop_up_interventions": (
        "Trigger pop-up interventions during vigilance dips: "
        "e.g., HRV breathing cues (5 breaths/min), auditory entrainment (10 Hz), "
        "or visual flash biofeedback to re-engage attention."
    ),
    "real_time_alerts": (
        "Send real-time alerts to clinicians when vigilance drops below B2 or when high beta exceeds 25% total power, "
        "indicating potential anxiety or hyperarousal."
    ),
    "neurofeedback_integration": (
        "Integrate with neurofeedback systems (e.g., NeuroSky, Muse) to provide live SMR training "
        "or alpha uptraining based on current vigilance stage and EEG patterns."
    ),
    "patient_feedback_loop": (
        "Provide patients with a simplified feedback loop: e.g., a mobile app showing their vigilance state "
        "(color-coded) and suggesting mindfulness exercises or HRV training when needed."
    ),
    "data_logging": (
        "Log vigilance transitions and EEG metrics in real-time to a cloud database for longitudinal analysis, "
        "enabling clinicians to track progress and adjust protocols dynamically."
    )
}

# -------------------- Dynamic Helper Functions --------------------
def suggest_pyramid_level(metrics: Dict[str, Union[float, dict]]) -> Tuple[int, Dict, str]:
    """
    Evaluate computed metrics from EEG data and suggest a pyramid level with detailed reasoning.

    Args:
        metrics (Dict[str, Union[float, dict]]): A dictionary of computed metrics, expected to include:
            - 'theta_beta_ratio' (float): Average Theta/Beta ratio across channels.
            - 'alpha_change' (float): Percentage change in Alpha power (EO→EC).
            - 'paf' (float): Peak Alpha Frequency (Hz).
            - 'coherence' (dict): Coherence metrics, e.g., {'F3_F4': float, 'Fz_Pz': float}.
            - 'vigilance_index' (float, optional): A measure of vigilance (0–1, where 1 is fully alert).
            - 'delta_power' (float, optional): Percentage of total power in Delta band during wake.

    Returns:
        Tuple[int, Dict, str]: A tuple containing:
            - level (int): Suggested pyramid level (1–5).
            - mapping (Dict): The corresponding refined clinical mapping.
            - summary (str): A detailed summary of the evaluation process.
    """
    if not metrics:
        logger.warning("No metrics provided for pyramid level suggestion.")
        return 1, REFINED_CLINICAL_MAPPING[1], "No metrics provided. Defaulting to Level 1."

    tb_ratio = metrics.get("theta_beta_ratio")
    alpha_change = metrics.get("alpha_change")
    paf = metrics.get("paf")
    coherence = metrics.get("coherence", {})
    vigilance_index = metrics.get("vigilance_index", 1.0)
    delta_power = metrics.get("delta_power", 0.0)

    summary = "Metrics Evaluated: "
    if tb_ratio is not None:
        summary += f"Theta/Beta Ratio = {tb_ratio:.2f}, "
    if alpha_change is not None:
        summary += f"Alpha Change = {alpha_change:.1f}%, "
    if paf is not None:
        summary += f"PAF = {paf:.1f} Hz, "
    if coherence:
        summary += f"Coherence (F3↔F4) = {coherence.get('F3_F4', 'N/A'):.2f}, "
        summary += f"Coherence (Fz↔Pz) = {coherence.get('Fz_Pz', 'N/A'):.2f}, "
    summary += f"Vigilance Index = {vigilance_index:.2f}, "
    summary += f"Delta Power = {delta_power:.1f}%"

    # Scoring system for pyramid level
    score = 0
    if tb_ratio is not None:
        if tb_ratio > 4.5:
            score += 5
        elif tb_ratio > 3.5:
            score += 3
        elif tb_ratio > 3.0:
            score += 2
        elif tb_ratio > 2.0:
            score += 1

    if paf is not None:
        if paf < 8.0:
            score += 2
        elif paf < 9.5:
            score += 1

    if coherence:
        f3_f4 = coherence.get("F3_F4")
        fz_pz = coherence.get("Fz_Pz")
        if f3_f4 is not None and f3_f4 < 0.5:
            score += 2
        if fz_pz is not None and fz_pz < 0.5:
            score += 2

    if delta_power > 30:
        score += 5
    elif delta_power > 20:
        score += 3

    if vigilance_index < 0.3:
        score += 3
    elif vigilance_index < 0.5:
        score += 1

    # Determine level based on score
    if score >= 8:
        level = 5
    elif score >= 5:
        level = 4
    elif score >= 3:
        level = 3
    elif score >= 1:
        level = 2
    else:
        level = 1

    mapping = REFINED_CLINICAL_MAPPING.get(level, REFINED_CLINICAL_MAPPING[1])
    summary += f"\nScore: {score}. Suggested Level: {level} ({mapping['level_name']})."
    return level, mapping, summary

def get_refined_mapping(level: int) -> Optional[Dict]:
    """
    Retrieve refined clinical mapping for a given pyramid level (1–5).

    Args:
        level (int): The pyramid level (1–5).

    Returns:
        Optional[Dict]: The refined clinical mapping, or None if the level is invalid.
    """
    if not isinstance(level, int) or level < 1 or level > 5:
        logger.warning(f"Invalid pyramid level: {level}. Must be between 1 and 5.")
        return None
    return REFINED_CLINICAL_MAPPING.get(level)

def get_connectivity_mapping(level: int) -> Optional[Dict]:
    """
    Retrieve EEG connectivity mapping for a given pyramid level (1–5).

    Args:
        level (int): The pyramid level (1–5).

    Returns:
        Optional[Dict]: The EEG connectivity mapping, or None if the level is invalid.
    """
    if not isinstance(level, int) or level < 1 or level > 5:
        logger.warning(f"Invalid pyramid level: {level}. Must be between 1 and 5.")
        return None
    return EEG_CONNECTIVITY_MAPPING.get(level)

def get_vigilance_logic(stage: str) -> Optional[Dict]:
    """
    Retrieve vigilance transition logic for a given stage (e.g., 'A1', 'B2').

    Args:
        stage (str): The vigilance stage (e.g., 'A1', 'B2').

    Returns:
        Optional[Dict]: The vigilance transition logic, or None if the stage is invalid.
    """
    stage = stage.upper()
    if stage not in VIGILANCE_TRANSITION_LOGIC:
        logger.warning(f"Invalid vigilance stage: {stage}. Valid stages: {list(VIGILANCE_TRANSITION_LOGIC.keys())}")
        return None
    return VIGILANCE_TRANSITION_LOGIC.get(stage)

def get_condition_differentiation(condition: str) -> Optional[Dict]:
    """
    Retrieve EEG condition differentiation details for a given condition (e.g., 'ADHD', 'ASD').

    Args:
        condition (str): The condition name (e.g., 'ADHD', 'ASD').

    Returns:
        Optional[Dict]: The condition differentiation details, or None if the condition is invalid.
    """
    condition = condition.capitalize()
    if condition not in CONDITION_DIFFERENTIATION:
        logger.warning(f"Invalid condition: {condition}. Valid conditions: {list(CONDITION_DIFFERENTIATION.keys())}")
        return None
    return CONDITION_DIFFERENTIATION.get(condition)

def list_all_refined_mappings() -> List[Tuple[int, Dict]]:
    """
    Return all refined clinical mappings as a sorted list of tuples (level, mapping).

    Returns:
        List[Tuple[int, Dict]]: A sorted list of (level, mapping) tuples.
    """
    return sorted(REFINED_CLINICAL_MAPPING.items())

def list_all_connectivity_mappings() -> List[Tuple[int, Dict]]:
    """
    Return all EEG connectivity mappings as a sorted list of tuples (level, mapping).

    Returns:
        List[Tuple[int, Dict]]: A sorted list of (level, mapping) tuples.
    """
    return sorted(EEG_CONNECTIVITY_MAPPING.items())

def map_to_pyramid(bp_EO: Dict, bp_EC: Dict, site_metrics: Dict, global_metrics: Dict) -> List[str]:
    """
    Maps EEG metrics to a pyramid model for clinical interpretation using refined clinical and connectivity mappings.

    Args:
        bp_EO (Dict): Band powers for eyes-open condition (channel -> band -> power).
        bp_EC (Dict): Band powers for eyes-closed condition (channel -> band -> power).
        site_metrics (Dict): Site-specific metrics (channel -> metric -> value).
        global_metrics (Dict): Global metrics (metric -> value).

    Returns:
        List[str]: List of strings representing pyramid model mappings.
    """
    mappings = []

    try:
        # Compute average Theta/Beta Ratio
        tb_ratios = [metrics.get("Theta_Beta_Ratio", 0) for metrics in site_metrics.values() if "Theta_Beta_Ratio" in metrics]
        avg_tb_ratio = sum(tb_ratios) / len(tb_ratios) if tb_ratios else None

        # Compute average Alpha Change (EO→EC)
        alpha_changes = []
        for channel in bp_EO.keys():
            if channel in bp_EC:
                alpha_EO = bp_EO[channel].get("Alpha", 0)
                alpha_EC = bp_EC[channel].get("Alpha", 0)
                alpha_change = ((alpha_EC - alpha_EO) / alpha_EO * 100) if alpha_EO != 0 else 0
                alpha_changes.append(alpha_change)
        avg_alpha_change = sum(alpha_changes) / len(alpha_changes) if alpha_changes else None

        # Compute PAF (simplified: average peak alpha frequency across channels)
        paf_values = []
        for channel in bp_EO.keys():
            alpha_power = bp_EO[channel].get("Alpha", 0)
            if alpha_power > 0:
                # Placeholder: Assume PAF is computed elsewhere; here we simulate
                paf_values.append(10.0)  # Replace with actual PAF computation
        paf = sum(paf_values) / len(paf_values) if paf_values else None

        # Compute coherence metrics (simplified)
        coherence = {
            "F3_F4": global_metrics.get("F3_F4_coherence", 0.7),
            "Fz_Pz": global_metrics.get("Fz_Pz_coherence", 0.7)
        }

        # Compute vigilance index (simplified)
        vigilance_index = global_metrics.get("vigilance_index", 1.0)

        # Compute delta power (average percentage)
        delta_powers = []
        for channel in bp_EO.keys():
            delta_power = bp_EO[channel].get("Delta", 0)
            total_power = sum(bp_EO[channel].values())
            delta_powers.append((delta_power / total_power * 100) if total_power > 0 else 0)
        delta_power = sum(delta_powers) / len(delta_powers) if delta_powers else 0

        # Suggest pyramid level based on metrics
        metrics = {
            "theta_beta_ratio": avg_tb_ratio,
            "alpha_change": avg_alpha_change,
            "paf": paf,
            "coherence": coherence,
            "vigilance_index": vigilance_index,
            "delta_power": delta_power
        }
        level, mapping, summary = suggest_pyramid_level(metrics)
        mappings.append(summary)

        # Add refined clinical mapping details
        refined_mapping = get_refined_mapping(level)
        if refined_mapping:
            mappings.append(f"Refined Clinical Mapping: {refined_mapping['level_name']}")
            mappings.append("  EEG Patterns: " + ", ".join(refined_mapping["eeg_patterns"]))
            mappings.append("  Cognitive/Behavioral Implications: " + refined_mapping["cognitive_behavior"])
            mappings.append("  Clinical Implications: " + refined_mapping["clinical_implications"])
            mappings.append("  Protocols: " + "; ".join(refined_mapping["protocols"]))

        # Add connectivity mapping details
        connectivity_mapping = get_connectivity_mapping(level)
        if connectivity_mapping:
            mappings.append(f"EEG Connectivity Mapping: {connectivity_mapping['level_name']}")
            mappings.append("  EEG Patterns: " + ", ".join(connectivity_mapping["eeg_patterns"]))
            mappings.append("  Differentiators: " + connectivity_mapping["differentiators"])
            mappings.append("  Cognitive/Behavioral Implications: " + connectivity_mapping["cognition_behavior"])
            mappings.append("  Clinical Implications: " + connectivity_mapping["clinical_implications"])
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
    except Exception as e:
        logger.error(f"Error in map_to_pyramid: {e}")
        mappings.append(f"Error mapping to pyramid model: {str(e)}")

    # If no mappings are generated, add a default message
    if not mappings:
        mappings.append("No pyramid mappings generated: Insufficient data")

    return mappings

if __name__ == "__main__":
    # Configure logging for the demo
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Example dynamic usage: simulate metrics from EEG data
    sample_metrics = {
        "theta_beta_ratio": 2.5,
        "alpha_change": 30.0,
        "paf": 9.8,
        "coherence": {"F3_F4": 0.7, "Fz_Pz": 0.65},
        "vigilance_index": 0.8,
        "delta_power": 5.0
    }
    level, mapping, summary = suggest_pyramid_level(sample_metrics)
    logger.info("Dynamic Pyramid Level Evaluation:")
    logger.info(summary)

    logger.info("\n=== Refined Clinical Mapping ===")
    for lvl, mapping in list_all_refined_mappings():
        logger.info(f"{mapping['level_name']}")
        logger.info("  EEG Patterns: " + ", ".join(mapping["eeg_patterns"]))
        logger.info("  Cognitive/Behavioral Implications: " + mapping["cognitive_behavior"])
        logger.info("  Clinical Implications: " + mapping["clinical_implications"])
        logger.info("  Protocols: " + "; ".join(mapping["protocols"]))
        logger.info("")

    logger.info("\n=== EEG Connectivity Mapping ===")
    for lvl, mapping in list_all_connectivity_mappings():
        logger.info(f"{mapping['level_name']}")
        logger.info("  EEG Patterns: " + ", ".join(mapping["eeg_patterns"]))
        logger.info("  Differentiators: " + mapping["differentiators"])
        logger.info("  Cognitive/Behavioral Implications: " + mapping["cognition_behavior"])
        logger.info("  Clinical Implications: " + mapping["clinical_implications"])
        logger.info("  Vigilance Stage: " + mapping["vigilance_stage"])
        logger.info("  Neurofeedback Targets: " + "; ".join(mapping["neurofeedback_targets"]))
        logger.info("")

    # Example: Retrieve and display vigilance transition logic for stage A2
    stage = "A2"
    logic = get_vigilance_logic(stage)
    if logic:
        logger.info(f"Vigilance Transition Logic for Stage {stage}:")
        logger.info("  EEG Signature: " + logic["eeg_signature"])
        logger.info("  Clinical Meaning: " + logic["clinical_meaning"])
        logger.info("  Possible Response: " + logic["possible_response"])
        logger.info("  Transition Probabilities: " + str(logic["transition_probabilities"]))

    # Example: Retrieve condition differentiation for ADHD
    condition = "ADHD"
    cond_info = get_condition_differentiation(condition)
    if cond_info:
        logger.info(f"\nCondition Differentiation for {condition}:")
        logger.info("  Theta Frequencies: " + str(cond_info.get("theta_frequencies", "N/A")))
        logger.info("  Alpha Frequencies: " + cond_info.get("alpha_frequencies", "N/A"))
        logger.info("  Beta Frequencies: " + cond_info.get("beta_frequencies", "N/A"))
        logger.info("  Coherence: " + cond_info.get("coherence", "N/A"))
        logger.info("  Interpretation: " + cond_info.get("interpretation", ""))
        logger.info("  Clinical Implications: " + cond_info.get("clinical_implications", ""))

    # Print live use suggestions
    logger.info("\nLive Use / Integration Suggestions:")
    for key, suggestion in LIVE_USE_SUGGESTIONS.items():
        logger.info(f"  {key}: {suggestion}")
