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
  • Vigilance Analysis: Functions to compute and plot vigilance states from MNE Raw data.

Dynamic helper functions are provided to map EEG metrics to pyramid levels, suggest interventions, and support live clinical use.
For example, `suggest_pyramid_level()` uses computed metrics to recommend a pyramid level with a detailed summary.
The `compute_vigilance_states` function calculates vigilance stages, and `plot_vigilance_hypnogram` visualizes them.

Ensure this file is saved with UTF-8 encoding to display emojis (🟢, 🟡, 🟠, 🔴, ⚫, etc.) correctly.
"""

import logging
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from pathlib import Path

# --- Optional Imports for Vigilance ---
# These are only strictly needed if using the vigilance functions.
# Wrap in try/except for environments where MNE/Matplotlib might not be installed.
try:
    import mne
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch # Specifically needed for legend
    VIGILANCE_ENABLED = True
except ImportError:
    mne = None
    plt = None
    Patch = None # No need for Patch if plt failed
    VIGILANCE_ENABLED = False
    logging.getLogger(__name__).warning(
        "MNE or Matplotlib not found. Vigilance analysis functions will be disabled."
    )
# --- End Optional Imports ---


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
    },
    # Added Undefined state for error handling
    "Undefined": {
        "eeg_signature": "Could not compute vigilance (e.g., data quality issue)",
        "clinical_meaning": "Data quality issue or processing error",
        "possible_response": "Review data quality and preprocessing steps",
        "transition_probabilities": {}
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


# -------------------- Section 6: Vigilance Analysis Functions --------------------
# --- Copied and Integrated Vigilance Code ---

# Define frequency bands (can be customized)
ALPHA_BAND = (8, 12)
THETA_BAND = (4, 8)

# Clinical vigilance color mapping (for the strip plot)
VIGILANCE_COLORS = {
    'A1': '#000080',  # navy (posterior alpha dominant)
    'A2': '#008080',  # teal (emerging anterior alpha)
    'A3': '#00FFFF',  # cyan (anterior alpha dominant)
    'B1': '#FFBF00',  # amber (alpha drop-out begins)
    'B2': '#FF8000',  # orange (frontal theta appears)
    'B3': '#FF0000',  # red (theta prominent)
    'C': '#800080',   # purple (sleep onset markers)
    'Undefined': '#808080' # Gray for undefined states
}


def compute_band_power(epoch: np.ndarray, sfreq: float, band: tuple[float, float], channel_idx: int = 0, min_samples_for_filter: int = 256, filter_length_factor: float = 0.8) -> float:
    """
    Compute the power in a specific frequency band for an epoch, dynamically adjusting filter parameters.
    (Integrated from vigilance.py)

    Args:
        epoch (np.ndarray): The epoch data (channels x samples).
        sfreq (float): Sampling frequency in Hz.
        band (tuple[float, float]): Frequency band as (low, high) in Hz.
        channel_idx (int): Index of the channel to process (default: 0).
        min_samples_for_filter (int): Minimum samples for FIR filtering (default: 256). PSD used otherwise.
        filter_length_factor (float): Factor for FIR filter length (default: 0.8).

    Returns:
        float: Average power in the specified band.

    Raises:
        ValueError: If filtering or PSD fails or channel index is invalid.
        ImportError: If MNE is not installed.
    """
    if not VIGILANCE_ENABLED:
        raise ImportError("MNE is required for compute_band_power. Vigilance analysis disabled.")

    if not 0 <= channel_idx < epoch.shape[0]:
        raise ValueError(f"Channel index {channel_idx} out of bounds for epoch with shape {epoch.shape}")

    epoch_channel = epoch[channel_idx, :]
    signal_length = epoch_channel.shape[0]

    if signal_length == 0:
        logger.warning(f"Received zero-length signal for band {band}. Returning 0 power.")
        return 0.0

    if signal_length < min_samples_for_filter:
        logger.debug(f"Signal too short ({signal_length} < {min_samples_for_filter}); using PSD.")
        try:
            nperseg = min(256, signal_length) if signal_length > 0 else 1
            nperseg = max(1, min(nperseg, signal_length))

            psd, freqs = mne.time_frequency.psd_array_welch(
                epoch_channel[None, :], sfreq, fmin=band[0], fmax=band[1],
                n_per_seg=nperseg, average='mean', verbose=False
            )
            if freqs.size > 0:
                freq_res = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
                power = np.sum(psd[0] * freq_res)
            else: power = 0.0
            logger.debug(f"PSD Power: {power:.4f}")
            return power
        except Exception as e:
            logger.error(f"Error computing PSD for {band} on channel {channel_idx}: {e}", exc_info=True)
            raise ValueError(f"Error computing PSD for band {band}: {e}")
    else:
        logger.debug(f"Signal long enough ({signal_length} >= {min_samples_for_filter}); using filtering.")
        try:
            target_filter_length = max(int(filter_length_factor * signal_length), 3)
            if target_filter_length > 0:
                trans_bandwidth = max(2.0, sfreq / target_filter_length)
            else: trans_bandwidth = 2.0

            band_width = band[1] - band[0]
            if band_width > 0: trans_bandwidth = min(trans_bandwidth, max(band_width / 2, 1.0))
            else: trans_bandwidth = max(trans_bandwidth, 1.0)

            epoch_channel_float64 = epoch_channel.astype(np.float64)
            filtered = mne.filter.filter_data(
                epoch_channel_float64[None, :], sfreq, l_freq=band[0], h_freq=band[1],
                l_trans_bandwidth=trans_bandwidth, h_trans_bandwidth=trans_bandwidth,
                filter_length='auto', phase='zero', fir_window='hamming', fir_design='firwin',
                verbose=False
            )
            power = np.mean(filtered ** 2)
            logger.debug(f"Filter Power: {power:.4f}")
            return power
        except Exception as e:
            logger.error(f"Error filtering data for {band} on channel {channel_idx}: {e}", exc_info=True)
            raise ValueError(f"Error filtering data for band {band}: {e}")


def classify_epoch(epoch: np.ndarray, sfreq: float, channel_idx: int = 0) -> str:
    """
    Classify the vigilance state for a given epoch.
    (Integrated from vigilance.py)

    Args:
        epoch (np.ndarray): Epoch data (channels x samples).
        sfreq (float): Sampling frequency.
        channel_idx (int): Index of the channel to use (default: 0).

    Returns:
        str: Vigilance stage ('A1'-'C' or 'Undefined').
    """
    if not VIGILANCE_ENABLED:
        logger.error("Cannot classify epoch, MNE not available.")
        return 'Undefined'

    try:
        alpha_power = compute_band_power(epoch, sfreq, ALPHA_BAND, channel_idx=channel_idx)
        theta_power = compute_band_power(epoch, sfreq, THETA_BAND, channel_idx=channel_idx)
        ratio = alpha_power / (theta_power + 1e-9) # Epsilon for stability
        logger.debug(f"Classifying epoch (Ch {channel_idx}): Alpha={alpha_power:.4f}, Theta={theta_power:.4f}, Ratio={ratio:.4f}")
    except (ValueError, ImportError) as e:
        logger.error(f"Cannot classify epoch (Ch {channel_idx}): {e}")
        return 'Undefined'

    if ratio > 2.0: stage = 'A1'
    elif 1.5 < ratio <= 2.0: stage = 'A2'
    elif 1.0 < ratio <= 1.5: stage = 'A3'
    elif 0.75 < ratio <= 1.0: stage = 'B1'
    elif 0.5 < ratio <= 0.75: stage = 'B2'
    elif ratio <= 0.5: stage = 'B3'
    elif np.isnan(ratio) or np.isinf(ratio):
        logger.warning(f"Ratio is NaN/Inf (A={alpha_power:.4f}, T={theta_power:.4f}). Assigning 'C'.")
        stage = 'C'
    else: # Should not happen with epsilon, but as fallback
        logger.warning(f"Unexpected ratio {ratio:.4f}. Assigning 'C'.")
        stage = 'C'

    logger.debug(f"Epoch classified as stage: {stage}")
    return stage


def compute_vigilance_states(raw: 'mne.io.Raw', epoch_length: float = 1.0, channel_name: str = 'OZ') -> list[tuple[float, str]]:
    """
    Compute vigilance states for the given raw EEG data.
    (Integrated from vigilance.py)

    Args:
        raw (mne.io.Raw): Raw EEG data. Requires MNE installation.
        epoch_length (float): Duration of each epoch in seconds (default: 1.0).
        channel_name (str): Channel name for classification (default: 'OZ').

    Returns:
        list[tuple[float, str]]: List of (start_time, stage) tuples. Empty if fails.
    """
    if not VIGILANCE_ENABLED or not mne:
        logger.error("Cannot compute vigilance states, MNE not available.")
        return []
    if not isinstance(raw, mne.io.Raw):
        logger.error("Input must be an MNE Raw object.")
        return []

    sfreq = raw.info['sfreq']
    n_samples_epoch = int(epoch_length * sfreq)

    if n_samples_epoch <= 0:
        logger.error(f"Invalid epoch setup: {epoch_length}s * {sfreq}Hz = {n_samples_epoch} samples.")
        return []

    logger.info(f"Computing vigilance: sfreq={sfreq:.2f}Hz, epoch_len={epoch_length}s, samples/epoch={n_samples_epoch}")

    ch_names_lower = [ch.lower() for ch in raw.ch_names]
    try:
        channel_idx = ch_names_lower.index(channel_name.lower())
    except ValueError:
        logger.error(f"Channel '{channel_name}' not found. Available: {raw.ch_names}")
        return []

    logger.info(f"Using channel '{raw.ch_names[channel_idx]}' (idx {channel_idx})")

    try:
        # Explicitly load data with preload=True if not already loaded
        # This prevents potential issues with accessing data later.
        if not raw.preload:
            logger.debug("Preloading raw data for vigilance computation...")
            raw.load_data(verbose=False)
        data = raw.get_data(picks=[channel_idx]) # Load only needed channel
    except Exception as e:
        logger.error(f"Failed getting data for '{raw.ch_names[channel_idx]}': {e}", exc_info=True)
        return []

    n_total_samples = data.shape[1]
    n_epochs = n_total_samples // n_samples_epoch

    if n_epochs == 0:
        logger.warning(f"Data too short ({n_total_samples} samples) for epoch length ({n_samples_epoch} samples).")
        return []

    vigilance_states = []
    logger.info(f"Processing {n_epochs} epochs...")
    for i in range(n_epochs):
        start = i * n_samples_epoch
        end = start + n_samples_epoch
        epoch_data = data[:, start:end] # Shape (1, n_samples_epoch)
        stage = classify_epoch(epoch_data, sfreq, channel_idx=0) # Use index 0 (only channel loaded)
        start_time = start / sfreq
        vigilance_states.append((start_time, stage))
        if (i + 1) % 100 == 0 or (i + 1) == n_epochs:
            logger.info(f"Processed {i + 1}/{n_epochs} epochs.")

    logger.info(f"Vigilance computation complete. Found {len(vigilance_states)} states.")
    return vigilance_states


def plot_vigilance_hypnogram(vigilance_states: list[tuple[float, str]], output_dir: Union[str, Path], condition: str, epoch_length: float = 1.0) -> None:
    """
    Plot and save a vigilance hypnogram and strip plot.
    (Integrated from vigilance.py)

    Args:
        vigilance_states: List of (start_time, stage) tuples.
        output_dir: Directory to save plots.
        condition: Condition identifier (e.g., "EO", "EC").
        epoch_length: Duration of each epoch in seconds.
    """
    if not VIGILANCE_ENABLED or not plt:
        logger.error("Cannot plot vigilance, Matplotlib not available.")
        return

    output_dir = Path(output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create output directory {output_dir}: {e}")
        return

    if not vigilance_states:
        logger.warning(f"No vigilance states for '{condition}'. Skipping plots.")
        return

    stage_to_level = {'A1': 6, 'A2': 5, 'A3': 4, 'B1': 3, 'B2': 2, 'B3': 1, 'C': 0, 'Undefined': -1}
    times = [t for t, s in vigilance_states]
    levels = [stage_to_level.get(s, -1) for t, s in vigilance_states]

    if not times:
        logger.warning(f"No valid states to plot for '{condition}'.")
        return

    # Extend plot to end of last epoch
    times.append(times[-1] + epoch_length)
    levels.append(levels[-1])

    # --- Hypnogram Plot ---
    hypnogram_path = output_dir / f"vigilance_hypnogram_{condition}.png"
    fig_hyp = None
    try:
        plt.style.use('dark_background')
        fig_hyp, ax_hyp = plt.subplots(figsize=(12, 5))
        fig_hyp.patch.set_facecolor('black')
        ax_hyp.set_facecolor('black')
        ax_hyp.step(times, levels, where='post', color='cyan', linewidth=1.5)
        ax_hyp.set_ylim(-1.5, 6.5)
        ax_hyp.tick_params(axis='x', colors='white')
        ax_hyp.tick_params(axis='y', colors='white')
        for spine in ax_hyp.spines.values(): spine.set_color('white')
        ax_hyp.set_xlabel("Time (s)", color='white', fontsize=10)
        ax_hyp.set_ylabel("Vigilance Stage", color='white', fontsize=10)

        yticks_map = {v: k for k, v in stage_to_level.items()}
        present_levels = sorted(list(set(levels)), reverse=True)
        ax_hyp.set_yticks(present_levels)
        ax_hyp.set_yticklabels([yticks_map.get(lvl, 'UNK') for lvl in present_levels])

        ax_hyp.set_title(f"Vigilance Hypnogram ({condition})", color='white', fontsize=12)
        plt.tight_layout(pad=1.5)
        plt.savefig(hypnogram_path, facecolor='black')
        logger.info(f"Vigilance hypnogram saved: {hypnogram_path}")
    except Exception as e:
        logger.error(f"Failed plotting/saving hypnogram {hypnogram_path}: {e}", exc_info=True)
    finally:
        if fig_hyp: plt.close(fig_hyp) # Ensure figure is closed

    # --- Strip Plot ---
    strip_path = output_dir / f"vigilance_strip_{condition}.png"
    fig_strip = None
    try:
        fig_strip = plot_vigilance_strip(vigilance_states, epoch_length)
        # Check if plot_vigilance_strip returned a figure (it might return None on error)
        if fig_strip:
             fig_strip.savefig(strip_path, facecolor='black')
             logger.info(f"Vigilance strip saved: {strip_path}")
        else:
             logger.warning(f"Could not save vigilance strip for {condition}, plotting failed.")
    except Exception as e:
        logger.error(f"Failed plotting/saving strip {strip_path}: {e}", exc_info=True)
    finally:
        if fig_strip: plt.close(fig_strip) # Ensure figure is closed


def plot_vigilance_strip(vigilance_states: list[tuple[float, str]], epoch_length: float = 1.0) -> Optional['plt.Figure']:
    """
    Plot a vigilance strip using colored rectangles.
    (Integrated from vigilance.py)

    Args:
        vigilance_states: List of (start_time, stage) tuples.
        epoch_length: Duration of each epoch in seconds.

    Returns:
        matplotlib.figure.Figure or None if plotting failed or disabled.
    """
    if not VIGILANCE_ENABLED or not plt or not Patch: # Check Patch too
        logger.error("Cannot plot strip, Matplotlib not available.")
        return None
    if not vigilance_states:
        logger.warning("No vigilance states for strip plot.")
        # Return an empty figure or handle as appropriate
        fig, ax = plt.subplots(figsize=(10, 1.5))
        return fig

    try:
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 1.5))
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        valid_stages_present = set()
        for (start_time, stage) in vigilance_states:
            color = VIGILANCE_COLORS.get(stage, 'gray')
            ax.add_patch(plt.Rectangle((start_time, 0), epoch_length, 1, color=color))
            if stage in VIGILANCE_COLORS:
                valid_stages_present.add(stage)

        # Legend for present stages
        # Order legend elements consistently based on VIGILANCE_COLORS order
        ordered_stages = [s for s in VIGILANCE_COLORS if s in valid_stages_present]
        legend_elements = [Patch(facecolor=VIGILANCE_COLORS[stage], label=stage) for stage in ordered_stages]

        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                      ncol=len(legend_elements), fontsize=8, frameon=False, labelcolor='white')

        total_time = vigilance_states[-1][0] + epoch_length
        ax.set_xlim(0, total_time)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)
        plt.tight_layout(pad=0.5)
        plt.subplots_adjust(bottom=0.2)
        return fig
    except Exception as e:
        logger.error(f"Error creating vigilance strip plot: {e}", exc_info=True)
        # Attempt to close figure if it was created before the error
        if 'fig' in locals() and fig:
             plt.close(fig)
        return None # Indicate failure

# --- End Vigilance Code ---


# -------------------- Section 7: Dynamic Helper & Mapping Functions --------------------

def suggest_pyramid_level(
    metrics: Dict[str, Union[float, dict]],
    vigilance_summary: Optional[Dict] = None
) -> Tuple[int, Dict, str]:
    """
    Evaluate computed metrics and suggest a pyramid level with detailed reasoning.
    Optionally uses vigilance summary information.

    Args:
        metrics (Dict[str, Union[float, dict]]): Dictionary of computed metrics (see original docstring).
        vigilance_summary (Optional[Dict]): Summary of vigilance states, e.g.,
                                           {'percent_low_vigilance': float (0-100)}.

    Returns:
        Tuple[int, Dict, str]: Suggested level (1–5), corresponding mapping, evaluation summary.
    """
    if not metrics:
        logger.warning("No metrics provided for pyramid level suggestion.")
        return 1, REFINED_CLINICAL_MAPPING[1], "No metrics provided. Defaulting to Level 1."

    tb_ratio = metrics.get("theta_beta_ratio")
    alpha_change = metrics.get("alpha_change")
    paf = metrics.get("paf")
    coherence = metrics.get("coherence", {})
    # Use provided vigilance_index if available, otherwise default to 1.0 (high vigilance)
    # This ensures backward compatibility if vigilance_index is still passed directly.
    vigilance_index = metrics.get("vigilance_index", 1.0)
    delta_power = metrics.get("delta_power", 0.0)
    percent_low_vigilance = vigilance_summary.get('percent_low_vigilance') if vigilance_summary else None

    # --- Build Summary String ---
    summary_parts = ["Metrics Evaluated:"]
    if tb_ratio is not None: summary_parts.append(f"T/B Ratio={tb_ratio:.2f}")
    if alpha_change is not None: summary_parts.append(f"Alpha Change={alpha_change:.1f}%")
    if paf is not None: summary_parts.append(f"PAF={paf:.1f}Hz")
    # Use .get() for coherence keys to avoid errors if missing
    if coherence.get('F3_F4') is not None: summary_parts.append(f"Coh(F3↔F4)={coherence['F3_F4']:.2f}")
    if coherence.get('Fz_Pz') is not None: summary_parts.append(f"Coh(Fz↔Pz)={coherence['Fz_Pz']:.2f}")
    summary_parts.append(f"Vigilance Index={vigilance_index:.2f}") # Keep original index for now
    if delta_power is not None: summary_parts.append(f"Delta Power={delta_power:.1f}%")
    if percent_low_vigilance is not None:
        summary_parts.append(f"% Low Vigilance={percent_low_vigilance:.1f}%")
    summary = ", ".join(summary_parts)

    # --- Scoring System ---
    score = 0
    # Theta/Beta Ratio Score
    if tb_ratio is not None:
        if tb_ratio > 4.5: score += 5
        elif tb_ratio > 3.5: score += 3
        elif tb_ratio > 3.0: score += 2
        elif tb_ratio > 2.0: score += 1
    # PAF Score
    if paf is not None:
        if paf < 8.0: score += 2
        elif paf < 9.5: score += 1
    # Coherence Score
    f3_f4 = coherence.get("F3_F4")
    fz_pz = coherence.get("Fz_Pz")
    if f3_f4 is not None and f3_f4 < 0.5: score += 2
    if fz_pz is not None and fz_pz < 0.5: score += 2
    # Delta Power Score
    if delta_power is not None:
        if delta_power > 30: score += 5
        elif delta_power > 20: score += 3
    # Vigilance Score (using summary if available, else index)
    if percent_low_vigilance is not None:
        if percent_low_vigilance > 50: score += 3 # More than 50% time in low vigilance
        elif percent_low_vigilance > 20: score += 1 # More than 20% time in low vigilance
    else: # Fallback to vigilance_index if vigilance_summary not provided
        if vigilance_index < 0.3: score += 3
        elif vigilance_index < 0.5: score += 1

    # --- Determine Level ---
    if score >= 8: level = 5
    elif score >= 5: level = 4
    elif score >= 3: level = 3
    elif score >= 1: level = 2
    else: level = 1

    mapping = REFINED_CLINICAL_MAPPING.get(level, REFINED_CLINICAL_MAPPING[1])
    summary += f"\nScore: {score}. Suggested Level: {level} ({mapping['level_name']})."
    return level, mapping, summary


def get_refined_mapping(level: int) -> Optional[Dict]:
    """Retrieve refined clinical mapping for a given pyramid level (1–5)."""
    if not isinstance(level, int) or not 1 <= level <= 5:
        logger.warning(f"Invalid pyramid level: {level}. Must be 1-5.")
        return None
    return REFINED_CLINICAL_MAPPING.get(level)


def get_connectivity_mapping(level: int) -> Optional[Dict]:
    """Retrieve EEG connectivity mapping for a given pyramid level (1–5)."""
    if not isinstance(level, int) or not 1 <= level <= 5:
        logger.warning(f"Invalid pyramid level: {level}. Must be 1-5.")
        return None
    return EEG_CONNECTIVITY_MAPPING.get(level)


def get_vigilance_logic(stage: str) -> Optional[Dict]:
    """Retrieve vigilance transition logic for a stage (e.g., 'A1', 'B2')."""
    stage = stage.upper()
    if stage not in VIGILANCE_TRANSITION_LOGIC:
        logger.warning(f"Invalid vigilance stage: {stage}. Valid: {list(VIGILANCE_TRANSITION_LOGIC.keys())}")
        return None
    return VIGILANCE_TRANSITION_LOGIC.get(stage)


def get_condition_differentiation(condition: str) -> Optional[Dict]:
    """Retrieve EEG condition differentiation details (e.g., 'ADHD', 'ASD')."""
    # Find case-insensitive match
    found_key = next((key for key in CONDITION_DIFFERENTIATION if key.lower() == condition.lower()), None)
    if found_key is None:
        logger.warning(f"Condition '{condition}' not found. Valid: {list(CONDITION_DIFFERENTIATION.keys())}")
        return None
    return CONDITION_DIFFERENTIATION.get(found_key)


def list_all_refined_mappings() -> List[Tuple[int, Dict]]:
    """Return all refined clinical mappings sorted by level."""
    return sorted(REFINED_CLINICAL_MAPPING.items())


def list_all_connectivity_mappings() -> List[Tuple[int, Dict]]:
    """Return all EEG connectivity mappings sorted by level."""
    return sorted(EEG_CONNECTIVITY_MAPPING.items())


def calculate_vigilance_summary(vigilance_states: List[Tuple[float, str]]) -> Dict:
    """
    Calculates summary statistics from vigilance states.

    Args:
        vigilance_states: List of (start_time, stage) tuples.

    Returns:
        Dict: Summary dict, e.g., {'percent_low_vigilance': float, 'stage_counts': dict}.
    """
    summary = {'percent_low_vigilance': 0.0, 'stage_counts': {}}
    if not vigilance_states:
        return summary

    total_epochs = len(vigilance_states)
    low_vigilance_stages = {'B2', 'B3', 'C', 'Undefined'}
    low_vigilance_count = 0
    stage_counts = {}

    for _, stage in vigilance_states:
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
        if stage in low_vigilance_stages:
            low_vigilance_count += 1

    summary['stage_counts'] = stage_counts
    if total_epochs > 0:
        summary['percent_low_vigilance'] = (low_vigilance_count / total_epochs) * 100

    return summary


def map_to_pyramid(
    bp_EO: Dict, bp_EC: Dict, site_metrics: Dict, global_metrics: Dict,
    vigilance_states: Optional[List[Tuple[float, str]]] = None
) -> List[str]:
    """
    Maps EEG metrics to a pyramid model, optionally including vigilance state summary.

    Args:
        bp_EO (Dict): Band powers for eyes-open (channel -> band -> power).
        bp_EC (Dict): Band powers for eyes-closed (channel -> band -> power).
        site_metrics (Dict): Site-specific metrics (channel -> metric -> value).
        global_metrics (Dict): Global metrics (metric -> value).
        vigilance_states (Optional[List[Tuple[float, str]]]): Computed vigilance states.

    Returns:
        List[str]: List of strings representing pyramid model mappings.
    """
    mappings = []
    try:
        # --- Metric Calculation (Ensure robustness with .get and checks) ---
        # Average Theta/Beta Ratio
        tb_ratios = [m.get("Theta_Beta_Ratio", 0) for m in site_metrics.values() if isinstance(m, dict) and "Theta_Beta_Ratio" in m]
        avg_tb_ratio = np.mean(tb_ratios) if tb_ratios else None

        # Average Alpha Change (EO→EC)
        alpha_changes = []
        for ch, eo_bands in bp_EO.items():
             ec_bands = bp_EC.get(ch)
             if isinstance(eo_bands, dict) and isinstance(ec_bands, dict):
                 alpha_eo = eo_bands.get("Alpha")
                 alpha_ec = ec_bands.get("Alpha")
                 if alpha_eo is not None and alpha_ec is not None and alpha_eo != 0:
                      change = ((alpha_ec - alpha_eo) / alpha_eo) * 100
                      alpha_changes.append(change)
        avg_alpha_change = np.mean(alpha_changes) if alpha_changes else None

        # Average PAF (Placeholder - replace with actual calculation if available)
        paf_values = [m.get("PAF", 10.0) for m in site_metrics.values() if isinstance(m, dict)]
        paf = np.mean(paf_values) if paf_values else None

        # Coherence (Use .get for safety)
        coherence = {
            "F3_F4": global_metrics.get("Coherence_F3_F4"), # Example key
            "Fz_Pz": global_metrics.get("Coherence_Fz_Pz")  # Example key
        }
        coherence_for_level = {k: v for k, v in coherence.items() if v is not None} # Filter None

        # Delta Power (Average % during EO)
        delta_powers_pct = []
        for ch, eo_bands in bp_EO.items():
             if isinstance(eo_bands, dict):
                 total_power = sum(v for v in eo_bands.values() if isinstance(v, (int, float)))
                 if total_power > 0:
                      delta_power = eo_bands.get("Delta", 0)
                      if isinstance(delta_power, (int, float)):
                           delta_pct = (delta_power / total_power) * 100
                           delta_powers_pct.append(delta_pct)
        avg_delta_power = np.mean(delta_powers_pct) if delta_powers_pct else 0.0

        # --- Vigilance Summary ---
        vigilance_summary = None
        if vigilance_states:
            mappings.append("--- Vigilance Analysis ---")
            vigilance_summary = calculate_vigilance_summary(vigilance_states)
            # Add summary to mappings
            for stage, count in sorted(vigilance_summary.get('stage_counts', {}).items()):
                 percent = (count / len(vigilance_states)) * 100
                 mappings.append(f"  Stage {stage:<9}: {count:>4} epochs ({percent:>5.1f}%)")
            mappings.append(f"  % Low Vigilance : {vigilance_summary.get('percent_low_vigilance', 0.0):.1f}%")
            mappings.append("-" * 26) # Separator

        # --- Suggest Pyramid Level ---
        metrics_for_level = {
            "theta_beta_ratio": avg_tb_ratio,
            "alpha_change": avg_alpha_change,
            "paf": paf,
            "coherence": coherence_for_level,
            "delta_power": avg_delta_power,
            "vigilance_index": global_metrics.get("vigilance_index", 1.0) # Legacy index
        }
        level, mapping_data, summary_text = suggest_pyramid_level(metrics_for_level, vigilance_summary)
        mappings.append("--- Pyramid Level Suggestion ---")
        mappings.append(summary_text) # Includes score and level
        mappings.append("-" * 28) # Separator

        # --- Add Mapping Details ---
        # Refined Clinical Mapping
        refined_map = get_refined_mapping(level)
        if refined_map:
            mappings.append(f"\n--- Refined Clinical Mapping (Level {level}) ---")
            mappings.append(f"{refined_map['level_name']}")
            mappings.append("  EEG Patterns: " + ", ".join(refined_map.get("eeg_patterns", [])))
            mappings.append("  Cognitive/Behavior: " + refined_map.get("cognitive_behavior", "N/A"))
            mappings.append("  Clinical Implications: " + refined_map.get("clinical_implications", "N/A"))
            mappings.append("  Protocols: " + "; ".join(refined_map.get("protocols", [])))
            mappings.append("-" * 34) # Separator

        # Connectivity Mapping
        conn_map = get_connectivity_mapping(level)
        if conn_map:
            mappings.append(f"\n--- EEG Connectivity Mapping (Level {level}) ---")
            mappings.append(f"{conn_map['level_name']}")
            mappings.append("  Differentiators: " + conn_map.get("differentiators", "N/A"))
            mappings.append("  Vigilance Stage Link: " + conn_map.get("vigilance_stage", "N/A"))
            mappings.append("  Neurofeedback Targets: " + "; ".join(conn_map.get("neurofeedback_targets", [])))
            mappings.append("-" * 34) # Separator

        # --- Add Specific Metric Details (Example) ---
        mappings.append("\n--- Specific Metrics (Examples) ---")
        global_fa = global_metrics.get("Frontal_Asymmetry")
        if global_fa is not None:
             mappings.append(f"  Global Frontal Asymmetry (F4/F3): {global_fa:.2f}")
        for ch, met in site_metrics.items():
            if isinstance(met, dict):
                 tbr = met.get("Theta_Beta_Ratio")
                 if tbr is not None:
                      mappings.append(f"  {ch} T/B Ratio: {tbr:.2f}")
        mappings.append("-" * 31) # Separator

    except Exception as e:
        logger.error(f"Error during map_to_pyramid: {e}", exc_info=True)
        # Add error message to the start for visibility
        mappings.insert(0, f"❌ ERROR MAPPING TO PYRAMID MODEL: {str(e)}")

    if not mappings:
        mappings.append("No pyramid mappings generated: Check input data or logs.")

    return mappings


if __name__ == "__main__":
    # Configure logging for the demo
    # Use a more detailed format including the module name
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger.info("Starting Pyramid Model Demo...") # Use the module logger

    # === Vigilance Analysis Demo (Requires MNE/Matplotlib) ===
    computed_vigilance_states = None # Initialize to None
    if VIGILANCE_ENABLED:
        logger.info("\n=== Vigilance Analysis Demo ===")
        # --- Simulate MNE Raw Data ---
        try:
            ch_names = ['Fz', 'Cz', 'Pz', 'Oz']
            sfreq = 250  # Hz
            duration = 60  # seconds (shortened for faster demo)
            n_samples = duration * sfreq
            # Create some plausible EEG-like data with changing alpha/theta
            times = np.arange(n_samples) / sfreq
            signal = np.zeros((len(ch_names), n_samples))
            # Phase 1: Alpha dominant (A1/A2) - First 20s
            idx1 = int(duration / 3 * sfreq)
            signal[:, :idx1] += 10 * np.sin(2 * np.pi * 10 * times[:idx1]) # 10Hz Alpha
            # Phase 2: Theta emerges (A3/B1/B2) - Next 20s
            idx2 = int(2 * duration / 3 * sfreq)
            signal[:, idx1:idx2] += 5 * np.sin(2 * np.pi * 10 * times[idx1:idx2])
            signal[:, idx1:idx2] += 8 * np.sin(2 * np.pi * 6 * times[idx1:idx2]) # 6Hz Theta
            # Phase 3: Theta dominant (B3/C) - Last 20s
            signal[:, idx2:] += 2 * np.sin(2 * np.pi * 10 * times[idx2:])
            signal[:, idx2:] += 12 * np.sin(2 * np.pi * 6 * times[idx2:])
            # Add noise
            signal += np.random.randn(len(ch_names), n_samples) * 2 # Noise

            info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
            simulated_raw = mne.io.RawArray(signal, info, verbose=False)
            logger.info(f"Simulated MNE Raw created ({duration}s, {sfreq}Hz, {len(ch_names)} channels).")

            # --- Compute Vigilance States ---
            epoch_len_sec = 2.0
            vigilance_channel = 'Oz' # Common choice for vigilance
            computed_vigilance_states = compute_vigilance_states(
                simulated_raw, epoch_length=epoch_len_sec, channel_name=vigilance_channel
            )

            if computed_vigilance_states:
                logger.info(f"Computed {len(computed_vigilance_states)} vigilance states using '{vigilance_channel}'.")
                # --- Plot Vigilance ---
                output_plot_dir = Path("./vigilance_plots_demo")
                plot_condition = "SimulatedEO" # Example condition name
                logger.info(f"Attempting to save vigilance plots to: {output_plot_dir.resolve()}")
                plot_vigilance_hypnogram(
                    computed_vigilance_states,
                    output_dir=output_plot_dir,
                    condition=plot_condition,
                    epoch_length=epoch_len_sec
                )
            else:
                logger.warning("Vigilance computation did not return any states.")

        except Exception as e:
            logger.error(f"Error during vigilance simulation/plotting: {e}", exc_info=True)
            computed_vigilance_states = None # Ensure it's None if error occurred

    else:
        logger.warning("Skipping vigilance demo as MNE/Matplotlib are not installed.")

    # === Mapping Demo (Using Simulated Metrics) ===
    logger.info("\n=== Pyramid Mapping Demo ===")
    # --- Simulate Metrics (as if computed from real data) ---
    # Make these more realistic/varied
    sample_bp_EO = {
        'Fz': {'Delta': 8, 'Theta': 18, 'Alpha': 15, 'Beta': 22, 'HighBeta': 12},
        'Cz': {'Delta': 6, 'Theta': 15, 'Alpha': 20, 'Beta': 28, 'HighBeta': 14},
        'Pz': {'Delta': 4, 'Theta': 12, 'Alpha': 35, 'Beta': 25, 'HighBeta': 10},
        'Oz': {'Delta': 3, 'Theta': 10, 'Alpha': 45, 'Beta': 20, 'HighBeta': 7}
    }
    sample_bp_EC = { # Assume higher alpha EC
        'Fz': {'Delta': 9, 'Theta': 19, 'Alpha': 25, 'Beta': 23, 'HighBeta': 13},
        'Cz': {'Delta': 7, 'Theta': 16, 'Alpha': 30, 'Beta': 29, 'HighBeta': 15},
        'Pz': {'Delta': 5, 'Theta': 13, 'Alpha': 55, 'Beta': 26, 'HighBeta': 11},
        'Oz': {'Delta': 4, 'Theta': 11, 'Alpha': 65, 'Beta': 21, 'HighBeta': 8}
    }
    sample_site_metrics = {
        'Fz': {'Theta_Beta_Ratio': sample_bp_EO['Fz']['Theta'] / sample_bp_EO['Fz']['Beta'], 'PAF': 9.2},
        'Cz': {'Theta_Beta_Ratio': sample_bp_EO['Cz']['Theta'] / sample_bp_EO['Cz']['Beta'], 'PAF': 9.6},
        'Pz': {'Theta_Beta_Ratio': sample_bp_EO['Pz']['Theta'] / sample_bp_EO['Pz']['Beta'], 'PAF': 10.1},
        'Oz': {'Theta_Beta_Ratio': sample_bp_EO['Oz']['Theta'] / sample_bp_EO['Oz']['Beta'], 'PAF': 10.3}
    }
    sample_global_metrics = {
        'Frontal_Asymmetry': 1.1,   # Example F4/F3 Alpha ratio
        'Coherence_F3_F4': 0.55,    # Example coherence value
        'Coherence_Fz_Pz': 0.65,    # Example coherence value
        'vigilance_index': 0.6     # Example legacy index (might be overridden by computed states)
    }
    logger.info("Using simulated band powers and metrics for mapping demo.")

    # --- Perform Mapping ---
    pyramid_mappings = map_to_pyramid(
        sample_bp_EO, sample_bp_EC, sample_site_metrics, sample_global_metrics,
        vigilance_states=computed_vigilance_states # Pass computed states (or None)
    )

    # --- Display Mapping Results ---
    logger.info("\n--- Pyramid Mapping Results ---")
    for line in pyramid_mappings:
        # Indent lines that are part of sub-sections for readability
        if line.strip().startswith(("-", "Stage", "% Low", "EEG", "Cognitive", "Clinical", "Protocols", "Differentiators", "Vigilance Stage", "Neurofeedback", "Global", "Fz", "Cz", "Pz", "Oz")):
            logger.info(f"  {line.strip()}")
        else:
            logger.info(line)
    logger.info("--- End of Mapping Results ---\n")


    # === Other Examples (Briefly show retrieval functions) ===
    logger.info("=== Other Examples ===")

    # Example: Retrieve condition differentiation for 'TBI'
    condition = "TBI"
    cond_info = get_condition_differentiation(condition)
    if cond_info:
        logger.info(f"Differentiation Info for {condition}:")
        logger.info(f"  Interpretation: {cond_info.get('interpretation', 'N/A')}")

    logger.info("\nDemo finished.")
