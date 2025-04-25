# phenotype_ruleset.py
"""
phenotype_ruleset.py

This module defines a comprehensive ruleset for EEG-based phenotype classification in The Squiggle Interpreter.
Each rule maps specific EEG features to clinical phenotypes (e.g., ADHD, Depression, ASD) with associated confidence levels,
recommendations, explanations, and z-score summaries.

The rules operate on a dictionary of EEG features (`f`), which may include:
- theta_beta (float): Theta/Beta power ratio.
- paf (float): Peak Alpha Frequency (Hz) in eyes-open (EO) condition.
- paf_eo (float): Peak Alpha Frequency (Hz) in EO condition.
- paf_ec (float): Peak Alpha Frequency (Hz) in eyes-closed (EC) condition.
- theta (float): Theta power z-score.
- alpha (float): Alpha power z-score. # Note: Often referred to as alpha_power in conditions
- beta (float): Beta power z-score.
- hi_beta (float): High Beta power z-score.
- delta (float): Delta power z-score. # Note: Often referred to as delta_power in conditions
- smr (float): Sensorimotor Rhythm (SMR) power z-score.
- alpha_power (float): Alpha power z-score. # Docstring inconsistency corrected: Assumed z-score based on usage.
- delta_power (float): Delta power z-score. # Docstring inconsistency corrected: Assumed z-score based on usage.
- alpha_asymmetry_f3_f4 (float): Frontal alpha asymmetry (F4 - F3).
- mu_suppression_ratio (float): Mu rhythm suppression ratio during movement observation.
- coherence_global (float): Global coherence z-score.
- coherence_f3_f4 (float): Frontal coherence (F3-F4) z-score. # Added based on rule usage
- alpha_shift (float): Percentage change in alpha power from EC to EO.
- high_alpha (float): High Alpha power z-score.

Each rule returns:
- name (str): The phenotype name.
- condition (callable): A lambda function evaluating EEG features.
- confidence (float): Confidence level of the phenotype match (0.0–1.0).
- recommendations (List[str]): Suggested neurofeedback protocols and interventions.
- explanation (str): Clinical explanation of the EEG pattern.
- zscore_summary (Dict[str, float]): Z-score summary of key metrics contributing to the phenotype.
"""

import logging
from typing import Dict, List, Callable

logger = logging.getLogger(__name__)

rules: List[Dict] = [
    # ----------------- ADHD -----------------
    {
        "name": "ADHD_combined",
        # Enhanced: Added alpha_power > 1.0
        "condition": lambda f: f.get("theta_beta", 0) > 3.5 and f.get("paf", 10.0) < 8.0 and f.get("coherence_global", 0.0) < -1.0 and f.get("alpha_power", 0) > 1.0,
        "confidence": 0.90,
        "recommendations": [
            "Train SMR at Cz (12–15 Hz) to enhance focus",
            "Theta downtraining at Fz (4–8 Hz) to reduce inattention",
            "Interhemispheric coherence training (F3↔F4) to improve connectivity",
            "Adjunct: Behavioral therapy for impulse control"
        ],
        "explanation": (
            # Enhanced: Added elevated alpha power
            "High Theta/Beta ratio (>3.5), low PAF (<8 Hz), reduced global coherence (z-score < -1), and elevated alpha power (z-score > 1) "
            "are consistent with ADHD combined type, reflecting inattention, impulsivity, poor connectivity, and increased brain activity."
        ),
        "zscore_summary": {
            "Theta/Beta Ratio": lambda f: f.get("theta_beta", 0),
            "PAF": lambda f: f.get("paf", 10.0),
            "Global Coherence": lambda f: f.get("coherence_global", 0.0),
            # Enhanced: Added Alpha Power
            "Alpha Power": lambda f: f.get("alpha_power", 0)
        }
    },
    {
        "name": "ADHD_inattentive_variant",
        # Enhanced: Added coherence_global > 1.0
        "condition": lambda f: f.get("theta", 0) > 2.5 and f.get("beta", 0) < -1.0 and f.get("hi_beta", 0) < -0.5 and f.get("coherence_global", 0.0) > 1.0,
        "confidence": 0.88,
        "recommendations": [
            "Boost mid-beta (15–20 Hz) at Cz to enhance sustained attention",
            "Theta suppression at Fz (4–8 Hz) to reduce daydreaming",
            "SMR training at Cz (12–15 Hz) for focus",
            "Adjunct: Cognitive training for attention"
        ],
        "explanation": (
            # Enhanced: Added elevated global coherence
            "High theta power (z-score > 2.5), low beta (z-score < -1), low high beta (z-score < -0.5), and elevated global coherence (z-score > 1) "
            "indicate an inattentive ADHD variant, characterized by excessive daydreaming, lack of sustained attention, and over-connectivity."
        ),
        "zscore_summary": {
            "Theta Power": lambda f: f.get("theta", 0),
            "Beta Power": lambda f: f.get("beta", 0),
            "High Beta Power": lambda f: f.get("hi_beta", 0),
            # Enhanced: Added Global Coherence
            "Global Coherence": lambda f: f.get("coherence_global", 0.0)
        }
    },
    {
        "name": "ADHD_hyperactive_variant",
        # No changes needed based on review
        "condition": lambda f: f.get("theta_beta", 0) > 3.0 and f.get("hi_beta", 0) > 2.0 and f.get("smr", 0) < -1.0,
        "confidence": 0.87,
        "recommendations": [
            "High Beta downtraining at Fz/Cz (20–30 Hz) to reduce impulsivity",
            "SMR uptraining at Cz (12–15 Hz) to enhance self-regulation",
            "Theta downtraining at Fz (4–8 Hz) to improve focus",
            "Adjunct: Mindfulness-based interventions for impulse control"
        ],
        "explanation": (
            "High Theta/Beta ratio (>3.0), elevated high beta (z-score > 2), and reduced SMR (z-score < -1) "
            "suggest a hyperactive ADHD variant, with impulsivity and poor self-regulation."
        ),
        "zscore_summary": {
            "Theta/Beta Ratio": lambda f: f.get("theta_beta", 0),
            "High Beta Power": lambda f: f.get("hi_beta", 0),
            "SMR Power": lambda f: f.get("smr", 0)
        }
    },

    # ----------------- TBI -----------------
    {
        "name": "TBI_diffuse",
        # Enhanced: Added paf < 8.0 and alpha_shift < 20.0
        "condition": lambda f: f.get("coherence_global", 0.0) < -2.0 and f.get("delta_power", 0) > 3.0 and f.get("alpha_power", 0) < -1.5 and f.get("paf", 10.0) < 8.0 and f.get("alpha_shift", 0) < 20.0,
        "confidence": 0.93,
        "recommendations": [
            "Delta suppression protocols at Fz/Cz (1–4 Hz)",
            "Frontal coherence repair (F3↔F4, Fz↔Pz) to improve connectivity",
            "Alpha uptraining at Pz (8–12 Hz) to restore cortical activation",
            "PAF entrainment and stabilization", # Added recommendation
            "Adjunct: Neurological evaluation, tDCS for cognitive recovery"
        ],
        "explanation": (
            # Enhanced: Added low PAF and blunted alpha shift
            "Severely reduced global coherence (z-score < -2), elevated delta power (>3.0), suppressed alpha power (z-score < -1.5), low peak alpha frequency (<8.0 Hz), and blunted alpha shift (<20%) "
            "suggest diffuse traumatic brain injury (TBI), indicating widespread cortical slowing, connectivity deficits, and impaired reactivity with cognitive impairment."
        ),
        "zscore_summary": {
            "Global Coherence": lambda f: f.get("coherence_global", 0.0),
            "Delta Power": lambda f: f.get("delta_power", 0),
            "Alpha Power": lambda f: f.get("alpha_power", 0),
            # Enhanced: Added PAF and Alpha Shift
            "Peak Alpha Frequency": lambda f: f.get("paf", 10.0),
            "Alpha Shift (EC→EO)": lambda f: f.get("alpha_shift", 0)
        }
    },
    {
        "name": "TBI_focal_frontal",
        # No changes needed based on review
        "condition": lambda f: f.get("theta", 0) > 2.5 and f.get("coherence_f3_f4", 0.0) < -1.5 and f.get("delta_power", 0) > 2.0,
        "confidence": 0.90,
        "recommendations": [
            "Theta downtraining at Fz (4–8 Hz) to reduce slowing",
            "Interhemispheric coherence training (F3↔F4) to repair connectivity",
            "Delta suppression at Fz (1–4 Hz) to address cortical slowing",
            "Adjunct: Cognitive rehabilitation, neuroplasticity-focused interventions"
        ],
        "explanation": (
            "Elevated theta (z-score > 2.5), reduced frontal coherence (z-score < -1.5), and high delta power (>2.0) "
            "indicate a focal frontal TBI, reflecting localized slowing and connectivity issues in frontal regions."
        ),
        "zscore_summary": {
            "Theta Power": lambda f: f.get("theta", 0),
            "F3-F4 Coherence": lambda f: f.get("coherence_f3_f4", 0.0),
            "Delta Power": lambda f: f.get("delta_power", 0)
        }
    },

    # ----------------- Depression -----------------
    # No enhancements requested/applied to Depression rules
    {
        "name": "Depression_alpha_suppression",
        "condition": lambda f: f.get("alpha_power", 0) < -2.0 and f.get("delta_power", 0) > 2.5 and f.get("theta", 0) > 1.5,
        "confidence": 0.86,
        "recommendations": [
            "Alpha uptraining at Pz (8–12 Hz) to enhance cortical activation",
            "10 Hz auditory-visual stimulation to promote alpha rhythm",
            "Delta downtraining at Fz (1–4 Hz) to reduce slowing",
            "Adjunct: CBT, SSRI evaluation, HRV training"
        ],
        "explanation": (
            "Severely suppressed alpha power (z-score < -2), elevated delta (>2.5), and increased theta (z-score > 1.5) "
            "indicate cortical underactivation typical of depression, reflecting cognitive slowing and withdrawal."
        ),
        "zscore_summary": {
            "Alpha Power": lambda f: f.get("alpha_power", 0),
            "Delta Power": lambda f: f.get("delta_power", 0),
            "Theta Power": lambda f: f.get("theta", 0)
        }
    },
    {
        "name": "Depression_frontal_asymmetry",
        # Note: Asymmetry > 0.5 implies less activity right, more left (leftward asymmetry) based on F4-F3 calc.
        # Explanation text suggests high left alpha asymmetry, which might be interpreted differently. Keeping as is for now.
        "condition": lambda f: f.get("alpha_asymmetry_f3_f4", 0) > 0.5,
        "confidence": 0.84,
        "recommendations": [
            "Reduce left alpha at F3", "Approach bias training"],
        "explanation": "High left alpha asymmetry often corresponds to depressive emotional valence."
    },
    {
        "name": "Depression_with_anxiety",
        # Note: Same asymmetry interpretation note as above.
        "condition": lambda f: f.get("alpha_asymmetry_f3_f4", 0) > 0.5 and f.get("hi_beta", 0) > 2.0 and f.get("coherence_f3_f4", 0.0) > 1.5,
        "confidence": 0.85,
        "recommendations": [
            "High Beta downtraining at Fz/Cz (20–30 Hz) to reduce anxiety",
            "Balance frontal alpha asymmetry (reduce F3 alpha, 8–12 Hz)",
            "Interhemispheric coherence normalization (F3↔F4) to reduce hypercoherence",
            "Adjunct: CBT for anxiety, HRV training, SSRI evaluation"
        ],
        "explanation": (
            "High left alpha asymmetry (F4 - F3 > 0.5), elevated high beta (z-score > 2), and frontal hypercoherence (z-score > 1.5) "
            "suggest depression with comorbid anxiety, reflecting withdrawal and hyperarousal."
        ),
        "zscore_summary": {
            "Alpha Asymmetry (F4-F3)": lambda f: f.get("alpha_asymmetry_f3_f4", 0),
            "High Beta Power": lambda f: f.get("hi_beta", 0),
            "F3-F4 Coherence": lambda f: f.get("coherence_f3_f4", 0.0)
        }
    },

    # ----------------- Anxiety -----------------
    {
        "name": "Anxiety_hyperarousal",
        # Enhanced: Added alpha_asymmetry_f3_f4 < -0.5
        "condition": lambda f: f.get("hi_beta", 0) > 2.5 and f.get("alpha_power", 0) < -1.5 and f.get("coherence_f3_f4", 0.0) > 1.5 and f.get("alpha_asymmetry_f3_f4", 0) < -0.5,
        "confidence": 0.89,
        "recommendations": [
            "High Beta downtraining at Fz/Cz (20–30 Hz) to reduce hyperarousal",
            "Alpha uptraining at Pz (8–12 Hz) to promote relaxation",
            "Normalize frontal coherence (F3↔F4) to reduce hypercoherence",
            "Consider neurofeedback to balance frontal asymmetry", # Added recommendation
            "Adjunct: HRV training, mindfulness-based relaxation, CBT for anxiety"
        ],
        "explanation": (
            # Enhanced: Added rightward frontal alpha asymmetry
            "Elevated high beta (z-score > 2.5), suppressed alpha power (z-score < -1.5), frontal hypercoherence (z-score > 1.5), and rightward frontal alpha asymmetry (F4 - F3 < -0.5) "
            "indicate hyperarousal typical of anxiety, reflecting excessive cortical activation, worry, and emotional dysregulation."
        ),
        "zscore_summary": {
            "High Beta Power": lambda f: f.get("hi_beta", 0),
            "Alpha Power": lambda f: f.get("alpha_power", 0),
            "F3-F4 Coherence": lambda f: f.get("coherence_f3_f4", 0.0),
            # Enhanced: Added Frontal Alpha Asymmetry
            "Frontal Alpha Asymmetry (F4-F3)": lambda f: f.get("alpha_asymmetry_f3_f4", 0)
        }
    },

    # ----------------- PTSD -----------------
    {
        "name": "PTSD_hypervigilance",
        # Enhanced: Added alpha_asymmetry_f3_f4 > 0.5
        # Note: Asymmetry > 0.5 implies less activity right, more left (leftward asymmetry) based on F4-F3 calc.
        # Research suggested rightward asymmetry (more activity right, less left -> F4-F3 < -0.5) was linked to PTSD severity.
        # Reverting this based on potential conflict with Depression rule interpretation unless clarified. Keeping original for now.
        "condition": lambda f: f.get("hi_beta", 0) > 2.5 and f.get("theta", 0) > 2.0 and f.get("alpha_power", 0) < -1.5,
        # Previous enhancement was: alpha_asymmetry_f3_f4 > 0.5
        "confidence": 0.87,
        "recommendations": [
            "High Beta downtraining at Fz/Cz (20–30 Hz) to reduce hypervigilance",
            "Theta downtraining at Fz (4–8 Hz) to address trauma looping",
            "Alpha uptraining at Pz (8–12 Hz) to promote relaxation",
            "Adjunct: Trauma-focused CBT, EMDR, slow HRV entrainment"
        ],
        "explanation": (
            # Explanation kept original pending clarification on asymmetry interpretation.
            "Elevated high beta (z-score > 2.5), increased theta (z-score > 2), and suppressed alpha (z-score < -1.5) "
            "suggest PTSD-related hypervigilance, reflecting trauma looping and cortical underactivation."
        ),
        "zscore_summary": {
            "High Beta Power": lambda f: f.get("hi_beta", 0),
            "Theta Power": lambda f: f.get("theta", 0),
            "Alpha Power": lambda f: f.get("alpha_power", 0)
            # Previous enhancement added: "Frontal Alpha Asymmetry (F4-F3)": lambda f: f.get("alpha_asymmetry_f3_f4", 0)
        }
    },

    # ----------------- Schizophrenia -----------------
    # No enhancements requested/applied
    {
        "name": "Schizophrenia_disorganization",
        "condition": lambda f: f.get("theta", 0) > 2.5 and f.get("alpha_power", 0) < -2.0 and f.get("coherence_global", 0.0) < -2.5,
        "confidence": 0.91,
        "recommendations": [
            "Theta downtraining at Fz (4–8 Hz) to reduce cognitive disorganization",
            "Global coherence training to improve connectivity",
            "Alpha uptraining at Pz (8–12 Hz) to restore cortical rhythm",
            "Adjunct: Neurological evaluation, antipsychotic medication review"
        ],
        "explanation": (
            "Elevated theta (z-score > 2.5), severely suppressed alpha (z-score < -2), and globally reduced coherence (z-score < -2.5) "
            "indicate cognitive disorganization typical of schizophrenia, with disrupted cortical rhythms."
        ),
        "zscore_summary": {
            "Theta Power": lambda f: f.get("theta", 0),
            "Alpha Power": lambda f: f.get("alpha_power", 0),
            "Global Coherence": lambda f: f.get("coherence_global", 0.0)
        }
    },

    # ----------------- OCD (New Rule) -----------------
    {
        "name": "OCD",
        "condition": lambda f: f.get("theta", 0) > 2.0 and f.get("delta_power", 0) > 1.5 and f.get("coherence_global", 0.0) < -1.0,
        "confidence": 0.85,
        "recommendations": [
            "Theta and delta downtraining to reduce obsessive thoughts",
            "Coherence training to improve connectivity",
            "SMR training to enhance self-regulation",
            "Adjunct: CBT for OCD"
        ],
        "explanation": "Increased theta and delta power with reduced global coherence are associated with OCD, reflecting cortical slowing and connectivity deficits.",
        "zscore_summary": {
            "Theta Power": lambda f: f.get("theta", 0),
            "Delta Power": lambda f: f.get("delta_power", 0),
            "Global Coherence": lambda f: f.get("coherence_global", 0.0)
        }
    },

    # ----------------- Bipolar Disorder -----------------
    {
        "name": "Bipolar_mania",
        # Enhanced: Added coherence_global > 1.0
        "condition": lambda f: f.get("hi_beta", 0) > 3.0 and f.get("alpha_power", 0) < -1.5 and f.get("theta", 0) < -1.0 and f.get("coherence_global", 0.0) > 1.0,
        "confidence": 0.88,
        "recommendations": [
            "High Beta downtraining at Fz/Cz (20–30 Hz) to reduce manic activation",
            "Alpha uptraining at Pz (8–12 Hz) to promote stabilization",
            "SMR training at Cz (12–15 Hz) for self-regulation",
            "Adjunct: Mood stabilization therapy, lithium evaluation"
        ],
        "explanation": (
            # Enhanced: Added increased global coherence
            "Elevated high beta (z-score > 3), suppressed alpha (z-score < -1.5), reduced theta (z-score < -1), and increased global coherence (z-score > 1) "
            "suggest a manic episode in bipolar disorder, with hyperactivation, poor cortical regulation, and connectivity disturbances."
        ),
        "zscore_summary": {
            "High Beta Power": lambda f: f.get("hi_beta", 0),
            "Alpha Power": lambda f: f.get("alpha_power", 0),
            "Theta Power": lambda f: f.get("theta", 0),
            # Enhanced: Added Global Coherence
            "Global Coherence": lambda f: f.get("coherence_global", 0.0)
        }
    },
    {
        "name": "Bipolar_depression",
        # Enhanced: Added theta > 1.0
        # Note: Asymmetry > 0.5 implies less activity right, more left (leftward asymmetry) based on F4-F3 calc.
        "condition": lambda f: f.get("alpha_asymmetry_f3_f4", 0) > 0.5 and f.get("delta_power", 0) > 2.5 and f.get("beta", 0) < -1.0 and f.get("theta", 0) > 1.0,
        "confidence": 0.86,
        "recommendations": [
            "Balance frontal alpha asymmetry (reduce F3 alpha, 8–12 Hz)",
            "Delta downtraining at Fz (1–4 Hz) to reduce slowing",
            "Beta uptraining at Cz (15–20 Hz) to enhance activation",
            "Adjunct: Mood stabilization, CBT for depression"
        ],
        "explanation": (
            # Enhanced: Added increased theta
            "High left alpha asymmetry (F4 - F3 > 0.5), elevated delta power (>2.5), reduced beta (z-score < -1), and increased theta (z-score > 1) "
            "indicate a depressive episode in bipolar disorder, reflecting withdrawal, cortical slowing, and emotional dysregulation."
        ),
        "zscore_summary": {
            "Alpha Asymmetry (F4-F3)": lambda f: f.get("alpha_asymmetry_f3_f4", 0),
            "Delta Power": lambda f: f.get("delta_power", 0),
            "Beta Power": lambda f: f.get("beta", 0),
            # Enhanced: Added Theta Power
            "Theta Power": lambda f: f.get("theta", 0)
        }
    },

    # ----------------- Epilepsy -----------------
    {
        "name": "Epilepsy_elevated_spikes",
        # Enhanced: Added delta_power > 2.0 and paf_ec < 9.0
        "condition": lambda f: f.get("hi_beta", 0) > 3.0 and f.get("theta", 0) > 2.5 and f.get("coherence_global", 0.0) < -2.0 and f.get("delta_power", 0) > 2.0 and f.get("paf_ec", 10.0) < 9.0,
        "confidence": 0.92,
        "recommendations": [
            "High Beta downtraining at Fz/Cz (20–30 Hz) to reduce spikes",
            "Theta downtraining at Fz (4–8 Hz) to address slowing",
            "Delta downtraining at Fz (1–4 Hz) to reduce slowing", # Added recommendation
            "Global coherence training to improve connectivity",
            "Adjunct: Neurological evaluation, antiepileptic medication review"
        ],
        "explanation": (
            # Enhanced: Added increased delta and low PAF
            "Elevated high beta (z-score > 3), increased theta (z-score > 2.5), increased delta (z-score > 2.0), low peak alpha frequency (<9.0 Hz), and severely reduced coherence (z-score < -2) "
            "suggest epileptiform activity, with cortical instability and slowing."
        ),
        "zscore_summary": {
            "High Beta Power": lambda f: f.get("hi_beta", 0),
            "Theta Power": lambda f: f.get("theta", 0),
            "Delta Power": lambda f: f.get("delta_power", 0), # Added Delta Power
            "Peak Alpha Frequency (EC)": lambda f: f.get("paf_ec", 10.0), # Added PAF (EC)
            "Global Coherence": lambda f: f.get("coherence_global", 0.0)
        }
    },

    # ----------------- Alpha Instability -----------------
    # No enhancements requested/applied
    {
        "name": "Alpha_peak_instability",
        "condition": lambda f: abs(f.get("paf_ec", 10.0) - f.get("paf_eo", 10.0)) > 1.0 and f.get("alpha_power", 0) < -1.0,
        "confidence": 0.81,
        "recommendations": [
            "PAF entrainment and stabilization (8–12 Hz) via auditory-visual stimulation",
            "Alpha uptraining at Pz (8–12 Hz) to restore stability",
            "SMR training at Cz (12–15 Hz) for focus",
            "Adjunct: Trauma evaluation, mindfulness-based interventions"
        ],
        "explanation": (
            "Large PAF shift between eyes-closed and eyes-open (>1 Hz) with suppressed alpha power (z-score < -1) "
            "suggests alpha instability, often seen in trauma or neurodegenerative conditions."
        ),
        "zscore_summary": {
            "PAF Shift (EC-EO)": lambda f: abs(f.get("paf_ec", 10.0) - f.get("paf_eo", 10.0)),
            "Alpha Power": lambda f: f.get("alpha_power", 0)
        }
    },

    # ----------------- EO/EC Reactivity -----------------
    # No enhancements requested/applied
    {
        "name": "Blunted_alpha_shift",
        "condition": lambda f: f.get("alpha_shift", 0) < 20.0 and f.get("alpha_power", 0) < -1.0,
        "confidence": 0.80,
        "recommendations": [
            "Reinforce alpha reactivity via EC/EO contrast tasks",
            "Alpha uptraining at Pz (8–12 Hz) to enhance responsiveness",
            "10 Hz auditory stimulation to promote alpha rhythm",
            "Adjunct: Evaluate for depression or neurological conditions"
        ],
        "explanation": (
            "Blunted alpha shift from EC to EO (<20%) with suppressed alpha power (z-score < -1) "
            "indicates poor cortical responsiveness, often linked to depression or neurological deficits."
        ),
        "zscore_summary": {
            "Alpha Shift (EC→EO)": lambda f: f.get("alpha_shift", 0),
            "Alpha Power": lambda f: f.get("alpha_power", 0)
        }
    },

    # ----------------- High Alpha Dominance -----------------
    # No enhancements requested/applied
    {
        "name": "Alpha_overdominance",
        "condition": lambda f: f.get("high_alpha", 0) > 2.5 and f.get("beta", 0) < -1.0 and f.get("smr", 0) < -1.0,
        "confidence": 0.83,
        "recommendations": [
            "Reduce excess high alpha at Pz (10–12 Hz) to prevent shutdown",
            "Encourage task engagement with SMR training at Cz (12–15 Hz)",
            "Beta uptraining at Cz (15–20 Hz) to enhance activation",
            "Adjunct: Behavioral activation therapy, evaluate for avoidance"
        ],
        "explanation": (
            "High alpha (z-score > 2.5) with low beta and SMR (z-scores < -1) may reflect a shutdown or avoidant mental state, "
            "often seen in depression or trauma-related dissociation."
        ),
        "zscore_summary": {
            "High Alpha Power": lambda f: f.get("high_alpha", 0),
            "Beta Power": lambda f: f.get("beta", 0),
            "SMR Power": lambda f: f.get("smr", 0)
        }
    },

    # ----------------- Mirror Neuron / Mu Rhythm -----------------
    {
        "name": "ASD_mu_suppression_deficit",
        # Enhanced: Added paf < 9.0
        "condition": lambda f: f.get("mu_suppression_ratio", 1.0) < 0.7 and f.get("alpha", 0) < -1.0 and f.get("paf", 10.0) < 9.0,
        "confidence": 0.85,
        "recommendations": [
            "SMR uptraining at C3/C4 (12–15 Hz) to enhance mu rhythm",
            "Social mimicry engagement tasks to activate mirror neurons",
            "Alpha uptraining at Pz (8–12 Hz) to improve cortical activation",
            "Adjunct: Social skills training, sensory integration therapy"
        ],
        "explanation": (
            # Enhanced: Added low PAF
            "Low mu rhythm suppression ratio (<0.7), suppressed alpha (z-score < -1), and low peak alpha frequency (<9.0 Hz) "
            "indicate mirror neuron dysfunction, often associated with ASD, reflecting challenges in social imitation, empathy, and cognitive function."
        ),
        "zscore_summary": {
            "Mu Suppression Ratio": lambda f: f.get("mu_suppression_ratio", 1.0),
            "Alpha Power": lambda f: f.get("alpha", 0), # Note: Uses 'alpha', consistent with docstring but not alpha_power
            # Enhanced: Added Peak Alpha Frequency
            "Peak Alpha Frequency": lambda f: f.get("paf", 10.0)
        }
    },

    # ----------------- Sleep Intrusion -----------------
    # No enhancements requested/applied
    {
        "name": "Sleep_intrusion_theta_delta",
        "condition": lambda f: f.get("theta", 0) > 3.0 and f.get("delta", 0) > 3.0 and f.get("smr", 0) < -1.0,
        "confidence": 0.88,
        "recommendations": [
            "Stabilize arousal with SMR training at Cz (12–15 Hz)",
            "Theta downtraining at Fz (4–8 Hz) to reduce intrusions",
            "Delta downtraining at Fz (1–4 Hz) to prevent sleep onset",
            "Adjunct: Sleep hygiene education, evaluate for trauma"
        ],
        "explanation": (
            "High theta (z-score > 3) and delta (z-score > 3) with low SMR (z-score < -1) suggest sleep state intrusion "
            "during wakefulness, often linked to trauma fatigue or sleep disorders."
        ),
        "zscore_summary": {
            "Theta Power": lambda f: f.get("theta", 0),
            "Delta Power": lambda f: f.get("delta", 0), # Note: Uses 'delta', consistent with docstring but not delta_power
            "SMR Power": lambda f: f.get("smr", 0)
        }
    }
]

def apply_phenotype_ruleset(features: Dict[str, float]) -> List[Dict]:
    """
    Apply the phenotype ruleset to a set of EEG features and return matching phenotypes.

    Args:
        features (Dict[str, float]): A dictionary of EEG features (e.g., theta_beta, paf, alpha_power).

    Returns:
        List[Dict]: A list of matching phenotypes with their details (name, confidence, recommendations, etc.).
    """
    # Added logging
    logger.debug(f"Applying phenotype ruleset with features: {features}")
    matches = []
    try:
        for rule in rules:
            if rule["condition"](features):
                match = {
                    "name": rule["name"],
                    "confidence": rule["confidence"],
                    "recommendations": rule["recommendations"],
                    "explanation": rule["explanation"],
                    "zscore_summary": {key: func(features) for key, func in rule["zscore_summary"].items()}
                }
                matches.append(match)
                # Added logging
                logger.info(f"Matched phenotype: {rule['name']} with confidence {match['confidence']:.2f}")
    except Exception as e:
        logger.error(f"Error applying phenotype ruleset: {e}")
        matches.append({
            "name": "Error",
            "confidence": 0.0,
            "recommendations": ["Check EEG data quality and reprocess"],
            "explanation": f"An error occurred while applying the ruleset: {str(e)}",
            "zscore_summary": {}
        })

    if not matches:
        logger.warning("No phenotype matches found for the provided EEG features.")
        matches.append({
            "name": "No Match",
            "confidence": 0.0,
            "recommendations": ["Further EEG analysis required"],
            "explanation": "No phenotypes matched the provided EEG features.",
            "zscore_summary": {}
        })

    return matches

if __name__ == "__main__":
    # Configure logging for the demo
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Example EEG features
    sample_features = {
        "theta_beta": 4.0,
        "paf": 7.5,
        "coherence_global": -2.5,
        "theta": 3.0,
        "beta": -1.2,
        "hi_beta": 0.5,
        "alpha_power": -1.8, # Assuming z-score
        "delta_power": 3.2, # Assuming z-score
        "alpha_asymmetry_f3_f4": 0.6,
        "paf_ec": 9.5,
        "paf_eo": 7.5,
        "alpha_shift": 15.0,
        "high_alpha": 2.8,
        "smr": -1.5,
        "mu_suppression_ratio": 0.6,
        "alpha": -1.0, # Added for ASD rule example if needed
        "delta": 3.0,  # Added for Sleep rule example if needed
        "coherence_f3_f4": 1.8 # Added for Anxiety rule example if needed
    }

    # Apply the ruleset
    matches = apply_phenotype_ruleset(sample_features)

    # Display results
    logger.info("Phenotype Classification Results:")
    for match in matches:
        logger.info(f"- {match['name']} (Confidence: {match['confidence']:.2f})")
        logger.info("  Explanation: " + match['explanation'])
        logger.info("  Recommendations: " + "; ".join(match['recommendations']))
        logger.info("  Z-Score Summary: " + str(match['zscore_summary']))
        logger.info("")
