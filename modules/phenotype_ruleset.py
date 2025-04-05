# phenotype_ruleset.py

rules = [
    # ----------------- ADHD -----------------
    {
        "name": "ADHD_combined",
        "condition": lambda f: f.get("theta_beta", 0) > 3.5 and f.get("paf", 10.0) < 8.0,
        "confidence": 0.9,
        "recommendations": ["Train SMR at Cz", "Theta down-train at Fz"],
        "explanation": "High Theta/Beta and low PAF consistent with ADHD pattern."
    },
    {
        "name": "ADHD_inattentive_variant",
        "condition": lambda f: f.get("theta", 0) > 3.5 and f.get("beta", 0) < 1.5 and f.get("hi_beta", 0) < 1.2,
        "confidence": 0.88,
        "recommendations": ["Boost mid-beta (15–20 Hz)", "Theta suppression"],
        "explanation": "Inattentive type often shows high theta and low beta range activity."
    },

    # ----------------- TBI -----------------
    {
        "name": "TBI_diffuse",
        "condition": lambda f: f.get("coherence_global", 0.4) < 0.2 and f.get("delta_power", 0) > 3.0,
        "confidence": 0.93,
        "recommendations": ["Delta suppression protocols", "Frontal coherence repair"],
        "explanation": "Low coherence and elevated delta suggest diffuse TBI."
    },

    # ----------------- Depression -----------------
    {
        "name": "Depression_alpha_suppression",
        "condition": lambda f: f.get("alpha_power", 0) < 1.0 and f.get("delta_power", 0) > 3.5,
        "confidence": 0.86,
        "recommendations": ["Alpha uptraining", "10 Hz stimulation"],
        "explanation": "Suppressed alpha and elevated delta indicate cortical underactivation typical of depression."
    },

    # ----------------- Frontal Asymmetry -----------------
    {
        "name": "Depression_frontal_asymmetry",
        "condition": lambda f: f.get("alpha_asymmetry_f3_f4", 0) > 0.5,
        "confidence": 0.84,
        "recommendations": ["Reduce left alpha at F3", "Approach bias training"],
        "explanation": "High left alpha asymmetry often corresponds to depressive emotional valence."
    },

    # ----------------- Alpha Instability -----------------
    {
        "name": "Alpha_peak_instability",
        "condition": lambda f: abs(f.get("paf_ec", 10.0) - f.get("paf_eo", 10.0)) > 1.0,
        "confidence": 0.81,
        "recommendations": ["PAF entrainment and stabilization"],
        "explanation": "Large PAF shift between eyes closed and open suggests instability, often in trauma or neurodegeneration."
    },

    # ----------------- EO/EC Reactivity -----------------
    {
        "name": "Blunted_alpha_shift",
        "condition": lambda f: f.get("alpha_shift", 0) < 20.0,
        "confidence": 0.80,
        "recommendations": ["Reinforce alpha reactivity via EC/EO contrast tasks"],
        "explanation": "Blunted alpha drop from EC to EO indicates poor cortical responsiveness."
    },

    # ----------------- High Alpha Dominance -----------------
    {
        "name": "Alpha_overdominance",
        "condition": lambda f: f.get("high_alpha", 0) > 4.0 and f.get("beta", 0) < 1.0,
        "confidence": 0.83,
        "recommendations": ["Reduce excess alpha, encourage task engagement"],
        "explanation": "High alpha with low beta may reflect shutdown or avoidant mental states."
    },

    # ----------------- Mirror Neuron / Mu Rhythm -----------------
    {
        "name": "ASD_mu_suppression_deficit",
        "condition": lambda f: f.get("mu_suppression_ratio", 1.0) < 0.7,
        "confidence": 0.85,
        "recommendations": ["SMR uptrain", "Social mimicry engagement tasks"],
        "explanation": "Low mu rhythm suppression can indicate mirror neuron dysfunction, often found in ASD."
    },

    # ----------------- Sleep Intrusion -----------------
    {
        "name": "Sleep_intrusion_theta_delta",
        "condition": lambda f: f.get("theta", 0) > 4.5 and f.get("delta", 0) > 4.0 and f.get("smr", 0) < 1.0,
        "confidence": 0.88,
        "recommendations": ["Stabilize arousal", "Reduce intrusions with SMR training"],
        "explanation": "Theta/delta dominance while awake may suggest intrusion from sleep states or trauma fatigue."
    }
]
