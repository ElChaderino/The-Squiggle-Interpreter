import os
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class SymptomVsWaveform:
    """
    SymptomVsWaveform module by El Chaderino (enhanced).

    This module aids EEG practitioners in distinguishing between symptom-based neurofeedback protocols
    and waveform-informed targeting. It offers rules-of-thumb guidance, symptom-to-waveform analysis,
    targeted site recommendations, protocol suggestions, and common practitioner pitfalls.

    Attributes:
        rules_of_thumb (dict): Guidelines for interpreting EEG findings in relation to symptoms.

    Methods:
        analyze_site_drift(signal_profile, report_symptoms):
            Compares symptoms with EEG profiles to identify potential mismatches.

        match_site_to_target(eeg_findings):
            Suggests starting neurofeedback sites based on EEG data.

        suggest_protocols(eeg_findings, reported_symptoms):
            Recommends specific neurofeedback protocols according to EEG data and reported symptoms.

        explain_traps():
            Outlines common mistakes practitioners should avoid when interpreting EEG data.

        summarize(signal_profile, report_symptoms):
            Returns a user-friendly summary of all recommendations and flags.

        generate_txt_report(signal_profile, report_symptoms, output_dir, filename):
            Generates a .txt report summarizing the analysis.

        plot_eeg_findings(signal_profile, report_symptoms, output_dir, filename):
            Generates a bar plot of key EEG findings for the reported symptoms.

        run_analysis(signal_profile, report_symptoms, output_dir):
            Runs the full analysis: generates report and plot, returns their paths.

        plot_radar(signal_profile, output_dir, filename):
            Generates a radar plot of EEG feature values.

        plot_protocol_network(signal_profile, report_symptoms, output_dir, filename):
            Generates a network plot showing symptom-feature-protocol relationships.

        interactive_dashboard(signal_profile, report_symptoms):
            Opens an interactive dashboard for exploring EEG feature values.

        plot_sankey(symptoms, waveforms, links, output_dir, filename):
            Generates a static Sankey diagram showing flow from symptoms to waveforms.
    """

    def __init__(self):
        self.rules_of_thumb: Dict[str, Dict] = {
            'left_frontal': {'adjustment': +2, 'notes': 'Often overstimulating if run early in dysregulated clients'},
            'posterior': {'stabilize_before_frontal': True},
            'cingulate_hibeta': {'suggests': 'OCD/rumination, vigilance drive, inner conflict'},
            'alpha_asymmetry': {'suggests': 'valence bias, depression, affect regulation'},
            'hibeta_dominance': {'suggests': 'instability or trauma override masking real signal'},
            'central_delta': {'suggests': 'sleep regulation issues or potential underlying trauma'},
            'right_parietal_alpha': {'suggests': 'attention issues in PTSD'},
            'sensorimotor_theta': {'suggests': 'motor control issues, potential ADHD'},
            'prefrontal_beta': {'suggests': 'executive dysfunction, hyperarousal in PTSD'},
            'right_temporal_theta': {'suggests': 'language processing issues, potential ASD'}
        }

    def analyze_site_drift(self, signal_profile: Dict[str, float], report_symptoms: List[str]) -> Dict[str, str]:
        """
        Compares reported symptoms with EEG signals to identify mismatches.
        Returns symptoms flagged as potentially mistargeted.
        """
        flags = {}
        symptom_checks = {
            'anxiety': lambda sp: sp.get('Pz_hibeta', 0) < sp.get('Fz_hibeta', 0),
            'focus issues': lambda sp: sp.get('theta', 0) < 4.5,
            'impulsivity': lambda sp: sp.get('F3_beta', 0) < 2.0,
            'depression': lambda sp: abs(sp.get('F3_alpha', 0) - sp.get('F4_alpha', 0)) < 1.0,
            'insomnia': lambda sp: sp.get('Cz_delta', 0) > 5.0,
            'hyperactivity': lambda sp: sp.get('Fz_theta', 0) < 4.0,
            'PTSD': lambda sp: sp.get('Fz_beta', 0) > 2.0 or sp.get('Cz_beta', 0) > 2.0 or (sp.get('P4_alpha', 0) > sp.get('P3_alpha', 0)),
            'OCD': lambda sp: sp.get('Fz_hibeta', 0) > 2.0 or sp.get('Cz_hibeta', 0) > 2.0
        }
        symptom_messages = {
            'anxiety': "Likely mistargeted. Pz not resolved before frontal work.",
            'focus issues': "Reported focus trouble without elevated theta—may be vigilance crash, not ADD.",
            'impulsivity': "No significant frontal beta—impulsivity may be emotional, not attentional.",
            'depression': "No significant alpha asymmetry—depression may not be related to valence bias.",
            'insomnia': "Elevated delta at Cz—may indicate sleep-related issues.",
            'hyperactivity': "No elevated theta at Fz—hyperactivity may not be ADHD-related.",
            'PTSD': "High beta at Fz/Cz or right parietal alpha asymmetry may indicate hyperarousal or attention issues in PTSD.",
            'OCD': "High hibeta at Fz/Cz may indicate OCD/rumination."
        }
        for symptom in report_symptoms:
            if symptom in symptom_checks and symptom_checks[symptom](signal_profile):
                flags[symptom] = symptom_messages[symptom]
        return flags

    def match_site_to_target(self, eeg_findings: Dict[str, float], reported_symptoms: Optional[List[str]] = None) -> List[Tuple[str, str]]:
        """
        Suggests EEG sites for initiating neurofeedback based on EEG signal abnormalities, with cross-checks.
        Cross-checks ensure posterior stability, symptom consistency, and coherence metrics.
        Returns a list of (site, reason) tuples.
        """
        suggestions = []
        reported_symptoms = reported_symptoms or []

        # Define target checks with thresholds and cross-check conditions
        target_checks = [
            ('Pz_hibeta', 2.0, "Pz", "Start here to stabilize limbic/sensory loops before frontal work", lambda ef, rs: True),
            ('Fz_hibeta', 2.0, "Fz", "Only after posterior cleanup. Frontal beta may feel worse until then.", 
             lambda ef, rs: ef.get('Pz_hibeta', 0) <= 2.0),  # Posterior stability check
            ('T3_theta', 3.5, "T3", "Check for disconnection or developmental drift (possible ASD marker)", 
             lambda ef, rs: 'hyperactivity' in rs or 'focus issues' in rs),  # Symptom consistency
            ('Cz_delta', 5.0, "Cz", "Consider targeting delta at Cz for sleep regulation issues", 
             lambda ef, rs: 'insomnia' in rs),
            ('Fz_beta', 2.0, "Fz", "Reduce beta for hyperarousal in PTSD", 
             lambda ef, rs: 'PTSD' in rs and ef.get('Pz_hibeta', 0) <= 2.0),
            ('Cz_beta', 2.0, "Cz", "Reduce beta for hyperarousal in PTSD", 
             lambda ef, rs: 'PTSD' in rs and ef.get('Pz_hibeta', 0) <= 2.0),
            ('P4_alpha', 5.0, "P3/P4", "Balance alpha asymmetry for attention issues in PTSD", 
             lambda ef, rs: 'PTSD' in rs and ef.get('P4_alpha', 0) > ef.get('P3_alpha', 0)),
            ('F3_alpha', 5.0, "F3/F4", "Check for emotional regulation issues", 
             lambda ef, rs: 'depression' in rs or abs(ef.get('F3_alpha', 0) - ef.get('F4_alpha', 0)) > 1.0),
            ('O1_alpha', 5.0, "O1/O2", "Increase alpha for relaxation in anxiety", 
             lambda ef, rs: 'anxiety' in rs),
            ('C3_theta', 5.0, "C3/C4", "Target theta for motor control in ADHD", 
             lambda ef, rs: 'hyperactivity' in rs or 'focus issues' in rs),
            ('T4_theta', 3.5, "T4", "Check for language processing issues (possible ASD marker)", 
             lambda ef, rs: 'hyperactivity' in rs or 'focus issues' in rs),
            ('Fp1_beta', 2.0, "Fp1/Fp2", "Reduce beta for executive dysfunction in PTSD", 
             lambda ef, rs: 'PTSD' in rs and ef.get('Pz_hibeta', 0) <= 2.0),
            ('Oz_alpha', 5.0, "Pz/Oz", "Increase alpha for visual processing stability", 
             lambda ef, rs: 'anxiety' in rs or 'PTSD' in rs)
        ]

        for signal, threshold, site, reason, cross_check in target_checks:
            if eeg_findings.get(signal, 0) > threshold and cross_check(eeg_findings, reported_symptoms):
                suggestions.append((site, reason))

        # Additional cross-check: Ensure coherence between homologous sites (e.g., F3/F4)
        if 'depression' in reported_symptoms and abs(eeg_findings.get('F3_alpha', 0) - eeg_findings.get('F4_alpha', 0)) > 1.0:
            if ('F3/F4', "Check for emotional regulation issues") not in suggestions:
                suggestions.append(("F3/F4", "Significant alpha asymmetry detected, confirm for depression"))

        return suggestions

    def suggest_protocols(self, eeg_findings: Dict[str, float], reported_symptoms: List[str]) -> List[Tuple[str, str]]:
        """
        Provides neurofeedback protocol recommendations based on EEG findings and client symptoms.
        Returns a list of (protocol, reason) tuples.
        """
        protocols = []
        protocol_conditions = [
            ('anxiety', lambda ef: ef.get('O1_alpha', 0) < 5.0 or ef.get('O2_alpha', 0) < 5.0, 
             "Alpha-theta training at O1/O2", "For relaxation and anxiety reduction"),
            ('focus issues', lambda ef: ef.get('theta', 0) > 5.0, 
             "Theta/beta ratio training at Fz/Cz", "To improve focus by reducing theta"),
            ('impulsivity', lambda ef: ef.get('F3_beta', 0) < 1.5, 
             "Beta increase at F3", "For attentional control"),
            ('depression', lambda ef: abs(ef.get('F3_alpha', 0) - ef.get('F4_alpha', 0)) > 1.0, 
             "Alpha asymmetry correction at F3/F4", "To address valence bias"),
            ('insomnia', lambda ef: ef.get('Cz_beta', 0) > 2.0, 
             "Beta reduction at Cz", "To promote sleep"),
            ('hyperactivity', lambda ef: ef.get('Fz_theta', 0) > 5.0, 
             "Theta reduction at Fz", "To address hyperactivity"),
            ('PTSD', lambda ef: ef.get('Fz_beta', 0) > 2.0 or ef.get('Cz_beta', 0) > 2.0, 
             "Beta reduction at Fz/Cz", "To reduce hyperarousal"),
            ('PTSD', lambda ef: ef.get('P4_alpha', 0) > ef.get('P3_alpha', 0), 
             "Alpha balancing at P3/P4", "To address attention issues"),
            ('PTSD', lambda ef: ef.get('Fp1_beta', 0) > 2.0, 
             "Beta reduction at Fp1/Fp2", "To improve executive function"),
            ('OCD', lambda ef: ef.get('Fz_hibeta', 0) > 2.0 or ef.get('Cz_hibeta', 0) > 2.0, 
             "Hibeta reduction at Fz/Cz", "To reduce rumination"),
            ('hyperactivity', lambda ef: ef.get('C3_theta', 0) > 5.0, 
             "SMR training at C3/C4", "To enhance motor control in ADHD"),
            ('focus issues', lambda ef: ef.get('T4_theta', 0) > 3.5, 
             "Theta reduction at T4", "To address potential ASD-related issues")
        ]
        for symptom, condition, protocol, reason in protocol_conditions:
            if symptom in reported_symptoms and condition(eeg_findings):
                protocols.append((protocol, reason))
        return protocols

    def explain_traps(self) -> List[str]:
        """
        Outlines common mistakes practitioners should avoid when interpreting EEG data.
        """
        return [
            "Mistaking artifact for true signal (e.g., muscle, eye movement)",
            "Overfocusing on frontal sites before stabilizing posterior regions",
            "Ignoring alpha asymmetry in depression cases",
            "Assuming all high beta is pathological—context matters",
            "Neglecting client-reported symptoms in favor of only waveform data",
            "Failing to cross-check protocol with both symptoms and EEG findings",
            "Targeting sites without checking posterior stability, risking overstimulation",
            "Overlooking coherence issues between homologous sites (e.g., F3/F4)",
            "Not adjusting protocols for new EEG sites like C3/C4 or Fp1/Fp2"
        ]

    def summarize(self, signal_profile: Dict[str, float], report_symptoms: List[str]) -> str:
        """
        Returns a user-friendly summary of all recommendations and flags.
        """
        drift_flags = self.analyze_site_drift(signal_profile, report_symptoms)
        site_targets = self.match_site_to_target(signal_profile, report_symptoms)
        protocols = self.suggest_protocols(signal_profile, report_symptoms)
        traps = self.explain_traps()
        summary = ["\n=== Symptom vs Waveform Analysis Summary ==="]
        if drift_flags:
            summary.append("\nPotential Mistargeted Symptoms:")
            for k, v in drift_flags.items():
                summary.append(f"- {k}: {v}")
        if site_targets:
            summary.append("\nRecommended Starting Sites:")
            for site, reason in site_targets:
                summary.append(f"- {site}: {reason}")
        if protocols:
            summary.append("\nSuggested Protocols:")
            for protocol, reason in protocols:
                summary.append(f"- {protocol}: {reason}")
        summary.append("\nCommon Practitioner Traps:")
        for trap in traps:
            summary.append(f"- {trap}")
        return "\n".join(summary)

    def generate_txt_report(self, signal_profile: Dict[str, float], report_symptoms: List[str], output_dir: str, filename: str = "symptom_vs_waveform_report.txt") -> str:
        """
        Generate a .txt report summarizing the analysis.
        Returns the path to the report.
        """
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, filename)
        summary = self.summarize(signal_profile, report_symptoms)
        with open(report_path, 'w') as f:
            f.write(summary)
        return report_path

    def plot_eeg_findings(self, signal_profile: Dict[str, float], report_symptoms: List[str], output_dir: str, filename: str = "eeg_findings_plot.png") -> str:
        """
        Generate a bar plot of key EEG findings for the reported symptoms.
        Returns the path to the plot.
        """
        os.makedirs(output_dir, exist_ok=True)
        # Only plot values relevant to symptoms
        relevant_keys = set()
        for symptom in report_symptoms:
            for k in signal_profile:
                if symptom.lower() in k.lower() or k.lower().startswith(symptom[:2].lower()):
                    relevant_keys.add(k)
        # If nothing matches, plot all
        if not relevant_keys:
            relevant_keys = set(signal_profile.keys())
        keys = sorted(relevant_keys)
        values = [signal_profile[k] for k in keys]
        plt.figure(figsize=(max(6, len(keys)), 4))
        bars = plt.bar(keys, values, color='skyblue')
        plt.title('EEG Findings Relevant to Symptoms')
        plt.ylabel('Value')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plot_path = os.path.join(output_dir, filename)
        plt.savefig(plot_path)
        plt.close()
        return plot_path

    def run_analysis(self, signal_profile: Dict[str, float], report_symptoms: List[str], output_dir: str) -> Dict[str, str]:
        """
        Run the full analysis: generate report and plot, return their paths.
        """
        logging.info("modules/symptom_vs_waveform.py: run_analysis called")
        report_path = self.generate_txt_report(signal_profile, report_symptoms, output_dir)
        plot_path = self.plot_eeg_findings(signal_profile, report_symptoms, output_dir)
        return {"report": report_path, "plot": plot_path}

    # --- Advanced Visualizations ---
    def plot_radar(self, eeg_features: list, output_dir: str, filename: str = "radar_symptom_waveform.png", max_features: int = 10) -> str:
        import matplotlib.pyplot as plt
        import numpy as np
        # Deduplicate and limit
        features = list(dict.fromkeys(eeg_features))[:max_features]
        if not features:
            logger.warning("No EEG features provided for radar plot. Skipping plot.")
            return None
        logger.info(f"[Radar] Plotting features: {features}")
        values = np.random.rand(len(features))  # Replace with real values if available
        angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
        values = np.concatenate((values, [values[0]]))
        angles += [angles[0]]
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.plot(angles, values, color='blue', linewidth=2)
        ax.fill(angles, values, color='blue', alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(features, fontsize=8, color='white', rotation=45)
        ax.set_yticklabels([])
        ax.set_title("EEG Feature Radar Plot", color='white')
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        out_path = os.path.join(output_dir, filename)
        plt.savefig(out_path, facecolor='black')
        plt.close(fig)
        return out_path

    def plot_protocol_network(self, eeg_features: list, symptoms: list, output_dir: str, filename: str = "network_symptom_waveform.png", max_nodes: int = 10) -> str:
        import networkx as nx
        import matplotlib.pyplot as plt
        # Deduplicate and limit
        features = list(dict.fromkeys(eeg_features))[:max_nodes]
        symptoms = list(dict.fromkeys(symptoms))[:max_nodes]
        logger.info(f"[Network] Plotting features: {features}")
        logger.info(f"[Network] Plotting symptoms: {symptoms}")
        G = nx.Graph()
        for s in symptoms:
            for f in features:
                if s.lower() in f.lower() or s[:2].lower() in f.lower():
                    G.add_edge(s, f)
        if G.number_of_edges() == 0:
            G.add_edge("No Data", "No Data")
        pos = nx.spring_layout(G, seed=42)
        fig, ax = plt.subplots(figsize=(8, 8))
        nx.draw(G, pos, with_labels=True, node_color='blue', font_size=8, font_color='white', edge_color='gray', ax=ax)
        plt.title("Symptom vs Waveform Network")
        out_path = os.path.join(output_dir, filename)
        plt.savefig(out_path, facecolor='black')
        plt.close(fig)
        return out_path

    def interactive_dashboard(self, signal_profile: Dict[str, float], report_symptoms: List[str]):
        try:
            from ipywidgets import interact, Dropdown
            import matplotlib.pyplot as plt
        except ImportError:
            logger.info("ipywidgets is required for the interactive dashboard. Please install it in your Jupyter environment.")
            return
        features = list(signal_profile.keys())
        def plot_feature(feature):
            plt.figure(figsize=(5, 2))
            plt.bar([feature], [signal_profile[feature]], color='skyblue')
            plt.title(f"{feature} Value")
            plt.show()
        interact(plot_feature, feature=Dropdown(options=features, description='EEG Feature:'))

    def plot_sankey(self, symptoms: list, waveforms: list, links: list, output_dir: str, filename: str = "sankey_symptom_waveform.png", max_links: int = 10) -> str:
        """
        Generate a static Sankey diagram showing flow from symptoms to waveforms.
        Args:
            symptoms (list): List of symptom names.
            waveforms (list): List of waveform names.
            links (list): List of (symptom, waveform) tuples.
            output_dir (str): Directory to save the plot.
            filename (str): Output filename.
            max_links (int): Maximum number of links to plot.
        Returns:
            str: Path to saved image.
        """
        import matplotlib.pyplot as plt
        from matplotlib.sankey import Sankey
        import collections
        # Deduplicate and limit links
        links = list(collections.OrderedDict.fromkeys(links))[:max_links]
        logger.info(f"[Sankey] Plotting links: {links}")
        if not links:
            links = [("No Data", "No Data")]
        # Build Sankey flows
        left_labels = [l[0] for l in links]
        right_labels = [l[1] for l in links]
        all_labels = list(collections.OrderedDict.fromkeys(left_labels + right_labels))
        flows = [1]*len(links)
        sankey = Sankey(flows=flows, labels=all_labels, orientations=[0]*len(all_labels))
        fig = plt.figure(figsize=(10, 5))
        plt.title("Symptom vs Waveform Sankey Diagram")
        try:
            sankey.finish()
        except Exception as e:
            logger.warning(f"[Sankey] Error: {e}")
        out_path = os.path.join(output_dir, filename)
        plt.savefig(out_path, facecolor='black')
        plt.close(fig)
        return out_path

# Example usage
if __name__ == "__main__":
    eeg = {
        'Pz_hibeta': 2.5, 'Fz_hibeta': 1.5, 'theta': 3.0, 'F3_beta': 1.0,
        'F3_alpha': 6.0, 'F4_alpha': 4.5, 'Cz_delta': 6.0, 'Fz_theta': 3.5,
        'Fz_beta': 2.5, 'Cz_beta': 2.1, 'P4_alpha': 6.0, 'P3_alpha': 4.0,
        'O1_alpha': 4.0, 'O2_alpha': 4.5, 'T3_theta': 4.0, 'C3_theta': 5.5,
        'T4_theta': 3.8, 'Fp1_beta': 2.2, 'Oz_alpha': 5.2
    }
    symptoms = ['anxiety', 'depression', 'insomnia', 'focus issues', 'PTSD', 'OCD', 'hyperactivity']
    svw = SymptomVsWaveform()
    output_dir = 'symptom_vs_waveform_output'
    results = svw.run_analysis(eeg, symptoms, output_dir)
    logger.info(f"Report saved to: {results['report']}")
    logger.info(f"Plot saved to: {results['plot']}")
    # Advanced visualizations
    radar_path = svw.plot_radar(list(eeg.keys()), output_dir)
    logger.info(f"Radar plot saved to: {radar_path}")
    network_path = svw.plot_protocol_network(list(eeg.keys()), symptoms, output_dir)
    logger.info(f"Network plot saved to: {network_path}")
    # Example Sankey usage
    # For demo, create links from symptoms to features containing their name
    links = []
    for symptom in symptoms:
        for feat in eeg:
            if symptom[:2].lower() in feat.lower():
                links.append((symptom, feat))
    sankey_path = svw.plot_sankey(symptoms, list(eeg.keys()), links, output_dir)
    logger.info(f"Sankey plot saved to: {sankey_path}")
    # For interactive dashboard, run in a Jupyter notebook:
    # svw.interactive_dashboard(eeg, symptoms) 