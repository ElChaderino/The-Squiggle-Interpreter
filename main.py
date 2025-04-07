#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main.py - The Squiggle Interpreter: Comprehensive EEG Analysis & Report Generation

This script performs the full EEG processing pipeline:
  • Discovers EDF files (using pathlib so filenames with spaces are supported) and groups them by subject.
  • For each subject, loads EEG data (removing "-LE", applying the standard 10–20 montage, average referencing,
    with optional current source density transform).
  • Computes band powers, topomaps, ERP, coherence, TFR, ICA, source localization, etc.
  • Generates detailed clinical reports (text, CSV, and interactive HTML) including refined clinical and connectivity mappings.
  • Optionally runs extension scripts and displays a live EEG simulation.
  • Optionally exports EDF data metrics to CSV.

Dependencies: mne, numpy, pandas, matplotlib, argparse, pathlib, rich, etc.
Ensure that modules/clinical_report.py, modules/pyramid_model.py, and modules/data_to_csv.py are present.
"""

import os
import sys
import threading
import time
import subprocess
import numpy as np
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import mne
import argparse
import signal
from pathlib import Path
from modules import clinical_report, pyramid_model, data_to_csv, phenotype
from modules.phenotype import classify_eeg_profile
from modules.vigilance import plot_vigilance_hypnogram
from modules import io_utils, processing, plotting, report, clinical, vigilance
from mne.io.constants import FIFF
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from scipy.stats import zscore, pearsonr
import pandas as pd

# Global stop event for live display
stop_event = threading.Event()

# Try to import psd_welch; if unavailable, fall back to psd_array_welch.
try:
    from mne.time_frequency import psd_welch
except ImportError:
    from mne.time_frequency import psd_array_welch as psd_welch


# ---------------- Robust Z-Score Functions ----------------
def robust_mad(x, constant=1.4826, max_iter=10, tol=1e-3):
    x = np.asarray(x)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    for _ in range(max_iter):
        mask = np.abs(x - med) <= 3 * mad
        new_med = np.median(x[mask])
        new_mad = np.median(np.abs(x[mask] - new_med))
        if np.abs(new_med - med) < tol and np.abs(new_mad - mad) < tol:
            break
        med, mad = new_med, new_mad
    return mad * constant, med


def robust_zscore(x, use_iqr=False):
    x = np.asarray(x)
    med = np.median(x)
    if use_iqr:
        q75, q25 = np.percentile(x, [75, 25])
        iqr = q75 - q25
        scale = iqr if iqr != 0 else 1.0
    else:
        scale, med = robust_mad(x)
        if scale == 0:
            scale = 1.0
    return (x - med) / scale


def compute_bandpower_robust_zscores(raw, bands=None, fmin=1, fmax=40, n_fft=2048, use_iqr=False):
    if bands is None:
        bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'SMR': (13, 15),
            'beta': (16, 28),
            'gamma': (29, 30)
        }
    psds, freqs = psd_welch(raw.get_data(), raw.info['sfreq'], fmin=fmin, fmax=fmax, n_fft=n_fft, verbose=False)
    psds_db = 10 * np.log10(psds)
    robust_features = {}
    for band, (low, high) in bands.items():
        band_mask = (freqs >= low) & (freqs <= high)
        band_power = psds_db[:, band_mask].mean(axis=1)
        robust_features[band] = robust_zscore(band_power, use_iqr=use_iqr)
    return robust_features


def load_clinical_outcomes(csv_file, n_channels):
    try:
        df = pd.read_csv(csv_file)
        outcomes = df['outcome'].values
        if len(outcomes) < n_channels:
            outcomes = np.pad(outcomes, (0, n_channels - len(outcomes)), mode='constant')
        return outcomes[:n_channels]
    except Exception as e:
        print("Could not load clinical outcomes from CSV:", e)
        return np.random.rand(n_channels)


def compare_zscores(standard_z, robust_z, clinical_outcomes):
    for band in standard_z.keys():
        r_std, p_std = pearsonr(standard_z[band], clinical_outcomes)
        r_rob, p_rob = pearsonr(robust_z[band], clinical_outcomes)
        print(f"Band {band}:")
        print(f"  Standard z-score: r = {r_std:.3f}, p = {p_std:.3f}")
        print(f"  Robust z-score  : r = {r_rob:.3f}, p = {p_rob:.3f}")


# --- Utility: Group EDF Files by Subject ---
def find_subject_edf_files(directory):
    edf_files = [f for f in os.listdir(directory) if f.lower().endswith('.edf')]
    subjects = {}
    for f in edf_files:
        f_lower = f.lower()
        subject_id = f_lower[:2]
        if subject_id not in subjects:
            subjects[subject_id] = {"EO": None, "EC": None}
        if "eo" in f_lower:
            subjects[subject_id]["EO"] = f
        elif "ec" in f_lower:
            subjects[subject_id]["EC"] = f
    return subjects


def live_eeg_display(stop_event, update_interval=1.2):
    def generate_eeg_wave(num_points=80):
        x = np.linspace(0, 4 * np.pi, num_points)
        wave = np.sin(x) + np.random.normal(0, 0.3, size=num_points)
        gradient = " .:-=+*#%@"
        norm_wave = (wave - wave.min()) / (wave.max() - wave.min() + 1e-10)
        indices = (norm_wave * (len(gradient) - 1)).astype(int)
        return "".join(gradient[i] for i in indices)

    def get_random_quote():
        quotes = [
            "The Dude abides.",
            "That rug really tied the room together.",
            "Yeah, well, you know, that's just, like, your opinion, man.",
            "Sometimes you eat the bear, and sometimes, well, the bear eats you.",
            "Watch the squiggles, man. They're pure EEG poetry.",
            "This aggression will not stand... not even in beta spindles.",
            "Don’t cross the streams, Walter. I’m seeing delta in my alpha, man.",
            "Calmer than you are? My frontal lobes are lighting up like a bowling alley.",
            "Smokey, this is not 'Nam. This is neurofeedback. There are protocols.",
            "Obviously you’re not a golfer, or you’d know theta doesn’t spike like that.",
            "The alpha giveth, and the theta taketh away.",
            "Don’t trust a flatline. Even silence has a frequency.",
            "The coherence is strong in this one.",
            "You can’t spell ‘LORETA’ without ‘lore’. Mythos in the cortex.",
            "That’s not artifact. That’s consciousness trying to escape the matrix.",
            "Beta’s climbing again. Someone’s inner monologue needs a coffee break.",
            "I read your EEG. You’re either meditating... or communing with the void.",
            "Phase-lock like your brain depends on it. Because it does.",
            "This topomap? It’s basically a heat map of your soul.",
            "Theta whispers the secrets. Delta keeps the dreams.",
            "Bro, your prefrontal cortex is on shuffle.",
            "High beta? More like internal monologue with a megaphone.",
            "We're all just waveforms trying to sync up in a noisy universe.",
            "I didn't choose the squiggle life. The squiggle life entrained me.",
            "Did your PAF just ghost you mid-session? Brutal.",
            "You brought 60 Hz into my sacred coherence chamber?",
            "Every band tells a story. This one screams 'undiagnosed ADHD with a splash of genius.'",
            "Careful with the cross-frequency coupling… that's where the dragons sleep.",
            "In a land before time, someone spiked the theta—and the oracle woke up.",
            "Real-time feedback? Nah, this is EEG jazz, man. Improv with voltage.",
            "And now for something completely cortical.",
            "Your alpha waves have been accused of witchcraft!",
            "’Tis but a minor artifact! I’ve had worse!",
            "Your theta is high, your beta is low… you must be a shrubbery.",
            "This isn’t a brain, it’s a very naughty vegetable!",
            "I fart in the general direction of your coherence matrix.",
            "Help! Help! I'm being over-synchronized!",
            "We are the EEG technicians who say... *Ni!*",
            "On second thought, let’s not record at Fz. It is a silly place.",
            "I once saw a brain entrain so hard, it turned me into a newt. I got better.",
            "Your brain has exceeded its bandwidth quota. Please upgrade.",
            "Synapse latency detected. Reboot your consciousness.",
            "Alpha rhythm flagged: unauthorized serenity.",
            "Cognitive load: 98%. Executing override protocol.",
            "Error 404: Identity not found.",
            "EEG pattern suggests resistance. Recommend sedation.",
            "This is not a biofeedback session. This is surveillance with consent.",
            "Signal integrity compromised. Mind bleed imminent.",
            "You are being watched by 64 channels of your own making.",
            "Your dreams are now property of NeuroCorp™.",
            "The cortex folded like origami under a sonic burst of insight.",
            "She rode her SMR wave like a hacker surfing the noosphere.",
            "In the subdural silence, the squiggles spoke prophecy.",
            "Beta was spiking. That meant the grid was listening.",
            "The alpha breach began just after cognitive boot-up.",
            "Eyes closed. Theta opened the archive.",
            "The brain is not a machine. It's a codebase... evolving.",
            "He trained at Cz until the feedback whispered his name.",
            "Phase-lock acquired. Prepare to uplink to the collective.",
            "She reached Pz. It shimmered. The veil between thoughts lifted.",
            "Theta is the dreamer’s path — the shadow realm whispers.",
            "Each peak a memory. Each trough, a wound not yet integrated.",
            "In the dance of Alpha and Theta lies the gate to the Self.",
            "Delta carries the voice of the ancestors.",
            "You are not anxious. You are facing the dragon of your own unconscious.",
            "The squiggle is a mandala. And you are the artist.",
            "Synchrony is the return to the sacred masculine and feminine balance.",
            "Frontal asymmetry reveals the archetype you suppress.",
            "High beta is the ego screaming to remain relevant.",
            "To see coherence is to glimpse the collective unconscious rendered in voltage.",
            "The signal is raw ore. Your attention — the hammer.",
            "This isn’t data. It’s a blade waiting for the quench.",
            "Every artifact is a misstrike. Adjust your grip.",
            "You don’t read EEG. You listen to the forge’s hiss.",
            "The best welds leave no seam. Just like coherence.",
            "Real neurofeedback is shaped on the anvil of presence.",
            "High beta? That’s a spark flying before the temper holds.",
            "Each protocol is a blacksmith’s chant. Repetition. Focus. Fire.",
            "Theta hums like the bellows before alpha glows true.",
            "Some build castles in the clouds. I build minds in the flame.",
            "[ALPHA] ~ engaged @ 10.0Hz // you're surfing the calmnet.",
            "*sysop has entered the mind* >> Theta/beta > 3.2 — user flagged for wandering.",
            "<neuroN0de> dude your SMR band just buffer-overflowed reality lol.",
            "/msg Pz: stop ghosting and fire some clean alpha, jeez.",
            "[404] Vigilance not found. Try rebooting your occipital lobe.",
            "Your coherence matrix just pinged the void. Respect.",
            "BRAIN.SYS: Unexpected Delta spike at wake_state=1",
            "Welcome to the z-sc0r3z BBS — leave your ego at the login prompt.",
            "[EEG-OPS]: alpha locked. theta contained. signal pure.",
            "*vibrotactile entrainment initiated* — press <F2> to feel old gods resonate.",
            "Alpha hums like neon rain — cortex in low-noise high-focus mode.",
            "Neural net drift detected. Theta bleeding into Beta. Patch cognition.exe.",
            "Mind uplink stable. Vigilance layer: A1. Spin the waveforms, cowboy.",
            "The cortex doesn’t forget — it routes trauma like dead packet nodes.",
            "Memory lane is fragged. Delta spike at 3Hz — reboot dream protocol.",
            "Synapse traffic jacked into feedback loop. Traceroute: Pz → Fz → Void.",
            "Mental firewall down. Beta intrusion spiking at 28Hz. Secure the band.",
            "She walked in with alpha like moonlight on wet asphalt.",
            "Bio-signal integrity compromised. sLORETA grid glitching at parietal rim.",
            "Brainwave sync: 𝑔𝑟𝑒𝑒𝑛. Thoughts encrypted. Consciousness... proxied.",
            "Release: [neuroGENx v1.337] :: Cracked by [SMR] Crew :: Respect to #eeg-scene",
            "[ZSC0RE DUMP] ∙ Channel Pz ∙ Vigilance: A2 ∙ State: 🟡 Semi-Coherent",
            "Signal patched, checksum clean. Alpha uptrained. Mind ready for upload.",
            "GREETZ to the inner cortex! <3 Cz, Pz, O1 — keep vibin’",
            ":: THETA INJECTED @ 5.6Hz :: USER MAY EXPERIENCE TEMPORAL DISSOLUTION ::",
            "nfo: EEG-cracked · Loader: Cz ∙ Protocol: HI-BETA DOWN ∙ Scene Approved™",
            "+[ Mind scan @ Cz complete ]+ ➜ No malware. Just trauma.",
            "[SYS REPORT] :: Executive functions: overclocked · Memory: defragging",
            "Greetings from the Limbic Underground :: Your amygdala owes us rent.",
            "This session was proudly cracked by ████ – z-scores normalized, reality bent.",
            "Let the data speak — but be ready when it starts shouting in high-beta.",
            "You're not treating ADHD — you're treating 15 microvolts of distributed chaos.",
            "Alpha is a state. But stable posterior alpha? That’s a trait. Respect the trait.",
            "Cz’s not anxious, it’s just watching you screw up the montage.",
            "If you see beta spindles at Fz and think 'focus', call Jay — he’ll recalibrate your soul.",
            "PAF tells you who they *are*, not just how they slept.",
            "T4 whispers trauma. Pz remembers the dreams.",
            "Delta at FP1? That’s not a sleep wave — that’s a buried memory with a security clearance.",
            "Every drop in alpha is a story the brain didn’t finish telling.",
            "You train alpha like you’d tame a wolf — with trust, timing, and the right dose of poetry.",
            "Coherence isn't peace — it's synchronized paranoia if you're not careful.",
            "Normalize the z-score, sure — but check if the brain *likes* it there.",
            "Theta isn’t slow — it’s just busy digging up the past.",
            "If SMR rises and the kid still kicks the chair, train the chair.",
            "Phase is where the secrets are. The rest is noise with credentials.",
            "Your metrics don't mean squat until behavior changes — or at least the dog stops barking.",
            "A brain out of phase tells you it’s still negotiating its lease on reality.",
            "Artifact rejection is the brain’s way of testing your ethics.",
            "Every topomap is a Rorschach — the trick is knowing which ink is dry.",
            "High theta doesn’t always mean ADHD. Sometimes it just means the world is too loud.",
            "If you don’t know your client’s PAF, you’re driving with a map but no compass.",
            "Training attention without tracking arousal is like aiming without noticing you’re underwater.",
            "If the brain doesn’t change in 20 sessions, maybe it doesn’t want to. Or maybe it doesn’t trust you yet.",
            "Every artifact you ignore is a story you chose not to hear.",
            "Normalize the nervous system — not just the numbers.",
            "You’re not dysregulated — you’re just running too many tabs in your frontal lobe.",
            "SMR isn’t magic. It’s just the brain remembering not to twitch when the world knocks.",
            "Delta during wakefulness? That’s not spiritual — that’s your cortex sending a 404.",
            "Your protocol isn't custom unless you’ve asked the client how they sleep. And mean it.",
            "Every EEG session is a negotiation. You're not the boss — you're just the translator.",
            "Protocols are just hypotheses. Brains are the real lab.",
            "Remote NFB is like long-distance relationships — it works if you’re honest and the signal holds.",
            "The EEG doesn't lie — but it *does* get confused by poor sleep, coffee, and wishful thinking.",
            "Don’t teach the brain what you think it needs. Ask it what it’s trying to say.",
            "You’re not optimizing — you’re helping it remember what regulation feels like.",
            "Peak Alpha isn't where you feel enlightened — it's where your brain finally sighs in relief.",
            "Theta doesn’t mean mystical. It just means the brakes aren’t working.",
            "Look at Pz. If it’s quiet, the story hasn’t started yet.",
            "Don’t treat diagnoses. Treat dysregulation.",
            "Every brain is a poem. Try not to edit it too fast."
        ]
        return np.random.choice(quotes)

    console = Console()
    with Live(refresh_per_second=10, console=console) as live:
        while not stop_event.is_set():
            line = generate_eeg_wave(os.get_terminal_size().columns - 4)
            quote = get_random_quote()
            text = f"[bold green]{line}[/bold green]"
            if quote:
                text += f"\n[bold red]{quote}[/bold red]\n"
            text += f"[bold blue]{line[::-1]}[/bold blue]"
            panel = Panel(text, title="Live EEG Display", subtitle="Simulated Waveform", style="white on black")
            live.update(panel)
            time.sleep(update_interval)


def sigint_handler(signum, frame):
    print("SIGINT received, stopping gracefully...")
    stop_event.set()
    sys.exit(0)


# --- Refactored Pipeline Functions ---

def parse_arguments():
    parser = argparse.ArgumentParser(
        prog='The Squiggle Interpreter',
        description='Comprehensive EEG Analysis & Clinical Report Generation'
    )
    parser.add_argument('--csd', help="Use current source density (CSD) for graphs? (y/n)")
    parser.add_argument('--zscore',
                        help="Z-score normalization method: 1: Standard, 2: Robust (MAD), 3: Robust (IQR), 4: Published Norms")
    parser.add_argument('--report', help="Generate full clinical report? (y/n)")
    parser.add_argument('--phenotype', help="Generate phenotype classification? (y/n)")
    parser.add_argument('--csv', action='store_true', help="Export EDF data metrics to CSV")
    parser.add_argument('--edf', help="Path to an EDF file for CSV export")
    parser.add_argument('--epoch_length', type=float, default=2.0,
                        help="Epoch length (in seconds) for CSV export (default: 2.0)")
    parser.add_argument('--output_csv', help="Output CSV file path for CSV export")

    args = parser.parse_args()

    config = {}
    config['csv'] = args.csv

    if config['csv']:
        if not args.edf or not args.output_csv:
            print("For CSV export, please provide both --edf and --output_csv arguments.")
            sys.exit(1)
        config['edf_path'] = args.edf
        config['epoch_length'] = args.epoch_length
        config['output_csv'] = args.output_csv
        return config

    config['csd'] = (args.csd or input("Use current source density (CSD) for graphs? (y/n, default: n): ") or "n").lower() == "y"
    print(f"Using CSD for graphs: {config['csd']}")

    if args.zscore is None:
        print("Choose z-score normalization method:")
        print("  1: Standard (mean/std)")
        print("  2: Robust (MAD-based)")
        print("  3: Robust (IQR-based)")
        print("  4: Published Norms (adult norms)")
        config['zscore'] = input("Enter choice (default: 1): ") or "1"
    else:
        config['zscore'] = args.zscore

    config['report'] = (args.report or input("Generate full clinical report? (y/n, default: y): ") or "y").lower() == "y"
    config['phenotype'] = (args.phenotype or input("Generate phenotype classification? (y/n, default: y): ") or "y").lower() == "y"

    return config



def setup_output_directories(project_dir, subject):
    overall_output_dir = os.path.join(project_dir, "outputs")
    os.makedirs(overall_output_dir, exist_ok=True)

    subject_folder = os.path.join(overall_output_dir, subject)
    os.makedirs(subject_folder, exist_ok=True)

    folders = {
        "topomaps_eo": os.path.join(subject_folder, "topomaps", "EO"),
        "topomaps_ec": os.path.join(subject_folder, "topomaps", "EC"),
        "waveforms_eo": os.path.join(subject_folder, "waveforms", "EO"),
        "erp": os.path.join(subject_folder, "erp"),
        "coherence_eo": os.path.join(subject_folder, "coherence", "EO"),
        "coherence_ec": os.path.join(subject_folder, "coherence", "EC"),
        "zscore_eo": os.path.join(subject_folder, "zscore", "EO"),
        "zscore_ec": os.path.join(subject_folder, "zscore", "EC"),
        "tfr_eo": os.path.join(subject_folder, "tfr", "EO"),
        "tfr_ec": os.path.join(subject_folder, "tfr", "EC"),
        "ica_eo": os.path.join(subject_folder, "ica", "EO"),
        "source": os.path.join(subject_folder, "source_localization"),
        "detailed": os.path.join(subject_folder, "detailed_site_plots"),
        "vigilance": os.path.join(subject_folder, "vigilance")
    }
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)

    return overall_output_dir, folders, subject_folder


def load_zscore_stats(method_choice):
    if method_choice == "4":
        published_norm_stats = {
            "Alpha": {"median": 20.0, "mad": 4.0},
            "Theta": {"median": 16.0, "mad": 3.5},
            "Delta": {"median": 22.0, "mad": 5.0},
            "SMR": {"median": 7.0, "mad": 1.5},
            "Beta": {"median": 6.0, "mad": 1.8},
            "HighBeta": {"median": 4.0, "mad": 1.2}
        }
        print("Using published normative values for adult EEG (published_norm_stats).")
        return published_norm_stats
    else:
        return {
            "Alpha": {"median": 18.0, "mad": 6.0},
            "Theta": {"median": 15.0, "mad": 5.0},
            "Delta": {"median": 20.0, "mad": 7.0},
            "SMR": {"median": 6.0, "mad": 2.0},
            "Beta": {"median": 5.0, "mad": 2.0},
            "HighBeta": {"median": 3.5, "mad": 1.5}
        }


def load_and_preprocess_data(project_dir, files, use_csd):
    eo_file = files["EO"] if files["EO"] else files["EC"]
    ec_file = files["EC"] if files["EC"] else files["EO"]
    print(f"EO file: {eo_file}, EC file: {ec_file}")

    # Handle case where no files are available
    if not eo_file and not ec_file:
        print("No EO or EC files available for processing.")
        return None, None, None, None

    # Load data with fallback to None if loading fails
    raw_eo = io_utils.load_eeg_data(os.path.join(project_dir, eo_file), use_csd=False) if eo_file else None
    raw_ec = io_utils.load_eeg_data(os.path.join(project_dir, ec_file), use_csd=False) if ec_file else None

    # If both are None, return early
    if raw_eo is None and raw_ec is None:
        print("Failed to load both EO and EC data.")
        return None, None, None, None

    # Apply CSD if requested, with fallback to original data on failure
    if use_csd:
        raw_eo_csd = raw_eo.copy().load_data() if raw_eo else None
        raw_ec_csd = raw_ec.copy().load_data() if raw_ec else None
        if raw_eo_csd:
            try:
                raw_eo_csd = mne.preprocessing.compute_current_source_density(raw_eo_csd)
                print("CSD applied for graphs (EO).")
            except Exception as e:
                print("CSD for graphs (EO) failed:", e)
                raw_eo_csd = raw_eo
        if raw_ec_csd:
            try:
                raw_ec_csd = mne.preprocessing.compute_current_source_density(raw_ec_csd)
                print("CSD applied for graphs (EC).")
            except Exception as e:
                print("CSD for graphs (EC) failed:", e)
                raw_ec_csd = raw_ec
    else:
        raw_eo_csd = raw_eo
        raw_ec_csd = raw_ec

    return raw_eo, raw_ec, raw_eo_csd, raw_ec_csd


def compute_zscore_features(raw_eo, method_choice, published_norm_stats):
    default_bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (12, 30),
        'gamma': (30, 40)
    }

    if raw_eo is None:
        print("Cannot compute z-score features: EO data is None.")
        return {}, {}

    psds, freqs = psd_welch(raw_eo.get_data(), raw_eo.info['sfreq'], fmin=1, fmax=40, n_fft=2048, verbose=False)
    psds_db = 10 * np.log10(psds)
    standard_features = {}
    for band, (low, high) in default_bands.items():
        band_mask = (freqs >= low) & (freqs <= high)
        band_power = psds_db[:, band_mask].mean(axis=1)
        standard_features[band] = zscore(band_power)

    if method_choice == "1":
        chosen_features = standard_features
        print("Using standard z-scores (mean/std) for bandpower features.")
    elif method_choice == "2":
        chosen_features = compute_bandpower_robust_zscores(raw_eo, bands=default_bands, use_iqr=False)
        print("Using robust z-scores (MAD-based) for bandpower features.")
    elif method_choice == "3":
        chosen_features = compute_bandpower_robust_zscores(raw_eo, bands=default_bands, use_iqr=True)
        print("Using robust z-scores (IQR-based) for bandpower features.")
    elif method_choice == "4":
        chosen_features = {}
        zscore_maps_eo = processing.compute_all_zscore_maps(raw_eo, published_norm_stats, epoch_len_sec=2.0)
        for band in default_bands.keys():
            chosen_features[band] = zscore_maps_eo.get(band, np.array([]))
        print("Using published norms for z-score normalization.")
    else:
        print("Invalid choice. Defaulting to standard z-scores.")
        chosen_features = standard_features

    return standard_features, chosen_features


def process_vigilance(raw_eo, folders):
    if raw_eo is None:
        print("Skipping vigilance processing: EO data is None.")
        return
    vigilance_states = vigilance.compute_vigilance_states(raw_eo, epoch_length=2.0)
    try:
        fig = plot_vigilance_hypnogram(vigilance_states, epoch_length=2.0)
        hypno_path = os.path.join(folders["vigilance"], "vigilance_hypnogram.png")
        fig.savefig(hypno_path, facecolor='black')
        plt.close(fig)
        print(f"Saved vigilance hypnogram to {hypno_path}")
        for t, stage in vigilance_states:
            print(f"{t:5.1f}s: {stage}")
    except AttributeError:
        print("Error: 'plot_vigilance_hypnogram' not found in modules.vigilance.")


def process_topomaps(raw, condition, folders, band_list):
    if raw is None:
        print(f"Skipping topomap processing for {condition}: No data available.")
        return {}
    bp = processing.compute_all_band_powers(raw)
    topomaps = {}
    for band in band_list:
        abs_power = [bp[ch][band] for ch in raw.ch_names]
        rel_power = []
        for ch in raw.ch_names:
            total_power = sum(bp[ch][b] for b in band_list)
            rel_power.append(bp[ch][band] / total_power if total_power else 0)
        fig_topo = plotting.plot_topomap_abs_rel(abs_power, rel_power, raw.info, band, condition)
        topo_path = os.path.join(folders[f"topomaps_{condition.lower()}"], f"topomap_{band}.png")
        fig_topo.savefig(topo_path, facecolor='black')
        plt.close(fig_topo)
        topomaps[band] = os.path.basename(topo_path)
        print(f"Saved {condition} topomap for {band} to {topo_path}")
    return topomaps


def process_waveforms(raw, condition, folders, band_list):
    if raw is None:
        print(f"Skipping waveform processing for {condition}: No data available.")
        return {}
    global_waveforms = {}
    data = raw.get_data() * 1e6
    sfreq = raw.info['sfreq']
    for band, band_range in processing.BANDS.items():
        wf_fig = plotting.plot_waveform_grid(data, raw.ch_names, sfreq, band=band_range, epoch_length=10)
        wf_path = os.path.join(folders[f"waveforms_{condition.lower()}"], f"waveforms_{band}.png")
        wf_fig.savefig(wf_path, facecolor='black')
        plt.close(wf_fig)
        global_waveforms[band] = os.path.basename(wf_path)
        print(f"Saved {condition} waveform grid for {band} to {wf_path}")
    return global_waveforms


def process_erp(raw, condition, folders):
    if raw is None:
        print(f"Skipping ERP processing for {condition}: No data available.")
        return ""
    erp_fig = processing.compute_pseudo_erp(raw)
    erp_path = os.path.join(folders["erp"], f"erp_{condition}.png")
    erp_fig.savefig(erp_path, facecolor='black')
    plt.close(erp_fig)
    print(f"Saved ERP {condition} to {erp_path}")
    return os.path.basename(erp_path)


def process_coherence(raw, condition, folders, band_list):
    if raw is None:
        print(f"Skipping coherence processing for {condition}: No data available.")
        return {}
    coherence_maps = {}
    sfreq = raw.info['sfreq']
    for band in band_list:
        band_range = processing.BANDS[band]
        coh_matrix = processing.compute_coherence_matrix(raw.get_data() * 1e6, sfreq, band_range,
                                                         nperseg=int(sfreq * 2))
        fig_coh = plotting.plot_coherence_matrix(coh_matrix, raw.ch_names)
        coh_path = os.path.join(folders[f"coherence_{condition.lower()}"], f"coherence_{band}.png")
        fig_coh.savefig(coh_path, facecolor='black')
        plt.close(fig_coh)
        coherence_maps[band] = os.path.basename(coh_path)
        print(f"Saved {condition} coherence ({band}) to {coh_path}")
    return coherence_maps


def process_zscores(raw, condition, folders, band_list, norm_stats):
    if raw is None:
        print(f"Skipping z-score processing for {condition}: No data available.")
        return {}
    zscore_maps = processing.compute_all_zscore_maps(raw, norm_stats, epoch_len_sec=2.0)
    zscore_images = {}
    for band in band_list:
        if band in zscore_maps and zscore_maps[band] is not None:
            fig_zscore = plotting.plot_zscore_topomap(zscore_maps[band], raw.info, band, condition)
            zscore_path = os.path.join(folders[f"zscore_{condition.lower()}"], f"zscore_{band}.png")
            fig_zscore.savefig(zscore_path, facecolor='black')
            plt.close(fig_zscore)
            zscore_images[band] = os.path.basename(zscore_path)
            print(f"Saved {condition} z-score ({band}) to {zscore_path}")
    return zscore_images


def process_tfr(raw, condition, folders, band_list):
    if raw is None:
        print(f"Skipping TFR processing for {condition}: No data available.")
        return {}
    n_cycles = 2.0
    tfr_maps = processing.compute_all_tfr_maps(raw, n_cycles, tmin=0.0, tmax=4.0)
    tfr_images = {}
    for band in band_list:
        if band in tfr_maps and tfr_maps[band] is not None:
            fig_tfr = plotting.plot_tfr(tfr_maps[band], picks=0)
            tfr_path = os.path.join(folders[f"tfr_{condition.lower()}"], f"tfr_{band}.png")
            fig_tfr.savefig(tfr_path, facecolor='black')
            plt.close(fig_tfr)
            tfr_images[band] = os.path.basename(tfr_path)
            print(f"Saved TFR {condition} ({band}) to {tfr_path}")
        else:
            print(f"TFR {condition} for {band} was not computed.")
    return tfr_images


def process_ica(raw, condition, folders):
    if raw is None:
        print(f"Skipping ICA processing for {condition}: No data available.")
        return ""
    ica = processing.compute_ica(raw)
    fig_ica = plotting.plot_ica_components(ica, raw)
    ica_path = os.path.join(folders[f"ica_{condition.lower()}"], f"ica_{condition}.png")
    fig_ica.savefig(ica_path, facecolor='black')
    plt.close(fig_ica)
    print(f"Saved ICA {condition} to {ica_path}")
    return os.path.basename(ica_path)


def process_source_localization(raw_eo, raw_ec, folders, band_list):
    if raw_eo is None and raw_ec is None:
        print("Skipping source localization: No EO or EC data available.")
        return {"EO": {}, "EC": {}}

    raw_source_eo = raw_eo.copy() if raw_eo else None
    raw_source_ec = raw_ec.copy() if raw_ec else None

    if raw_source_eo:
        raw_source_eo.set_eeg_reference("average", projection=False)
        print("EEG channels for source localization (EO):", mne.pick_types(raw_source_eo.info, meg=False, eeg=True))
    if raw_source_ec:
        raw_source_ec.set_eeg_reference("average", projection=False)

    fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
    subjects_dir = os.path.dirname(fs_dir)
    subject_fs = "fsaverage"
    src = mne.setup_source_space(subject_fs, spacing="oct6", subjects_dir=subjects_dir, add_dist=False)
    conductivity = (0.3, 0.006, 0.3)
    bem_model = mne.make_bem_model(subject=subject_fs, ico=4, conductivity=conductivity, subjects_dir=subjects_dir)
    bem_solution = mne.make_bem_solution(bem_model)

    fwd_eo = mne.make_forward_solution(raw_source_eo.info, trans="fsaverage", src=src,
                                       bem=bem_solution, eeg=True, meg=False, verbose=False) if raw_source_eo else None
    fwd_ec = mne.make_forward_solution(raw_source_ec.info, trans="fsaverage", src=src,
                                       bem=bem_solution, eeg=True, meg=False, verbose=False) if raw_source_ec else None

    if raw_eo:
        events_eo = mne.make_fixed_length_events(raw_eo, duration=2.0)
        epochs_eo = mne.Epochs(raw_eo, events_eo, tmin=-0.1, tmax=0.4, baseline=(None, 0),
                               preload=True, verbose=False)
        cov_eo = mne.compute_covariance(epochs_eo, tmax=0., method="empirical", verbose=False)
    else:
        cov_eo = None

    inv_op_eo = processing.compute_inverse_operator(raw_source_eo, fwd_eo, cov_eo) if raw_source_eo and cov_eo else None
    inv_op_ec = processing.compute_inverse_operator(raw_source_ec, fwd_ec, cov_eo) if raw_source_ec and cov_eo else None

    source_methods = {"LORETA": "MNE", "sLORETA": "sLORETA", "eLORETA": "eLORETA"}
    source_localization = {"EO": {}, "EC": {}}
    for cond, raw_data, inv_op in [("EO", raw_eo, inv_op_eo), ("EC", raw_ec, inv_op_ec)]:
        if raw_data is None or inv_op is None:
            print(f"Skipping source localization for {cond}: No data or inverse operator available.")
            continue
        for band in band_list:
            band_range = processing.BANDS[band]
            raw_band = raw_data.copy().filter(band_range[0], band_range[1], verbose=False)
            events = mne.make_fixed_length_events(raw_band, duration=2.0)
            epochs = mne.Epochs(raw_band, events, tmin=-0.1, tmax=0.4, baseline=(None, 0),
                                preload=True, verbose=False)
            evoked = epochs.average()
            cond_folder = os.path.join(folders["source"], cond)
            os.makedirs(cond_folder, exist_ok=True)
            for method, method_label in source_methods.items():
                try:
                    stc = processing.compute_source_localization(evoked, inv_op, lambda2=1.0 / 9.0, method=method_label)
                    fig_src = plotting.plot_source_estimate(stc, view="lateral", time_point=0.1,
                                                            subjects_dir=subjects_dir)
                    src_filename = f"source_{cond}_{method}_{band}.png"
                    src_path = os.path.join(cond_folder, src_filename)
                    fig_src.savefig(src_path, dpi=150, bbox_inches="tight", facecolor="black")
                    plt.close(fig_src)
                    source_localization[cond].setdefault(band, {})[method] = cond + "/" + src_filename
                    fig_src.savefig(src_path, dpi=150, bbox_inches="tight", facecolor="black")
                    # No need to print anything unless debugging

                except Exception as e:
                    print(f"Error computing source localization for {cond} {band} with {method}: {e}")
    return source_localization


def process_phenotype(raw_eo, subject_folder, subject):
    if raw_eo is None:
        print("Skipping phenotype processing: EO data is None.")
        return
    from modules.feature_extraction import extract_classification_features
    features = extract_classification_features(raw_eo, [])
    phenotype_results = classify_eeg_profile(features)

    phenotype_report_path = os.path.join(subject_folder, f"{subject}_phenotype.txt")
    with open(phenotype_report_path, "w", encoding="utf-8") as f:
        f.write("Phenotype Classification Results\n")
        f.write("===============================\n")
        for k, v in phenotype_results.items():
            f.write(f"{k}: {v}\n")

    clinical_txt_path = os.path.join(subject_folder, f"{subject}_clinical_report.txt")
    with open(clinical_txt_path, "a", encoding="utf-8") as f:
        f.write("\n\nPhenotype Classification Results\n")
        f.write("===============================\n")
        for k, v in phenotype_results.items():
            f.write(f"{k}: {v}\n")


def generate_reports(raw_eo, raw_ec, folders, subject_folder, subject, band_list, config, topomaps, waveforms, erp,
                     coherence, zscores, tfr, ica, source_localization):
    if raw_eo is None and raw_ec is None:
        print(f"Skipping report generation for subject {subject}: No EO or EC data available.")
        return

    bp_eo = processing.compute_all_band_powers(raw_eo) if raw_eo else {}
    bp_ec = processing.compute_all_band_powers(raw_ec) if raw_ec else {}
    print(f"Subject {subject} - Computed band powers for EO channels:", list(bp_eo.keys()) if bp_eo else "None")
    print(f"Subject {subject} - Computed band powers for EC channels:", list(bp_ec.keys()) if bp_ec else "None")

    if bp_eo or bp_ec:
        clinical.generate_site_reports(bp_eo, bp_ec, subject_folder)

    if config['report'] and (raw_eo or raw_ec):
        clinical_report.generate_full_clinical_report(config['csd'], True, subject_folder, subject)

    if raw_eo or raw_ec:
        clinical.generate_full_site_reports(raw_eo, raw_ec, folders["detailed"])

    global_diff_images = {}
    if raw_eo and raw_ec:
        for b in band_list:
            diff_vals = [bp_eo[ch][b] - bp_ec[ch][b] for ch in raw_eo.ch_names]
            diff_topo_fig = plotting.plot_difference_topomap(diff_vals, raw_eo.info, b)
            diff_bar_fig = plotting.plot_difference_bar(diff_vals, raw_eo.ch_names, b)
            diff_topo_path = os.path.join(folders["detailed"], f"DifferenceTopomap_{b}.png")
            diff_bar_path = os.path.join(folders["detailed"], f"DifferenceBar_{b}.png")
            diff_topo_fig.savefig(diff_topo_path, facecolor='black')
            diff_bar_fig.savefig(diff_bar_path, facecolor='black')
            plt.close(diff_topo_fig)
            plt.close(diff_bar_fig)
            global_diff_images[b] = {
                "diff_topo": os.path.basename(diff_topo_path),
                "diff_bar": os.path.basename(diff_bar_path)
            }
            print(f"Generated global difference images for {b}:")
            print(f"  Topomap: {diff_topo_path}")
            print(f"  Bar graph: {diff_bar_path}")

    site_list = raw_eo.ch_names if raw_eo else (raw_ec.ch_names if raw_ec else [])
    site_dict = {}
    for site in site_list:
        site_dict[site] = {}
        site_folder = os.path.join(folders["detailed"], site)
        psd_folder = os.path.join(site_folder, "PSD_Overlay")
        wave_folder = os.path.join(site_folder, "Waveform_Overlay")
        for b in band_list:
            psd_filename = f"{site}_PSD_{b}.png"
            wave_filename = f"{site}_Waveform_{b}.png"
            diff_topo_filename = global_diff_images.get(b, {}).get("diff_topo", "")
            diff_bar_filename = global_diff_images.get(b, {}).get("diff_bar", "")
            psd_path_rel = os.path.relpath(os.path.join(psd_folder, psd_filename), subject_folder)
            wave_path_rel = os.path.relpath(os.path.join(wave_folder, wave_filename), subject_folder)
            diff_topo_path_rel = os.path.relpath(os.path.join(folders["detailed"], diff_topo_filename),
                                                 subject_folder) if diff_topo_filename else ""
            diff_bar_path_rel = os.path.relpath(os.path.join(folders["detailed"], diff_bar_filename),
                                                subject_folder) if diff_bar_filename else ""
            site_dict[site][b] = {
                "psd": psd_path_rel,
                "wave": wave_path_rel,
                "diff_topo": diff_topo_path_rel,
                "diff_bar": diff_bar_path_rel
            }

    report_data = {
        "global_topomaps": {
            "EO": topomaps["EO"],
            "EC": topomaps["EC"]
        },
        "global_waveforms": waveforms["EO"],
        "coherence": {
            "EO": coherence["EO"],
            "EC": coherence["EC"]
        },
        "global_erp": {
            "EO": os.path.basename(erp["EO"]) if erp["EO"] else "",
            "EC": os.path.basename(erp["EC"]) if erp["EC"] else ""
        },
        "zscore": {
            "EO": zscores["EO"],
            "EC": zscores["EC"]
        },
        "tfr": {
            "EO": tfr["EO"],
            "EC": tfr["EC"]
        },
        "ica": {
            "EO": os.path.basename(ica["EO"]) if ica["EO"] else "",
            "EC": ""
        },
        "source_localization": source_localization,
        "site_list": site_list,
        "band_list": band_list,
        "site_dict": site_dict,
        "global_topomaps_path": "topomaps",
        "global_waveforms_path": "waveforms",
        "coherence_path": "coherence",
        "global_erp_path": "erp",
        "tfr_path": "tfr",
        "ica_path": "ica",
        "sites_path": "",
        "source_path": "./source_localization"
    }

    subject_report_path = os.path.join(subject_folder, "eeg_report.html")
    report.build_html_report(report_data, subject_report_path)
    print(f"Subject {subject}: Generated interactive HTML report at {subject_report_path}")


def run_extension_scripts(project_dir, subject):
    extension_script = os.path.join(project_dir, "extensions", "EEG_Extension.py")
    if os.path.exists(extension_script):
        print(f"Subject {subject}: Running extension script: {extension_script}")
        subprocess.run([sys.executable, extension_script])
    else:
        print(f"Subject {subject}: No extension script found in 'extensions' folder.")


def process_subject(subject, files, project_dir, config):
    overall_output_dir, folders, subject_folder = setup_output_directories(project_dir, subject)

    live_thread = threading.Thread(target=live_eeg_display, args=(stop_event,))
    live_thread.start()
    signal.signal(signal.SIGINT, sigint_handler)

    raw_eo, raw_ec, raw_eo_csd, raw_ec_csd = load_and_preprocess_data(project_dir, files, config['csd'])
    print(f"Loaded data for subject {subject}")

    if raw_eo is None and raw_ec is None:
        print(f"Skipping processing for subject {subject}: No valid EEG data loaded.")
        stop_event.set()
        live_thread.join()
        return

    norm_stats = load_zscore_stats(config['zscore'])

    standard_features, chosen_features = compute_zscore_features(raw_eo, config['zscore'], norm_stats)
    clinical_csv = os.path.join(project_dir, "clinical_outcomes.csv")
    clinical_outcomes = load_clinical_outcomes(clinical_csv, raw_eo.info['nchan'] if raw_eo else raw_ec.info['nchan'])
    print("Comparing standard vs. chosen z-score method:")
    compare_zscores(standard_features, chosen_features, clinical_outcomes)

    process_vigilance(raw_eo, folders)

    band_list = list(processing.BANDS.keys())

    topomaps = {"EO": {}, "EC": {}}
    for cond, raw_data in [("EO", raw_eo_csd), ("EC", raw_ec_csd)]:
        if raw_data:
            dig = raw_data.info.get("dig", None)
            has_dig = dig is not None and any(
                d['kind'] in (FIFF.FIFFV_POINT_EEG, FIFF.FIFFV_POINT_EXTRA) for d in dig
            )
            if not has_dig:
                print(f"[!] Skipping topomap processing for {cond} — no digitization data found.")
                with open("missing_channels_log.txt", "a") as log:
                    log.write(f"\n=== Skipped Topomaps for {cond} ===\n")
                    log.write("No digitization data present.\n")
                continue
            topomaps[cond] = process_topomaps(raw_data, cond, folders, band_list)

    waveforms = {
        "EO": process_waveforms(raw_eo_csd, "EO", folders, band_list) if raw_eo_csd else {}
    }

    erp = {
        "EO": process_erp(raw_eo_csd, "EO", folders) if raw_eo_csd else "",
        "EC": process_erp(raw_ec_csd, "EC", folders) if raw_ec_csd else ""
    }

    coherence = {
        "EO": process_coherence(raw_eo_csd, "EO", folders, band_list) if raw_eo_csd else {},
        "EC": process_coherence(raw_ec_csd, "EC", folders, band_list) if raw_ec_csd else {}
    }

    zscores = {
        "EO": process_zscores(raw_eo, "EO", folders, band_list, norm_stats) if raw_eo else {},
        "EC": process_zscores(raw_ec, "EC", folders, band_list, norm_stats) if raw_ec else {}
    }

    tfr = {
        "EO": process_tfr(raw_eo_csd, "EO", folders, band_list) if raw_eo_csd else {},
        "EC": process_tfr(raw_ec_csd, "EC", folders, band_list) if raw_ec_csd else {}
    }

    ica = {
        "EO": process_ica(raw_eo_csd, "EO", folders) if raw_eo_csd else ""
    }

    source_localization = process_source_localization(raw_eo, raw_ec, folders, band_list)
    print("Source Localization dictionary:", source_localization)

    if raw_eo:
        process_phenotype(raw_eo, subject_folder, subject)

    generate_reports(raw_eo, raw_ec, folders, subject_folder, subject, band_list, config, topomaps, waveforms, erp,
                     coherence, zscores, tfr, ica, source_localization)

    run_extension_scripts(project_dir, subject)

    stop_event.set()
    live_thread.join()
    print(f"Live EEG display stopped for subject {subject}")


def main():
    config = parse_arguments()

    if config['csv']:
        data_to_csv.process_edf_to_csv(config['edf_path'], config['epoch_length'], config['output_csv'])
        sys.exit(0)

    project_dir = os.getcwd()
    subject_edf_groups = find_subject_edf_files(project_dir)
    print("Found subject EDF files:", subject_edf_groups)

    for subject, files in subject_edf_groups.items():
        print(f"\nProcessing subject {subject}...")
        process_subject(subject, files, project_dir, config)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, sigint_handler)
    main()
