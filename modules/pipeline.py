#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pipeline.py

This module orchestrates the EEG analysis pipeline for The Squiggle Interpreter.
It integrates data loading, preprocessing, clinical analysis, plotting, phenotype classification,
vigilance analysis, source localization, and report generation (HTML and PDF).
"""

import os
import logging
from threading import Thread
from multiprocessing import Process, Queue
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import mne
import numpy as np
import matplotlib.pyplot as plt
import argparse
from . import report
from .pdf_report_builder import build_pdf_report
from .clinical_report import generate_reports as generate_clinical_reports

# Configure logging to include DEBUG level
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log', mode='w'),  # Overwrite to avoid clutter
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import modules
from .live_display import live_eeg_display, stop_event
from .io_utils import load_and_preprocess_data, setup_output_directories, load_clinical_outcomes, run_extension_scripts
from .vigilance import process_vigilance, select_vigilance_channels
from .clinical_report import generate_reports, process_phenotype
from .source_localization import process_source_localization
from .processing import compute_zscore_features, compare_zscores, compute_all_band_powers
from .plotting import (
    process_topomaps, process_waveforms, process_erp, process_coherence,
    process_zscores, process_variance_topomaps, process_tfr, process_ica,
    generate_full_site_plots, plot_difference_topomap, plot_difference_bar
)
from .clinical import generate_full_site_reports, compute_site_metrics, generate_site_reports
from .config import BANDS, load_zscore_stats, OUTPUT_FOLDERS
from .report_writer import format_phenotype_section, write_html_report
from .feature_extraction import extract_classification_features
from .phenotype import classify_eeg_profile
from .pdf_report_builder import build_pdf_report
from . import report

class TaskManager:
    def __init__(self):
        self.tasks = []
        self.queues = []

    def add_task(self, func, args):
        queue = Queue()
        self.tasks.append((func, args, queue))
        self.queues.append(queue)

    def run(self):
        processes = []
        for func, args, queue in self.tasks:
            p = Process(target=self.execute_task_with_queue, args=(func, args, queue))
            processes.append(p)
            p.start()
        results = [q.get() for q in self.queues]
        for p in processes:
            p.join()
        return results

    @staticmethod
    def execute_task_with_queue(func, args, queue):
        try:
            result = func(*args)
            queue.put(result)
        except Exception as e:
            logger.error(f"Task {func.__name__} failed: {e}")
            queue.put({})

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

def parse_arguments():
    parser = argparse.ArgumentParser(
        prog='The Squiggle Interpreter Pipeline',
        description='Comprehensive EEG Analysis & Clinical Report Generation'
    )
    parser.add_argument('--csd', choices=['y', 'n'], default='n', help="Use current source density (CSD) for graphs? (y/n, default: n)")
    parser.add_argument('--zscore', choices=['1', '2', '3', '4'], default='1',
                        help="Z-score normalization method: 1: Standard, 2: Robust (MAD), 3: Robust (IQR), 4: Published Norms")
    parser.add_argument('--report', choices=['y', 'n'], default='y', help="Generate full clinical report? (y/n, default: y)")
    parser.add_argument('--phenotype', choices=['y', 'n'], default='y', help="Generate phenotype classification? (y/n, default: y)")
    parser.add_argument('--data-dir', default='.', help="Directory containing EDF files")

    args = parser.parse_args()

    config = {
        'csd': args.csd == 'y',
        'zscore': args.zscore,
        'report': args.report == 'y',
        'phenotype': args.phenotype == 'y',
        'data_dir': args.data_dir
    }
    return config

def process_subject(subject_id, files, project_dir, config):
    # Setup directories
    overall_output_dir, folders, subject_folder = setup_output_directories(project_dir, subject_id)
    logger.info(f"Processing subject {subject_id} in {subject_folder}")

    live_thread = Thread(target=live_eeg_display, args=(stop_event,))
    live_thread.start()

    try:
        # Step 1: Load and preprocess data
        logger.info("Starting Step 1: Loading and preprocessing data...")
        raw_eo, raw_ec, raw_eo_csd, raw_ec_csd = load_and_preprocess_data(project_dir, files, config['csd'])
        logger.info(f"Completed Step 1: Loaded data for subject {subject_id}: raw_eo={bool(raw_eo)}, raw_ec={bool(raw_ec)}, raw_eo_csd={bool(raw_eo_csd)}, raw_ec_csd={bool(raw_ec_csd)}")

        if raw_eo is None and raw_ec is None and raw_eo_csd is None and raw_ec_csd is None:
            logger.warning(f"Skipping processing for subject {subject_id}: No valid EEG data loaded.")
            return

        # Step 2: Channels to process for vigilance
        logger.info("Starting Step 2: Selecting channels for vigilance processing...")
        available_channels = raw_eo.ch_names if raw_eo else (raw_ec.ch_names if raw_ec else [])
        if not available_channels:
            logger.warning(f"Skipping vigilance processing for subject {subject_id}: No channels available.")
            return

        occipital_candidates = ['OZ', 'O1', 'O2', 'PZ']
        channels_to_process = config.get('vigilance_channels', occipital_candidates)
        channels_to_process = [ch for ch in channels_to_process if
                               ch.upper() in [c.upper() for c in available_channels]]

        if not channels_to_process:
            logger.warning(
                f"Skipping vigilance processing for subject {subject_id}: No occipital channels available. Candidates: {occipital_candidates}, Available: {available_channels}")
            return

        if available_channels:
            is_uppercase = available_channels[0].isupper()
            if is_uppercase:
                channels_to_process = [ch.upper() for ch in channels_to_process]
            else:
                channels_to_process = [ch.lower() for ch in channels_to_process]

            # Validate channel availability for non-CSD and CSD data
            for raw_data, cond in [(raw_eo, "EO"), (raw_ec, "EC"), (raw_eo_csd, "EO_CSD"), (raw_ec_csd, "EC_CSD")]:
                if raw_data:
                    valid_channels = []
                    for ch_name in channels_to_process:
                        if ch_name not in raw_data.ch_names:
                            logger.warning(
                                f"Channel '{ch_name}' not found in {cond}. Available channels: {raw_data.ch_names}. Skipping this channel.")
                            continue
                        valid_channels.append(ch_name)
                    if valid_channels:
                        logger.info(f"Processing {cond} data for subject {subject_id} using channels {valid_channels}")
                        try:
                            process_vigilance(raw_data, subject_folder, cond, valid_channels)
                        except Exception as e:
                            logger.error(f"Failed to process vigilance for {cond}: {e}")
                            raise
        logger.info("Completed Step 2: Vigilance processing")

        # Step 3: Compute z-score features
        logger.info("Starting Step 3: Computing z-score features...")
        norm_stats = load_zscore_stats(config['zscore'])
        try:
            standard_features, chosen_features = compute_zscore_features(raw_eo, config['zscore'], norm_stats)
            clinical_csv = os.path.join(project_dir, "clinical_outcomes.csv")
            clinical_outcomes = load_clinical_outcomes(clinical_csv,
                                                       raw_eo.info['nchan'] if raw_eo else raw_ec.info['nchan'])
            logger.info("Comparing standard vs. chosen z-score method:")
            compare_zscores(standard_features, chosen_features, clinical_outcomes)
        except Exception as e:
            logger.error(f"Failed to compute z-score features: {e}")
            raise
        logger.info("Completed Step 3: Z-score features computation")

        # Step 4: Parallel processing of visualizations
        logger.info("Starting Step 4: Setting up visualization tasks...")
        band_list = list(BANDS.keys())
        task_manager = TaskManager()

        tasks = [
            (process_topomaps, (raw_eo_csd, "EO", folders, band_list)) if raw_eo_csd else None,
            (process_topomaps, (raw_ec_csd, "EC", folders, band_list)) if raw_ec_csd else None,
            (process_waveforms, (raw_eo_csd, "EO", folders, band_list)) if raw_eo_csd else None,
            (process_erp, (raw_eo_csd, "EO", folders)) if raw_eo_csd else None,
            (process_erp, (raw_ec_csd, "EC", folders)) if raw_ec_csd else None,
            (process_coherence, (raw_eo_csd, "EO", folders, band_list)) if raw_eo_csd else None,
            (process_coherence, (raw_ec_csd, "EC", folders, band_list)) if raw_ec_csd else None,
            (process_zscores, (raw_eo, "EO", folders, band_list, norm_stats)) if raw_eo else None,
            (process_zscores, (raw_ec, "EC", folders, band_list, norm_stats)) if raw_ec else None,
            (process_variance_topomaps, (raw_eo, "EO", folders, band_list)) if raw_eo else None,
            (process_variance_topomaps, (raw_ec, "EC", folders, band_list)) if raw_ec else None,
            (process_tfr, (raw_eo_csd, "EO", folders, band_list)) if raw_eo_csd else None,
            (process_tfr, (raw_ec_csd, "EC", folders, band_list)) if raw_ec_csd else None,
            (process_ica, (raw_eo_csd, "EO", folders)) if raw_eo_csd else None
        ]
        tasks = [task for task in tasks if task]

        for task in tasks:
            task_manager.add_task(*task)

        os.environ['NUMBA_DISABLE_JIT'] = '1'
        logger.info("Running visualization tasks in parallel...")
        results = task_manager.run()
        os.environ['NUMBA_DISABLE_JIT'] = '0'
        logger.info("Completed Step 4: Visualization tasks")

        # Parse parallel task results
        result_idx = 0
        topomaps = {"EO": {}, "EC": {}}
        if raw_eo_csd:
            topomaps["EO"] = results[result_idx]
            result_idx += 1
        if raw_ec_csd:
            topomaps["EC"] = results[result_idx]
            result_idx += 1
        logger.debug(f"topomaps after Step 4: {topomaps}")

        waveforms = {"EO": {}}
        if raw_eo_csd:
            waveforms["EO"] = results[result_idx]
            result_idx += 1
        logger.debug(f"waveforms after Step 4: {waveforms}")

        erp = {"EO": "", "EC": ""}
        if raw_eo_csd:
            erp["EO"] = results[result_idx]
            result_idx += 1
        if raw_ec_csd:
            erp["EC"] = results[result_idx]
            result_idx += 1
        logger.debug(f"erp after Step 4: {erp}")

        coherence = {"EO": {}, "EC": {}}
        if raw_eo_csd:
            coherence["EO"] = results[result_idx]
            result_idx += 1
        if raw_ec_csd:
            coherence["EC"] = results[result_idx]
            result_idx += 1
        logger.debug(f"coherence after Step 4: {coherence}")

        zscores = {"EO": {}, "EC": {}}
        if raw_eo:
            zscores["EO"] = results[result_idx]
            result_idx += 1
        if raw_ec:
            zscores["EC"] = results[result_idx]
            result_idx += 1
        logger.debug(f"zscores after Step 4: {zscores}")

        variance = {"EO": {}, "EC": {}}
        if raw_eo:
            variance["EO"] = results[result_idx]
            result_idx += 1
        if raw_ec:
            variance["EC"] = results[result_idx]
            result_idx += 1
        logger.debug(f"variance after Step 4: {variance}")

        tfr = {"EO": {}, "EC": {}}
        if raw_eo_csd:
            tfr["EO"] = results[result_idx]
            result_idx += 1
        if raw_ec_csd:
            tfr["EC"] = results[result_idx]
            result_idx += 1
        logger.debug(f"tfr after Step 4: {tfr}")

        ica = {"EO": ""}
        if raw_eo_csd:
            ica["EO"] = results[result_idx]
            result_idx += 1
        logger.debug(f"ica after Step 4: {ica}")

        # Step 5: Source localization
        logger.info("Starting Step 5: Processing source localization...")
        try:
            source_localization = process_source_localization(raw_eo, raw_ec, folders, band_list)
            logger.info(f"Source Localization dictionary: {source_localization}")
        except Exception as e:
            logger.error(f"Failed to process source localization: {e}")
            source_localization = {"EO": {}, "EC": {}}
        logger.info("Completed Step 5: Source localization")

        # Step 6: Generate phenotype classification and phenotype_report.html
        logger.info("Starting Step 6: Generating phenotype classification...")
        classification_result = {}
        if config.get('phenotype', True) and raw_eo:
            try:
                # Prepare sloreta_data with numerical values
                sloreta_data = {
                    'frontal_hi_beta': source_localization['EO'].get('HighBeta', {}).get('sLORETA_frontal_power', 0),
                    'parietal_alpha': source_localization['EO'].get('Alpha', {}).get('sLORETA_parietal_power', 0)
                }
                logger.debug(f"sloreta_data: {sloreta_data}")
                
                features = extract_classification_features(
                    raw_eo, [],
                    eyes_open_raw=raw_eo,
                    eyes_closed_raw=raw_ec,
                    csd_raw=raw_eo_csd,
                    sloreta_data=sloreta_data
                )
                # Log the features and convert any string values to floats
                logger.debug(f"Features before classification: {features}")
                for key, value in features.items():
                    if isinstance(value, str):
                        try:
                            features[key] = float(value)
                            logger.debug(f"Converted feature[{key}] from string '{value}' to float {features[key]}")
                        except ValueError:
                            logger.error(f"Feature[{key}] is a string '{value}' but cannot be converted to float")
                            raise
                classification_result = classify_eeg_profile(features, verbose=True)
                phenotype_html = format_phenotype_section(classification_result)

                template_path = os.path.join(project_dir, "templates", "report_template.html")
                if os.path.exists(template_path):
                    with open(template_path, encoding="utf-8") as f:
                        base_html = f.read()
                    phenotype_report_path = os.path.join(subject_folder, "phenotype_report.html")
                    write_html_report(phenotype_report_path, base_html, phenotype_html)
                    logger.info(f"{subject_id}: Wrote phenotype HTML report to {phenotype_report_path}")
                else:
                    logger.warning(f"{subject_id}: Phenotype report skipped: template not found at {template_path}")
            except Exception as e:
                logger.error(f"Failed to generate phenotype classification: {e}")
                classification_result = {}
        logger.info("Completed Step 6: Phenotype classification")

        # Update phenotype data in report_data
        phenotype_data = classification_result if classification_result else {
            "best_match": "Unknown",
            "confidence": 0.0,
            "explanations": ["No phenotype data available."],
            "recommendations": [],
            "zscore_summary": {}
        }

        # Step 7: Generate main reports
        logger.info("Starting Step 7: Generating main reports...")
        generate_reports(raw_eo, raw_ec, folders, subject_folder, subject_id, band_list, config, topomaps, waveforms,
                         erp, coherence, zscores, variance, tfr, ica, source_localization, phenotype_data)
        logger.info("Completed Step 7: Main reports generation")

        # Step 8: Run extension scripts
        logger.info("Starting Step 8: Running extension scripts...")
        run_extension_scripts(project_dir, subject_id)
        logger.info("Completed Step 8: Extension scripts")

    except Exception as e:
        logger.error(f"Pipeline failed for subject {subject_id}: {e}")
        raise
    finally:
        stop_event.set()
        live_thread.join()
        logger.info(f"Live EEG display stopped for subject {subject_id}")

def generate_reports(raw_eo, raw_ec, folders, subject_folder, subject, band_list, config, topomaps, waveforms, erp,
                     coherence, zscores, variance, tfr, ica, source_localization, phenotype_data):
    try:
        logger.info("Attempting to generate reports...")
        if not config["report"]:
            logger.warning("Report generation skipped: config['report'] is False.")
            return

        logger.info("Report generation is enabled (config['report'] = True).")

        # Log dictionaries immediately
        logger.debug(f"topomaps: {topomaps}")
        logger.debug(f"coherence: {coherence}")
        logger.debug(f"zscores: {zscores}")
        logger.debug(f"variance: {variance}")
        logger.debug(f"tfr: {tfr}")
        logger.debug(f"waveforms: {waveforms}")
        logger.debug(f"erp: {erp}")
        logger.debug(f"ica: {ica}")
        logger.debug(f"band_list: {band_list}")
        logger.debug(f"phenotype_data: {phenotype_data}")

        # Compute band powers
        logger.info("Step 7.1: Computing band powers...")
        bp_eo = compute_all_band_powers(raw_eo) if raw_eo else {}
        bp_ec = compute_all_band_powers(raw_ec) if raw_ec else {}
        logger.info(f"Subject {subject} - Computed band powers for EO channels: {list(bp_eo.keys()) if bp_eo else 'None'}")
        logger.info(f"Subject {subject} - Computed band powers for EC channels: {list(bp_ec.keys()) if bp_ec else 'None'}")
        logger.debug(f"Sample bp_eo: {list(bp_eo.items())[:1]}")
        logger.debug(f"Sample bp_ec: {list(bp_ec.items())[:1]}")

        # Generate clinical site reports
        logger.info("Step 7.2: Generating clinical site reports...")
        if bp_eo or bp_ec:
            try:
                generate_site_reports(bp_eo, bp_ec, subject_folder)
            except Exception as e:
                logger.error(f"Failed to generate clinical site reports: {e}")

        # Generate full site plots
        logger.info("Step 7.3: Generating full site plots...")
        if raw_eo and raw_ec:
            try:
                generate_full_site_reports(raw_eo, raw_ec, folders["detailed"])
            except Exception as e:
                logger.error(f"Failed to generate full site plots: {e}")

        # Generate difference plots
        logger.info("Step 7.4: Generating difference plots...")
        global_diff_images = {}
        if raw_eo and raw_ec:
            for b in band_list:
                try:
                    diff_vals = [bp_eo[ch][b] - bp_ec[ch][b] for ch in raw_eo.ch_names]
                    diff_topo_fig = plot_difference_topomap(diff_vals, raw_eo.info, b)
                    diff_bar_fig = plot_difference_bar(diff_vals, raw_eo.ch_names, b)
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
                    logger.info(f"Generated global difference images for {b}:")
                    logger.info(f"  Topomap: {diff_topo_path}")
                    logger.info(f"  Bar graph: {diff_bar_path}")
                    gc.collect()
                except Exception as e:
                    logger.warning(f"Failed to generate difference images for {b}: {e}")

        # Prepare site_dict
        logger.info("Step 7.5: Preparing site_dict...")
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
                diff_bar_filename = global_diff_images.get(b, {}).get("diff_bar", "")
                diff_topo_filename = global_diff_images.get(b, {}).get("diff_topo", "")
                psd_path = os.path.join(psd_folder, psd_filename)
                wave_path = os.path.join(wave_folder, wave_filename)
                diff_bar_path = os.path.join(folders["detailed"], diff_bar_filename) if diff_bar_filename else ""
                diff_topo_path = os.path.join(folders["detailed"], diff_topo_filename) if diff_topo_filename else ""
                psd_path_rel = os.path.relpath(psd_path, start=subject_folder).replace('\\', '/') if os.path.exists(psd_path) else ""
                wave_path_rel = os.path.relpath(wave_path, start=subject_folder).replace('\\', '/') if os.path.exists(wave_path) else ""
                diff_bar_path_rel = os.path.relpath(diff_bar_path, start=subject_folder).replace('\\', '/') if diff_bar_filename and os.path.exists(diff_bar_path) else ""
                diff_topo_path_rel = os.path.relpath(diff_topo_path, start=subject_folder).replace('\\', '/') if diff_topo_filename and os.path.exists(diff_topo_path) else ""
                site_dict[site][b] = {
                    "psd": psd_path_rel,
                    "wave": wave_path_rel,
                    "diff_bar": diff_bar_path_rel,
                    "diff_topo": diff_topo_path_rel
                }
                logger.debug(f"Site {site} Band {b}: psd_path={psd_path}, exists={os.path.exists(psd_path)}, rel={psd_path_rel}")
                logger.debug(f"Site {site} Band {b}: wave_path={wave_path}, exists={os.path.exists(wave_path)}, rel={wave_path_rel}")
                logger.debug(f"Site {site} Band {b}: diff_bar_path={diff_bar_path}, exists={os.path.exists(diff_bar_path) if diff_bar_path else False}, rel={diff_bar_path_rel}")
                logger.debug(f"Site {site} Band {b}: diff_topo_path={diff_topo_path}, exists={os.path.exists(diff_topo_path) if diff_topo_path else False}, rel={diff_topo_path_rel}")
                if not psd_path_rel and os.path.exists(psd_path):
                    logger.warning(f"PSD plot exists but path not included for {site} {b}: {psd_path}")
                if not wave_path_rel and os.path.exists(wave_path):
                    logger.warning(f"Waveform plot exists but path not included for {site} {b}: {wave_path}")
                if not diff_bar_path_rel and diff_bar_filename and os.path.exists(diff_bar_path):
                    logger.warning(f"Difference bar plot exists but path not included for {site} {b}: {diff_bar_path}")
                if not diff_topo_path_rel and diff_topo_filename and os.path.exists(diff_topo_path):
                    logger.warning(f"Difference topomap exists but path not included for {site} {b}: {diff_topo_path}")
        logger.info("Completed Step 7.5: site_dict preparation")
        logger.debug(f"site_dict sample: {dict(list(site_dict.items())[:1])}")
        logger.debug(f"subject_folder: {subject_folder}")
        logger.debug(f"folders['detailed']: {folders['detailed']}")

        # Prepare hypnogram data
        logger.info("Step 7.6: Preparing hypnogram data...")
        hypnograms = {
            "EO": {},
            "EC": {},
            "EO_CSD": {},
            "EC_CSD": {}
        }
        if raw_eo:
            logger.debug(f"Type of raw_eo.ch_names: {type(raw_eo.ch_names)}")
            logger.debug(f"raw_eo.ch_names: {raw_eo.ch_names}")
        if raw_ec:
            logger.debug(f"Type of raw_ec.ch_names: {type(raw_ec.ch_names)}")
            logger.debug(f"raw_ec.ch_names: {raw_ec.ch_names}")
        available_channels = raw_eo.ch_names if raw_eo else (raw_ec.ch_names if raw_ec else [])
        logger.debug(f"Type of available_channels: {type(available_channels)}")
        logger.debug(f"available_channels: {available_channels}")
        for ch_name in available_channels:
            if not isinstance(ch_name, str):
                logger.error(f"Found non-string channel name: {ch_name} (type: {type(ch_name)})")
                raise ValueError(f"Channel name must be a string, got {type(ch_name)}: {ch_name}")
            for condition in ["EO", "EC", "EO_CSD", "EC_CSD"]:
                hypnogram_file = f"vigilance_hypnogram_{condition}_{ch_name}.png"
                hypnogram_path = os.path.join(subject_folder, OUTPUT_FOLDERS["vigilance"], hypnogram_file)
                if os.path.exists(hypnogram_path):
                    hypnograms[condition][ch_name] = hypnogram_file
        logger.info(f"Completed Step 7.6: Hypnogram data prepared: {hypnograms}")
        logger.debug(f"hypnograms: {hypnograms}")

        # Prepare report data for HTML
        logger.info("Step 7.7: Preparing report data for HTML generation...")
        report_data = {
            "global_topomaps": {
                "EO": topomaps.get("EO", {}),
                "EC": topomaps.get("EC", {})
            },
            "global_waveforms": waveforms.get("EO", {}),
            "coherence": {
                "EO": coherence.get("EO", {}),
                "EC": coherence.get("EC", {})
            },
            "global_erp": {
                "EO": os.path.basename(erp.get("EO", "")) if erp.get("EO") else "",
                "EC": os.path.basename(erp.get("EC", "")) if erp.get("EC") else ""
            },
            "zscore": {
                "EO": zscores.get("EO", {}),
                "EC": zscores.get("EC", {})
            },
            "variance": {
                "EO": variance.get("EO", {}),
                "EC": variance.get("EC", {})
            },
            "tfr": {
                "EO": tfr.get("EO", {}),
                "EC": tfr.get("EC", {})
            },
            "ica": {
                "EO": os.path.basename(ica.get("EO", "")) if ica.get("EO") else "",
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
            "source_path": "./source_localization",
            "phenotype": phenotype_data,
            "hypnograms": hypnograms,
            "phenotype_html": format_phenotype_section(phenotype_data)
        }
        logger.debug(f"Report data prepared. Keys: {list(report_data.keys())}")
        logger.debug(f"Sample data: global_topomaps={report_data['global_topomaps']}")
        logger.debug(f"Sample data: site_dict={dict(list(report_data['site_dict'].items())[:1])}")
        logger.info("Completed Step 7.7: Report data preparation")

        # Generate HTML report
        logger.info("Step 7.8: Generating HTML report...")
        subject_report_path = os.path.join(subject_folder, "eeg_report.html")
        logger.info(f"Generating HTML report at {subject_report_path}...")
        report.build_html_report(report_data, subject_report_path)
        logger.info(f"Subject {subject}: Generated interactive HTML report at {subject_report_path}")

        # Generate PDF report
        logger.info("Step 7.9: Generating clinical PDF report...")
        if raw_eo or raw_ec:
            try:
                logger.info("Generating clinical PDF report...")
                instability_eo = compute_instability_index(raw_eo, BANDS) if raw_eo else {}
                instability_ec = compute_instability_index(raw_ec, BANDS) if raw_ec else {}
                build_pdf_report(
                    report_output_dir=Path(subject_folder),
                    band_powers={"EO": bp_eo, "EC": bp_ec},
                    instability_indices={"EO": instability_eo, "EC": instability_ec},
                    source_localization=source_localization,
                    vigilance_plots=hypnograms,
                    channels=site_list
                )
                logger.info(f"PDF report generated at: {subject_folder}/clinical_report.pdf")
            except Exception as e:
                logger.error(f"Failed to generate clinical PDF report: {e}")
                raise

        # Generate text reports via clinical_report.py
        logger.info("Step 7.10: Generating text reports via clinical_report...")
        try:
            generate_clinical_reports(
                raw_eo=raw_eo,
                raw_ec=raw_ec,
                folders=folders,
                subject_folder=subject_folder,
                subject=subject,
                band_list=band_list,
                config=config,
                topomaps=topomaps,
                waveforms=waveforms,
                erp=erp,
                coherence=coherence,
                zscores=zscores,
                variance=variance,
                tfr=tfr,
                ica=ica,
                source_localization=source_localization,
                phenotype_data=phenotype_data,
                site_dict=site_dict,
                skip_html=True
            )
            logger.info("Completed text report generation via clinical_report.")
        except Exception as e:
            logger.error(f"Failed to generate text reports via clinical_report: {e}")

    except Exception as e:
        logger.error(f"Error generating reports for subject {subject}: {e}")
        raise

        # Generate text reports via clinical_report.py
        logger.info("Step 7.10: Generating text reports via clinical_report...")
        try:
            generate_clinical_reports(
                raw_eo=raw_eo,
                raw_ec=raw_ec,
                folders=folders,
                subject_folder=subject_folder,
                subject=subject,
                band_list=band_list,
                config=config,
                topomaps=topomaps,
                waveforms=waveforms,
                erp=erp,
                coherence=coherence,
                zscores=zscores,
                variance=variance,
                tfr=tfr,
                ica=ica,
                source_localization=source_localization,
                phenotype_data=phenotype_data,
                site_dict=site_dict,
                skip_html=True  # Prevent duplicate HTML report
            )
            logger.info("Completed text report generation via clinical_report.")
        except Exception as e:
            logger.error(f"Failed to generate text reports via clinical_report: {e}")

    except Exception as e:
        logger.error(f"Error generating reports for subject {subject}: {e}")
        raise

        # Generate text reports via clinical_report.py
        logger.info("Step 7.10: Generating text reports via clinical_report...")
        try:
            generate_clinical_reports(
                raw_eo=raw_eo,
                raw_ec=raw_ec,
                folders=folders,
                subject_folder=subject_folder,
                subject=subject,
                band_list=band_list,
                config=config,
                topomaps=topomaps,
                waveforms=waveforms,
                erp=erp,
                coherence=coherence,
                zscores=zscores,
                variance=variance,
                tfr=tfr,
                ica=ica,
                source_localization=source_localization,
                phenotype_data=phenotype_data,
                site_dict=site_dict,
                skip_html=True  # Prevent duplicate HTML report
            )
            logger.info("Completed text report generation via clinical_report.")
        except Exception as e:
            logger.error(f"Failed to generate text reports via clinical_report: {e}")

    except Exception as e:
        logger.error(f"Error generating reports for subject {subject}: {e}")
        raise

def main():
    config = parse_arguments()
    project_dir = config['data_dir']
    subject_edf_groups = find_subject_edf_files(project_dir)
    logger.info(f"Found subject EDF files: {subject_edf_groups}")

    for subject, files in subject_edf_groups.items():
        logger.info(f"\nProcessing subject {subject}...")
        process_subject(subject, files, project_dir, config)

if __name__ == "__main__":
    main()
