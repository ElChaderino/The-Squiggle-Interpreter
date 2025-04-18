#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pipeline.py
A central subject processing pipeline that applies CSD, extracts features,
classifies EEG phenotype, and generates various reports.
"""

import os
import numpy as np
import mne
from feature_extraction import extract_classification_features
from phenotype import classify_eeg_profile
from report_writer import format_phenotype_section, write_html_report
from modules import processing, plotting, clinical, report


def process_subject(subject_id, raw_eo, raw_ec, raw_eo_csd, raw_ec_csd, args, folders, project_dir, source_localization,
                    vigilance_states):
    """
    Central subject pipeline: applies CSD, extracts features, classifies, generates all reports.

    Parameters:
        subject_id (str): Identifier for the subject.
        raw_eo (mne.io.Raw): Preprocessed raw data for Eyes Open.
        raw_ec (mne.io.Raw): Preprocessed raw data for Eyes Closed.
        raw_eo_csd (mne.io.Raw): CSD-transformed raw data for Eyes Open (for graphing).
        raw_ec_csd (mne.io.Raw): CSD-transformed raw data for Eyes Closed (for graphing).
        args (Namespace): Command-line arguments or configuration.
        folders (dict): Dictionary of output folders (including 'base' and 'detailed').
        project_dir (str): Project directory path.
        source_localization (dict): Source localization results.
        vigilance_states (list): Computed vigilance state data.

    Returns:
        dict: A summary dictionary including subject ID, phenotype classification, and band power data.
    """
    # Shared band definitions
    band_list = list(processing.BANDS.keys())

    # --- Compute bandpowers ---
    bp_eo = processing.compute_all_band_powers(raw_eo)
    bp_ec = processing.compute_all_band_powers(raw_ec)

    # --- Generate clinical site reports ---
    clinical.generate_site_reports(bp_eo, bp_ec, folders['base'])

    # --- Phenotype classification ---
    if getattr(args, 'phenotype', True):
        features = extract_classification_features(
            raw_eo, vigilance_states,
            eyes_open_raw=raw_eo,
            eyes_closed_raw=raw_ec,
            csd_raw=raw_eo_csd,
            sloreta_data={
                'frontal_hi_beta': source_localization['EO'].get('HighBeta', {}).get('sLORETA', 0),
                'parietal_alpha': source_localization['EO'].get('Alpha', {}).get('sLORETA', 0)
            }
        )

        classification_result = classify_eeg_profile(features, verbose=True)
        phenotype_html = format_phenotype_section(classification_result)

        template_path = os.path.join(project_dir, "report_template.html")
        if os.path.exists(template_path):
            with open(template_path) as f:
                base_html = f.read()
            phenotype_report_path = os.path.join(folders['base'], "phenotype_report.html")
            write_html_report(phenotype_report_path, base_html, phenotype_html)
            print(f"{subject_id}: Wrote phenotype HTML report to {phenotype_report_path}")
        else:
            print(f"{subject_id}: Phenotype report skipped: template not found.")

    # --- Site-level plots for detailed report ---
    from modules.clinical import generate_full_site_reports
    generate_full_site_reports(raw_eo, raw_ec, folders['detailed'])

    # --- Topomaps, z-scores, TFR, ICA, etc. ---
    # (Additional processing can be modularized here in the future)

    # --- Return summary if needed ---
    return {
        'subject': subject_id,
        'phenotype': classification_result if args.phenotype else None,
        'bp_eo': bp_eo,
        'bp_ec': bp_ec
    }
