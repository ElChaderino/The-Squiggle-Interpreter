#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
source_localization.py

This module performs EEG source localization using sLORETA (or other inverse methods) for The Squiggle Interpreter.
It sets up a forward solution, computes noise covariance, applies an inverse operator, and visualizes the results.
"""

import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import logging
from .config import BANDS
from .plotting import plot_source_estimate

logger = logging.getLogger(__name__)

def setup_forward_solution(raw, subject="fsaverage", subjects_dir=None, ico=4, conductivity=(0.3, 0.006, 0.3)):
    """
    Set up forward solution using the fsaverage subject.
    
    Args:
        raw (mne.io.Raw): Raw EEG data.
        subject (str): Subject name (default: "fsaverage").
        subjects_dir (str): Directory for subject MRI data (default: None).
        ico (int): ICO resolution for BEM model (default: 4).
        conductivity (tuple): Conductivity values for BEM model (default: (0.3, 0.006, 0.3)).
    
    Returns:
        tuple: (fwd, src, bem_solution, subjects_dir), or (None, None, None, None) if failed.
    """
    logger.info("Setting up forward solution...")
    if not raw or not raw.info:
        logger.error("Cannot set up forward solution: Invalid raw data.")
        return None, None, None, None

    try:
        # Set up subjects_dir
        if subjects_dir is None:
            logger.debug("Fetching fsaverage dataset...")
            fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
            subjects_dir = os.path.dirname(fs_dir)
        logger.info(f"Using subjects_dir: {subjects_dir}")

        # Set montage
        logger.debug("Setting standard_1020 montage...")
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, match_case=False, on_missing='warn')

        # Create a source space
        logger.debug(f"Setting up source space for subject {subject}...")
        src = mne.setup_source_space(subject, spacing="oct6", subjects_dir=subjects_dir, add_dist=False)
        logger.debug(f"Source space created: {src}")

        # Create BEM model and solution
        logger.debug("Creating BEM model...")
        bem_model = mne.make_bem_model(subject=subject, ico=ico, conductivity=conductivity, subjects_dir=subjects_dir)
        logger.debug("Creating BEM solution...")
        bem_solution = mne.make_bem_solution(bem_model)
        logger.debug(f"BEM solution created: {bem_solution}")

        # Compute forward solution
        logger.debug("Computing forward solution...")
        trans = "fsaverage"  # Use fsaverage trans for simplicity; in practice, this should be subject-specific
        fwd = mne.make_forward_solution(raw.info, trans=trans, src=src, bem=bem_solution,
                                        eeg=True, meg=False, verbose=False)
        logger.info("Forward solution successfully set up.")
        return fwd, src, bem_solution, subjects_dir

    except Exception as e:
        logger.error(f"Failed to set up forward solution: {e}")
        return None, None, None, None

def compute_noise_covariance(epochs, tmax=0.0):
    """
    Compute noise covariance from epochs (using the pre-stimulus period).
    
    Args:
        epochs (mne.Epochs): Epochs object.
        tmax (float): Time point for covariance computation (default: 0.0).
    
    Returns:
        mne.Covariance: Noise covariance matrix, or None if failed.
    """
    logger.info("Computing noise covariance...")
    if not epochs:
        logger.error("Cannot compute noise covariance: Invalid epochs data.")
        return None

    try:
        cov = mne.compute_covariance(epochs, tmax=tmax, method="empirical", verbose=False)
        logger.info("Noise covariance successfully computed.")
        return cov
    except Exception as e:
        logger.error(f"Failed to compute noise covariance: {e}")
        return None

def compute_inverse_operator(raw, fwd, cov, loose=0.2, depth=0.8):
    """
    Construct an inverse operator.
    
    Args:
        raw (mne.io.Raw): Raw EEG data.
        fwd: Forward solution.
        cov: Noise covariance matrix.
        loose (float): Loose orientation constraint (default: 0.2).
        depth (float): Depth weighting (default: 0.8).
    
    Returns:
        mne.InverseOperator: Inverse operator, or None if failed.
    """
    logger.info("Computing inverse operator...")
    if not raw or not fwd or not cov:
        logger.error("Cannot compute inverse operator: Missing raw, forward solution, or covariance.")
        return None

    try:
        inv_op = mne.minimum_norm.make_inverse_operator(raw.info, fwd, cov, loose=loose, depth=depth, verbose=False)
        logger.info("Inverse operator successfully computed.")
        return inv_op
    except Exception as e:
        logger.error(f"Failed to compute inverse operator: {e}")
        return None

def apply_inverse_for_band(evoked, inv_op, lambda2=1.0/9.0, method="sLORETA"):
    """
    Apply inverse solution using the specified method.
    
    Args:
        evoked (mne.Evoked): The evoked response (or pseudo-ERP).
        inv_op: Inverse operator.
        lambda2 (float): Regularization parameter (default: 1.0/9.0).
        method (str): Inverse method ("sLORETA", "MNE", etc.; default: "sLORETA").
    
    Returns:
        mne.SourceEstimate: The source estimate, or None if failed.
    """
    logger.info(f"Applying inverse solution with method {method}...")
    if not evoked or not inv_op:
        logger.error("Cannot apply inverse solution: Missing evoked data or inverse operator.")
        return None

    try:
        stc = mne.minimum_norm.apply_inverse(evoked, inv_op, lambda2=lambda2,
                                             method=method, pick_ori=None, verbose=False)
        logger.info(f"Source estimate computed using {method}.")
        return stc
    except Exception as e:
        logger.error(f"Failed to apply inverse solution with {method}: {e}")
        return None

def compute_source_localization(raw, band_range, method, tmin, tmax, fwd, inv_op):
    """
    Filter raw data to a frequency band, compute epochs/pseudo-ERP, apply the inverse operator,
    and return the source estimate.
    
    Args:
        raw (mne.io.Raw): Raw EEG data.
        band_range (tuple): Frequency band (fmin, fmax).
        method (str): Inverse method (e.g., "sLORETA", "MNE").
        tmin, tmax (float): Time window for epochs.
        fwd: Forward solution (not used in this implementation but kept for compatibility).
        inv_op: Inverse operator.
    
    Returns:
        mne.SourceEstimate: The computed source estimate, or None if failed.
    """
    logger.info(f"Computing source localization for band {band_range} using method {method}...")
    if not raw or not inv_op:
        logger.error("Cannot compute source localization: Missing raw data or inverse operator.")
        return None

    try:
        # Bandpass filter the raw data to the band of interest
        raw_band = raw.copy().filter(band_range[0], band_range[1], verbose=False)
        events = mne.make_fixed_length_events(raw_band, duration=tmax-tmin)
        epochs = mne.Epochs(raw_band, events, tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose=False)
        if len(epochs) == 0:
            logger.error("No epochs generated for source localization.")
            return None
        evoked = epochs.average()
        stc = apply_inverse_for_band(evoked, inv_op, method=method)
        if stc is None:
            logger.error("Source estimate computation returned None.")
            return None
        logger.info(f"Source localization completed for band {band_range}.")
        return stc
    except Exception as e:
        logger.error(f"Failed to compute source localization for band {band_range}: {e}")
        return None

def save_source_estimate_topomap(stc, subjects_dir, subject, output_path, time_point=0.1, hemi="both", colormap="hot"):
    """
    Generate and save a screenshot of the source estimate topomap at a specific time point.
    
    Args:
        stc (mne.SourceEstimate): Source estimate.
        subjects_dir (str): Directory for subject MRI data.
        subject (str): Subject name.
        output_path (str): File path to save the image.
        time_point (float): Time point to display (default: 0.1).
        hemi (str): Hemisphere to display (default: "both").
        colormap (str): Colormap to use (default: "hot").
    """
    logger.info(f"Saving source estimate topomap to {output_path} at t={time_point}s...")
    if not stc or not subjects_dir:
        logger.error("Cannot save source estimate topomap: Missing source estimate or subjects_dir.")
        return

    try:
        # Remove the 'show' parameter and rely on time_viewer=False for non-interactive plotting
        brain = stc.plot(hemi=hemi, subjects_dir=subjects_dir, subject=subject,
                         surface="inflated", time_viewer=False, colormap=colormap,
                         smoothing_steps=10)
        brain.set_time(time_point)
        brain.show_view("lateral")
        brain.save_image(output_path)
        brain.close()
        logger.info(f"Saved source estimate topomap to {output_path}.")
    except Exception as e:
        logger.error(f"Failed to save source estimate topomap to {output_path}: {e}")

def compute_source_for_band(cond, raw_data, inv_op, band, folders, source_methods, subjects_dir):
    """
    Compute source localization for a specific band and condition, saving images for each method.
    
    Args:
        cond (str): Condition ("EO" or "EC").
        raw_data (mne.io.Raw): Raw EEG data.
        inv_op: Inverse operator.
        band (str): Frequency band name.
        folders (dict): Dictionary of output directories.
        source_methods (dict): Dictionary of method names to labels (e.g., {"sLORETA": "sLORETA"}).
        subjects_dir (str): Directory for subject MRI data.
    
    Returns:
        list: List of tuples (condition, band, method, image_path).
    """
    logger.info(f"Computing source localization for {cond}, band {band}...")
    results = []
    band_range = BANDS[band]
    cond_folder = os.path.join(folders["source"], cond)
    os.makedirs(cond_folder, exist_ok=True)
    logger.debug(f"Output directory for {cond}: {cond_folder}")

    for method, method_label in source_methods.items():
        try:
            stc = compute_source_localization(raw_data, band_range, method_label, -0.1, 0.4, None, inv_op)
            if stc is None:
                logger.warning(f"No source estimate generated for {cond}, band {band}, method {method}.")
                continue
            src_filename = f"source_{cond}_{method}_{band}.png"
            src_path = os.path.join(cond_folder, src_filename)
            save_source_estimate_topomap(stc, subjects_dir, "fsaverage", src_path, time_point=0.1)
            results.append((cond, band, method, os.path.join(cond, src_filename)))
            logger.info(f"Completed source localization for {cond}, band {band}, method {method}.")
        except Exception as e:
            logger.error(f"Error computing source localization for {cond} {band} with {method}: {e}")
            continue
    return results

def process_source_localization(raw_eo, raw_ec, folders, band_list, config=None):
    """
    Process source localization for EO and EC data across all bands and methods.
    
    Args:
        raw_eo (mne.io.Raw): Eyes-open raw EEG data (non-CSD).
        raw_ec (mne.io.Raw): Eyes-closed raw EEG data (non-CSD).
        folders (dict): Dictionary of output directories.
        band_list (list): List of frequency band names.
        config (dict, optional): Configuration dictionary with 'csd' key.
    
    Returns:
        dict: Dictionary with source localization results, structured as:
              {condition: {band: {method: image_path}}}
    """
    logger.info("Starting source localization processing...")
    source_methods = {"sLORETA": "sLORETA", "MNE": "MNE"}  # Example methods
    results = {"EO": {}, "EC": {}}

    # Use CSD-transformed data if available
    config = config or {}
    raw_eo_csd = config.get('raw_eo_csd', raw_eo)
    raw_ec_csd = config.get('raw_ec_csd', raw_ec)
    use_csd = config.get('csd', False) and raw_eo_csd and raw_ec_csd
    src_raw_eo = raw_eo_csd if use_csd else raw_eo
    src_raw_ec = raw_ec_csd if use_csd else raw_ec
    logger.info(f"Source localization using {'CSD-transformed' if use_csd else 'non-CSD'} data.")

    for cond, raw_data in [("EO", src_raw_eo), ("EC", src_raw_ec)]:
        if raw_data is None:
            logger.warning(f"Skipping source localization for {cond}: No data available.")
            continue

        try:
            # Set up forward solution
            logger.info(f"Setting up forward solution for {cond}...")
            fwd, src, bem_solution, subjects_dir = setup_forward_solution(raw_data)
            if fwd is None:
                logger.error(f"Skipping source localization for {cond}: Forward solution setup failed.")
                continue

            # Compute noise covariance
            logger.info(f"Computing noise covariance for {cond}...")
            events = mne.make_fixed_length_events(raw_data, duration=2.0)
            epochs = mne.Epochs(raw_data, events, tmin=-0.1, tmax=0.4, baseline=None, preload=True, verbose=False)
            if len(epochs) == 0:
                logger.error(f"No epochs generated for {cond} source localization.")
                continue
            cov = compute_noise_covariance(epochs)
            if cov is None:
                logger.error(f"Skipping source localization for {cond}: Noise covariance computation failed.")
                continue

            # Compute inverse operator
            logger.info(f"Computing inverse operator for {cond}...")
            inv_op = compute_inverse_operator(raw_data, fwd, cov)
            if inv_op is None:
                logger.error(f"Skipping source localization for {cond}: Inverse operator computation failed.")
                continue

            # Process each band
            for band in band_list:
                logger.info(f"Processing {cond} for band {band}...")
                results[cond][band] = {}
                band_results = compute_source_for_band(cond, raw_data, inv_op, band, folders, source_methods, subjects_dir)
                for _, band_result, method, img_path in band_results:
                    results[cond][band][method] = img_path
                    logger.debug(f"Result for {cond}, {band}, {method}: {img_path}")

        except Exception as e:
            logger.error(f"Failed to process source localization for {cond}: {e}")
            results[cond] = {}

    logger.info("Completed source localization processing.")
    return results
