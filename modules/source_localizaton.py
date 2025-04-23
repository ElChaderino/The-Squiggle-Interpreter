# In modules/source_localization.py (new module)

import os
import mne
import numpy as np
import matplotlib.pyplot as plt

def setup_forward_solution(raw, subject="fsaverage", subjects_dir=None, ico=4, conductivity=(0.3, 0.006, 0.3)):
    """
    Set up forward solution using the fsaverage subject.
    """
    if subjects_dir is None:
        fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
        subjects_dir = os.path.dirname(fs_dir)
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, match_case=False)
    
    # Create a source space.
    src = mne.setup_source_space(subject, spacing="oct6", subjects_dir=subjects_dir, add_dist=False)
    
    # Create BEM model and solution.
    bem_model = mne.make_bem_model(subject=subject, ico=ico, conductivity=conductivity, subjects_dir=subjects_dir)
    bem_solution = mne.make_bem_solution(bem_model)
    
    # Compute forward solution.
    fwd = mne.make_forward_solution(raw.info, trans="fsaverage", src=src, bem=bem_solution,
                                    eeg=True, meg=False, verbose=False)
    return fwd, src, bem_solution, subjects_dir

def compute_noise_covariance(epochs, tmax=0.0):
    """
    Compute noise covariance from epochs (using the pre-stimulus period).
    """
    cov = mne.compute_covariance(epochs, tmax=tmax, method="empirical", verbose=False)
    return cov

def compute_inverse_operator(raw, fwd, cov, loose=0.2, depth=0.8):
    """
    Construct an inverse operator.
    """
    inv_op = mne.minimum_norm.make_inverse_operator(raw.info, fwd, cov, loose=loose, depth=depth, verbose=False)
    return inv_op

def apply_inverse_for_band(evoked, inv_op, lambda2=1.0/9.0, method="sLORETA"):
    """
    Apply inverse solution using the specified method.
    
    Parameters:
      evoked (mne.Evoked): The evoked response (or pseudo-ERP).
      inv_op: Inverse operator.
      lambda2 (float): Regularization parameter.
      method (str): "sLORETA", "MNE" (for LORETA-like, adjust parameters), etc.
      
    Returns:
      mne.SourceEstimate: The source estimate.
    """
    stc = mne.minimum_norm.apply_inverse(evoked, inv_op, lambda2=lambda2,
                                         method=method, pick_ori=None, verbose=False)
    return stc

def compute_source_localization(raw, band_range, method, tmin, tmax, fwd, inv_op):
    """
    Filter raw data to a frequency band, compute epochs/pseudo-ERP, apply the inverse operator,
    and return the source estimate.
    
    Parameters:
      raw (mne.io.Raw): Raw EEG data.
      band_range (tuple): Frequency band (fmin, fmax).
      method (str): Inverse method, e.g., "sLORETA" or "MNE".
      tmin, tmax (float): Time window for epochs.
      fwd: Forward solution.
      inv_op: Inverse operator.
      
    Returns:
      mne.SourceEstimate: The computed source estimate.
    """
    # Bandpass filter the raw data to the band of interest.
    raw_band = raw.copy().filter(band_range[0], band_range[1], verbose=False)
    events = mne.make_fixed_length_events(raw_band, duration=tmax-tmin)
    epochs = mne.Epochs(raw_band, events, tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose=False)
    evoked = epochs.average()
    stc = apply_inverse_for_band(evoked, inv_op, method=method)
    return stc

def save_source_estimate_topomap(stc, subjects_dir, subject, output_path, time_point=0.1, hemi="both", colormap="hot"):
    """
    Generate and save a screenshot of the source estimate topomap at a specific time point.
    
    Parameters:
      stc (mne.SourceEstimate): Source estimate.
      subjects_dir (str): Directory for subject MRI data.
      subject (str): Subject name.
      output_path (str): File path to save the image.
      time_point (float): Time point to display.
      hemi (str): Hemisphere to display.
      colormap (str): Colormap to use.
    """
    brain = stc.plot(hemi=hemi, subjects_dir=subjects_dir, subject=subject,
                     surface="inflated", time_viewer=False, colormap=colormap,
                     smoothing_steps=10, show=False)
    brain.set_time(time_point)
    brain.save_image(output_path)
    brain.close()

# Example function to loop over all bands and methods for a given raw data.
def compute_and_save_source_maps(raw, methods, output_base, tmin=0.0, tmax=0.4):
    """
    For each frequency band in BANDS and for each specified inverse method, compute
    the source estimate and save the topomap image.
    
    Parameters:
      raw (mne.io.Raw): Raw EEG data.
      methods (list): List of inverse methods, e.g., ["sLORETA", "MNE"].
      output_base (str): Base output directory to save images.
      tmin, tmax (float): Time window for epochs.
    """
    # Set up forward model.
    fwd, src, bem_solution, subjects_dir = setup_forward_solution(raw)
    # Compute noise covariance from raw data's fixed-length epochs (using tmax=0 for pre-stimulus).
    events = mne.make_fixed_length_events(raw, duration=tmax-tmin)
    epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose=False)
    cov = compute_noise_covariance(epochs, tmax=0.0)
    inv_op = compute_inverse_operator(raw, fwd, cov)
    
    subject = "fsaverage"  # Or your subject name.
    
    for band, band_range in BANDS.items():
        for method in methods:
            stc = compute_source_localization(raw, band_range, method, tmin, tmax, fwd, inv_op)
            out_dir = os.path.join(output_base, method, band)
            os.makedirs(out_dir, exist_ok=True)
            # Save a topomap screenshot at a specific time point (e.g., 0.1 sec).
            out_path = os.path.join(out_dir, f"topomap_{band}_{method}.png")
            try:
                save_source_estimate_topomap(stc, subjects_dir, subject, out_path, time_point=0.1)
                print(f"Saved {method} topomap for {band} to {out_path}")
            except Exception as e:
                print(f"Error saving {method} topomap for {band}: {e}")
