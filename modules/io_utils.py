import os
import re
import mne
from mne.io.constants import FIFF

def find_subject_edf_files(directory):
    # Find all EDF files in the directory and group them by subject based on naming conventions.
    edf_files = [f for f in os.listdir(directory) if f.lower().endswith('.edf')]
    subjects = {}
    pattern = re.compile(r'([a-zA-Z0-9]+)(eo|ec)', re.IGNORECASE)
    for f in edf_files:
        match = pattern.search(f)
        if match:
            subject_id = match.group(1).lower()
            condition = match.group(2).upper()
            if subject_id not in subjects:
                subjects[subject_id] = {"EO": None, "EC": None}
            subjects[subject_id][condition] = f
        else:
            print(f"File {f} did not match expected naming convention.")
    return subjects

def load_eeg_data(edf_file, use_csd=False, for_source=False):
    """
    Load an EDF file, remove "-LE" from channel names, fix channel unit issues,
    force all channels to EEG, apply the standard 10–20 montage, and re-reference the data.
    
    If for_source is True, it will permanently re-reference the data (projection=False)
    without adding any reference projection (needed for source localization).
    If for_source is False, it uses the default behavior (adds a reference projection) for other analyses.
    
    Parameters:
        edf_file (str): Path to the EDF file.
        use_csd (bool): If True, attempt to compute current source density (CSD) for graphing.
        for_source (bool): If True, do not add the reference projection (for source localization).
        
    Returns:
        mne.io.Raw: The loaded and preprocessed Raw object.
    """
    raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
    
    # Remove unwanted suffixes (e.g., "-LE")
    raw.rename_channels({ch: ch.replace("-LE", "") for ch in raw.ch_names})
    
    # Fix channel unit issues: change FIFF_UNIT_V_M2 (117) to FIFF_UNIT_V.
    for ch in raw.info["chs"]:
        if ch["unit"] == FIFF.FIFF_UNIT_V_M2:
            ch["unit"] = FIFF.FIFF_UNIT_V
            ch["unit_mul"] = 0

    # Force all channels to be of type 'eeg'
    raw.set_channel_types({ch: 'eeg' for ch in raw.ch_names})
    
    # Create the standard 10-20 montage.
    montage = mne.channels.make_standard_montage("standard_1020")
    
    # Try setting the montage.
    try:
        raw.set_montage(montage, match_case=False)
    except ValueError as e:
        print("Montage error:", e)
        # Identify channels not present in the montage.
        missing = [ch for ch in raw.ch_names if ch not in montage.ch_names]
        # Attempt to rename channels by removing a common prefix ("EEG ").
        new_names = {ch: ch.replace("EEG ", "") for ch in missing if ch.startswith("EEG ")}
        if new_names:
            print("Renaming channels to match montage:", new_names)
            raw.rename_channels(new_names)
        # Re-check which channels are still missing.
        missing = [ch for ch in raw.ch_names if ch not in montage.ch_names]
        if missing:
            print("Channels not found in montage after renaming, dropping them:", missing)
            raw.drop_channels(missing)
        # Now re-apply the montage.
        raw.set_montage(montage, match_case=False)

    # Set the EEG reference.
    if for_source:
        raw.set_eeg_reference("average", projection=False)
    else:
        raw.set_eeg_reference("average", projection=True)
        raw.apply_proj()
    
    # Optionally compute current source density (CSD) for graphing.
    if use_csd:
        try:
            raw = mne.preprocessing.compute_current_source_density(raw)
            print("CSD transform applied successfully.")
        except Exception as e:
            print("CSD computation failed:", e)
            print("Falling back to standard preprocessing.")
        raw.set_montage(montage, match_case=False)
        for ch in raw.info["chs"]:
            if ch["unit"] == FIFF.FIFF_UNIT_V_M2:
                ch["unit"] = FIFF.FIFF_UNIT_V
                ch["unit_mul"] = 0
        print("Restored channel info after CSD transform.")
    
    return raw
