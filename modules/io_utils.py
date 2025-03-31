import os
import re
import mne
from mne.io.constants import FIFF  # Import constants

def find_subject_edf_files(directory):
    """
    Scan the directory for .edf files and group them by subject based on filename.
    The filename should contain a subject identifier followed by "eo" or "ec" (case-insensitive).
    For example: "c1eo.edf", "c1ec.edf", "e1eo.edf", "e1ec.edf".
    
    Parameters:
        directory (str): Path to search for EDF files.
        
    Returns:
        dict: A dictionary structured as:
              { subject_id: {"EO": filename or None, "EC": filename or None}, ... }
    """
    edf_files = [f for f in os.listdir(directory) if f.lower().endswith('.edf')]
    subjects = {}
    pattern = re.compile(r'([a-zA-Z0-9]+)(eo|ec)', re.IGNORECASE)
    for f in edf_files:
        match = pattern.search(f)
        if match:
            subject_id = match.group(1).lower()  # e.g., "c1"
            condition = match.group(2).upper()     # "EO" or "EC"
            if subject_id not in subjects:
                subjects[subject_id] = {"EO": None, "EC": None}
            subjects[subject_id][condition] = f
        else:
            print(f"File {f} did not match expected naming convention.")
    return subjects

def load_eeg_data(edf_file, use_csd=False):
    """
    Load an EDF file, remove "-LE" suffix from channel names, fix channel unit issues,
    force all channels to EEG, apply the standard 10-20 montage, set an average reference projection,
    and then optionally compute current source density (CSD).
    
    Parameters:
        edf_file (str): Path to the EDF file.
        use_csd (bool): If True, attempt to compute current source density (CSD) on the raw data.
                        If CSD computation fails, it falls back to the standard pipeline.
        
    Returns:
        mne.io.Raw: The loaded and preprocessed Raw object.
    """
    raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
    raw.rename_channels({ch: ch.replace("-LE", "") for ch in raw.ch_names})
    
    # Fix channel unit issues (e.g., unit 117 (FIFF_UNIT_V_M2) -> FIFF_UNIT_V)
    for ch in raw.info["chs"]:
        if ch["unit"] == FIFF.FIFF_UNIT_V_M2:
            ch["unit"] = FIFF.FIFF_UNIT_V
            ch["unit_mul"] = 0

    # Force all channels to be of type 'eeg'
    raw.set_channel_types({ch: 'eeg' for ch in raw.ch_names})
    
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, match_case=False)
    
    # Set average reference as a projection.
    raw.set_eeg_reference("average", projection=True)
    raw.apply_proj()
    
    if use_csd:
        try:
            raw = mne.preprocessing.compute_current_source_density(raw)
            print("CSD transform applied successfully.")
        except Exception as e:
            print("CSD computation failed:", e)
            print("Falling back to standard preprocessing.")
        # After CSD, restore EEG channel types and fix unit issues again:
        for ch in raw.info["chs"]:
            if ch["unit"] == FIFF.FIFF_UNIT_V_M2:
                ch["unit"] = FIFF.FIFF_UNIT_V
                ch["unit_mul"] = 0
        raw.set_channel_types({ch: 'eeg' for ch in raw.ch_names})
        print("Restored channel types and units to EEG after CSD transform.")
    
    return raw

# Example usage:
if __name__ == "__main__":
    directory = os.getcwd()  # or specify a directory
    subjects_edf = find_subject_edf_files(directory)
    print("Grouped EDF files by subject:")
    for subj, files in subjects_edf.items():
        print(f"Subject {subj}: EO -> {files['EO']}, EC -> {files['EC']}")
    
    # To load data for a particular subject (e.g., "c1"):
    subject_id = "c1"  # adjust as needed
    if subject_id in subjects_edf:
        subj_files = subjects_edf[subject_id]
        # Use the EC file for both conditions if one is missing:
        if subj_files["EO"] is None and subj_files["EC"] is not None:
            subj_files["EO"] = subj_files["EC"]
        elif subj_files["EC"] is None and subj_files["EO"] is not None:
            subj_files["EC"] = subj_files["EO"]
        raw_eo = load_eeg_data(os.path.join(directory, subj_files["EO"]), use_csd=True)
        raw_ec = load_eeg_data(os.path.join(directory, subj_files["EC"]), use_csd=True)
        print("Loaded data for subject", subject_id)
    else:
        print("No EDF files found for subject", subject_id)
