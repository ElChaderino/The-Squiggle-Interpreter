# modules/io_utils.py
import os
import re
import mne
import numpy as np
import pandas as pd
import logging
import subprocess
import sys
from pathlib import Path
from .montage_tools import inject_ch_pos
from .config import CRITICAL_SITES, OUTPUT_FOLDERS


logger = logging.getLogger(__name__)

def find_subject_edf_files(project_dir: str) -> dict[str, dict[str, Path | None]]:
    """
    Find and group EDF files by subject in the project directory.
    
    Args:
        project_dir (str): Path to the project directory.
    
    Returns:
        dict: Mapping of subject IDs to {"EO": Path, "EC": Path} dictionaries.
    """
    project_dir = Path(project_dir)
    subject_edf_groups = {}
    edf_files = list(project_dir.rglob("*.edf"))
    missing_log = project_dir / "missing_channels_log.txt"

    for file in edf_files:
        name = file.name.lower()
        # Extract subject ID from filename (e.g., "subjectID_EO.edf" or "subjectID_EC.edf")
        match = re.match(r"^(.*?)(?:_eo|_ec)\.edf$", name, re.IGNORECASE)
        if match:
            subject_id = match.group(1)
            if subject_id not in subject_edf_groups:
                subject_edf_groups[subject_id] = {"EO": None, "EC": None}
            if "_eo" in name:
                subject_edf_groups[subject_id]["EO"] = file
            elif "_ec" in name:
                subject_edf_groups[subject_id]["EC"] = file

    # Log subjects with missing EO or EC files
    for subject_id, files in subject_edf_groups.items():
        if not files["EO"] or not files["EC"]:
            with open(missing_log, "a") as f:
                if not files["EO"]:
                    f.write(f"Subject {subject_id}: Missing EO EDF file\n")
                    logger.warning(f"Subject {subject_id}: Missing EO EDF file")
                if not files["EC"]:
                    f.write(f"Subject {subject_id}: Missing EC EDF file\n")
                    logger.warning(f"Subject {subject_id}: Missing EC EDF file")

    logger.info(f"Found EDF files for subjects: {list(subject_edf_groups.keys())}")
    return subject_edf_groups

def setup_output_directories(project_dir: str, subject: str) -> tuple[str, dict, Path]:
    """
    Set up output directories for a subject and return the folder paths.
    
    Args:
        project_dir (str): Root project directory.
        subject (str): Subject identifier.
    
    Returns:
        tuple: (root_dir, folders, subject_folder)
            - root_dir (str): Root output directory.
            - folders (dict): Dictionary of folder paths for different output types.
            - subject_folder (Path): Subject-specific output directory.
    """
    root_dir = os.path.join(project_dir, "outputs")
    subject_folder = Path(root_dir) / subject
    subject_folder.mkdir(parents=True, exist_ok=True)

    # Define subdirectories using OUTPUT_FOLDERS
    folders = {}
    for key, subdir in OUTPUT_FOLDERS.items():
        folder_path = subject_folder / subdir
        folder_path.mkdir(parents=True, exist_ok=True)
        folders[key] = str(folder_path)

    # Log the folders dictionary for debugging
    logger.info(f"Setup output directories for {subject}: {folders}")
    
    return root_dir, folders, subject_folder

def clean_channel_name_dynamic(ch: str) -> str:
    """
    Standardize EEG channel names by removing common prefixes and suffixes.
    
    Args:
        ch (str): Original channel name.
    
    Returns:
        str: Cleaned channel name.
    """
    ch = ch.upper()
    ch = re.sub(r"^EEG\s*", "", ch)
    ch = re.sub(r"[-._]?(LE|RE|AVG|M1|M2|A1|A2|REF|AV|CZREF|LINKED|AVERAGE)$", "", ch)
    return ch.strip()

def try_alternative_montages(raw):
    """
    Attempt to apply alternative montages if standard_1020 fails.
    
    Args:
        raw (mne.io.Raw): Raw EEG data.
    
    Returns:
        mne.channels.Montage: Applied montage, or None if all fail.
    """
    montages = ["biosemi64", "biosemi128", "GSN-HydroCel-129", "standard_alphabetic"]
    for m_name in montages:
        try:
            m = mne.channels.make_standard_montage(m_name)
            raw.set_montage(m, match_case=False, on_missing="warn")
            logger.info(f"Fallback montage applied: {m_name}")
            return m
        except Exception:
            continue
    logger.warning("All fallback montages failed.")
    return None

def inject_metadata_positions(raw):
    """
    Inject channel positions from annotations if available.
    
    Args:
        raw (mne.io.Raw): Raw EEG data.
    """
    annotations = getattr(raw, "annotations", None)
    if annotations:
        for desc in annotations.description:
            match = re.match(r"ch_pos:\s*([A-Z0-9]+),\s*([-.\d]+),\s*([-.\d]+),\s*([-.\d]+)", desc)
            if match:
                ch, x, y, z = match.groups()
                ch = ch.strip().upper()
                idx = raw.ch_names.index(ch) if ch in raw.ch_names else -1
                if idx >= 0:
                    raw.info["chs"][idx]["loc"][:3] = [float(x), float(y), float(z)]
                    logger.info(f"Injected loc from metadata: {ch} -> ({x}, {y}, {z})")

def setup_output_directories(project_dir, subject):
    """
    Set up output directories for a subject.
    
    Args:
        project_dir (str): Project directory path.
        subject (str): Subject identifier.
    
    Returns:
        tuple: (overall_output_dir, folders, subject_folder)
    """
    overall_output_dir = os.path.join(project_dir, "outputs")
    os.makedirs(overall_output_dir, exist_ok=True)
    subject_folder = os.path.join(overall_output_dir, subject)
    os.makedirs(subject_folder, exist_ok=True)
    folders = {
        key: os.path.join(subject_folder, path) for key, path in OUTPUT_FOLDERS.items()
    }
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    return overall_output_dir, folders, subject_folder

def load_clinical_outcomes(csv_file, n_channels):
    """
    Load clinical outcomes from a CSV file.
    
    Args:
        csv_file (str): Path to CSV file.
        n_channels (int): Number of channels.
    
    Returns:
        np.ndarray: Outcome values.
    """
    try:
        df = pd.read_csv(csv_file)
        outcomes = df['outcome'].values
        if len(outcomes) < n_channels:
            outcomes = np.pad(outcomes, (0, n_channels - len(outcomes)), mode='constant')
        return outcomes[:n_channels]
    except Exception as e:
        logger.error(f"Could not load clinical outcomes from CSV: {e}")
        return np.random.rand(n_channels)

def load_and_preprocess_data(project_dir, files, use_csd):
    """
    Load and preprocess EO and EC EDF files.
    
    Args:
        project_dir (str): Project directory path.
        files (dict): Dictionary with 'EO' and 'EC' file paths.
        use_csd (bool): Whether to apply CSD transform.
    
    Returns:
        tuple: (raw_eo, raw_ec, raw_eo_csd, raw_ec_csd)
    """
    eo_file = files.get("EO", files.get("EC"))
    ec_file = files.get("EC", files.get("EO"))
    logger.info(f"EO file: {eo_file}, EC file: {ec_file}")

    if not eo_file and not ec_file:
        logger.warning("No EO or EC files available for processing.")
        return None, None, None, None

    raw_eo = load_eeg_data(os.path.join(project_dir, eo_file), use_csd=False) if eo_file else None
    raw_ec = load_eeg_data(os.path.join(project_dir, ec_file), use_csd=False) if ec_file else None

    if raw_eo is None and raw_ec is None:
        logger.warning("Failed to load both EO and EC data.")
        return None, None, None, None

    raw_eo_csd = None
    raw_ec_csd = None
    if use_csd:
        if raw_eo:
            raw_eo_csd = raw_eo.copy().load_data()
            try:
                raw_eo_csd = mne.preprocessing.compute_current_source_density(raw_eo_csd)
                logger.info("CSD applied for graphs (EO).")
            except Exception as e:
                logger.error(f"CSD for graphs (EO) failed: {e}")
                raw_eo_csd = raw_eo
        if raw_ec:
            raw_ec_csd = raw_ec.copy().load_data()
            try:
                raw_ec_csd = mne.preprocessing.compute_current_source_density(raw_ec_csd)
                logger.info("CSD applied for graphs (EC).")
            except Exception as e:
                logger.error(f"CSD for graphs (EC) failed: {e}")
                raw_ec_csd = raw_ec
    else:
        raw_eo_csd = raw_eo
        raw_ec_csd = raw_ec

    return raw_eo, raw_ec, raw_eo_csd, raw_ec_csd

def run_extension_scripts(project_dir, subject):
    """
    Run extension scripts for a subject.
    
    Args:
        project_dir (str): Project directory path.
        subject (str): Subject identifier.
    
    Returns:
        None
    """
    extension_script = os.path.join(project_dir, "extensions", "EEG_Extension.py")
    if os.path.exists(extension_script):
        logger.info(f"Subject {subject}: Running extension script: {extension_script}")
        subprocess.run([sys.executable, extension_script])
    else:
        logger.info(f"Subject {subject}: No extension script found in 'extensions' folder.")

def remove_invalid_channels(raw, tol=0.001):
    """
    Remove channels with invalid or overlapping positions.
    
    Args:
        raw (mne.io.Raw): Raw EEG data.
        tol (float): Tolerance for position comparison.
    
    Returns:
        mne.io.Raw: Raw data with invalid channels removed.
    """
    ch_names = raw.ch_names
    pos = np.array([ch["loc"][:3] for ch in raw.info["chs"]])
    valid_idx = []
    seen_pos = set()

    for i, (p, name) in enumerate(zip(pos, ch_names)):
        if not np.all(np.isfinite(p)) or np.all(p == 0):
            logger.warning(f"Dropping {name}: Non-finite or zero position {p}")
            continue
        pos_tuple = tuple(p.round(3))
        if pos_tuple in seen_pos:
            logger.warning(f"{name} overlaps at {p}; keeping both for now")
        seen_pos.add(pos_tuple)
        valid_idx.append(i)

    if not valid_idx:
        logger.warning("No channels with finite positions; keeping all channels to avoid empty dataset.")
        return raw

    if len(valid_idx) < len(ch_names):
        valid_channels = [ch_names[i] for i in valid_idx]
        raw.pick(valid_channels)
        logger.info(f"Kept {len(valid_idx)}/{len(ch_names)} channels after removing invalid positions.")
    return raw

def load_eeg_data(file_path: str, use_csd: bool = False) -> mne.io.Raw:
    """
    Load and preprocess an EDF file.
    
    Args:
        file_path (str): Path to EDF file.
        use_csd (bool): Whether to apply CSD transform.
    
    Returns:
        mne.io.Raw: Loaded and preprocessed raw data.
    """
    try:
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        raw.set_eeg_reference("average", projection=True, verbose=False)
        raw.filter(l_freq=1, h_freq=None, verbose=False)
        logger.info(f"Applied 1 Hz high-pass filter to {Path(file_path).name}")

        ch_names = raw.ch_names
        rename_dict = {ch: clean_channel_name_dynamic(ch) for ch in ch_names if clean_channel_name_dynamic(ch) != ch}
        if rename_dict:
            for old_name, new_name in rename_dict.items():
                logger.info(f"Renamed legacy channel {old_name} -> {new_name}")
            raw.rename_channels(rename_dict)

        default_pos = {
            "T7": [-0.071, 0.008, 0.038],
            "T8": [0.071, 0.008, 0.038],
            "P7": [-0.055, -0.055, 0.038],
            "P8": [0.055, -0.055, 0.038],
            "F3": [-0.038, 0.071, 0.038],
            "F4": [0.038, 0.071, 0.038],
            "CZ": [0.0, 0.0, 0.087],
            "O1": [-0.038, -0.071, 0.038],
            "O2": [0.038, -0.071, 0.038],
            "FZ": [0.0, 0.071, 0.055],
            "PZ": [0.0, -0.055, 0.071],
        }
        for ch in raw.ch_names:
            ch_upper = ch.upper()
            if ch_upper in default_pos:
                idx = raw.ch_names.index(ch)
                current_pos = raw.info["chs"][idx]["loc"][:3]
                if np.all(current_pos == 0) or np.all(np.isnan(current_pos)):
                    raw.info["chs"][idx]["loc"][:3] = default_pos[ch_upper]
                    logger.info(f"Injected position for {ch}: {default_pos[ch_upper]}")

        try:
            montage = mne.channels.make_standard_montage("standard_1020")
            raw.set_montage(montage, match_case=False, on_missing="warn")
        except Exception as e:
            logger.warning(f"Failed to set standard_1020 montage: {e}")
            try_alternative_montages(raw)

        logger.debug(f"Channels and positions for {Path(file_path).name}: {[ch['ch_name'] + ': ' + str(ch['loc'][:3]) for ch in raw.info['chs']]}")
        return raw
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        raise
