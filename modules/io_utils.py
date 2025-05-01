import os
import re
import mne
from mne.io.constants import FIFF
from pathlib import Path
import numpy as np
from .montage_tools import inject_ch_pos  # Import from new module

CRITICAL_SITES = {"F3", "F4", "CZ", "PZ", "O1", "O2", "T7", "T8", "FZ"}


def clean_channel_name_dynamic(ch: str) -> str:
    ch = ch.upper()
    ch = re.sub(r"^EEG\s*", "", ch)
    ch = re.sub(r"[-._]?(LE|RE|AVG|M1|M2|A1|A2|REF|AV|CZREF|LINKED|AVERAGE)$", "", ch)
    return ch.strip()


def try_alternative_montages(raw):
    montages = ["biosemi64", "biosemi128", "GSN-HydroCel-129", "standard_alphabetic"]
    for m_name in montages:
        try:
            m = mne.channels.make_standard_montage(m_name)
            raw.set_montage(m, match_case=False, on_missing="warn")
            print(f"✅ Fallback montage applied: {m_name}")
            return m
        except Exception:
            continue
    return None


def inject_metadata_positions(raw):
    annotations = getattr(raw, "annotations", None)
    if annotations:
        for desc in annotations.description:
            match = re.match(r"ch_pos:\\s*([A-Z0-9]+),\\s*([-.\\d]+),\\s*([-.\\d]+),\\s*([-.\\d]+)", desc)
            if match:
                ch, x, y, z = match.groups()
                ch = ch.strip().upper()
                idx = raw.ch_names.index(ch) if ch in raw.ch_names else -1
                if idx >= 0:
                    raw.info["chs"][idx]["loc"][:3] = [float(x), float(y), float(z)]
                    print(f"📍 Injected loc from metadata: {ch} -> ({x}, {y}, {z})")


def remove_invalid_channels(raw, tol=0.001):
    ch_names = raw.ch_names
    pos = np.array([ch["loc"][:3] for ch in raw.info["chs"]])
    valid_idx = []
    seen_pos = set()

    for i, (p, name) in enumerate(zip(pos, ch_names)):
        if not np.all(np.isfinite(p)) or np.all(p == 0):
            print(f"⚠️ Dropping {name}: Non-finite or zero position {p}")
            continue
        pos_tuple = tuple(p.round(3))
        if pos_tuple in seen_pos:
            print(f"⚠️ {name} overlaps at {p}; keeping both for now")
        seen_pos.add(pos_tuple)
        valid_idx.append(i)

    if not valid_idx:
        print("⚠️ No channels with finite positions; keeping all channels to avoid empty dataset.")
        return raw  # Don't drop any channels if none are valid

    if len(valid_idx) < len(ch_names):
        valid_channels = [ch_names[i] for i in valid_idx]
        raw.pick(valid_channels)  # Use modern pick method
        print(f"✅ Kept {len(valid_idx)}/{len(ch_names)} channels after removing invalid positions.")
    return raw


def load_eeg_data(file_path: str, lpf_freq: float | None = None, notch_freq: float | None = None) -> mne.io.Raw:
    """Loads and preprocesses a single EDF file.

    Args:
        file_path (str): Path to the EDF file.
        lpf_freq (float | None, optional): Low-pass filter frequency. Defaults to None.
        notch_freq (float | None, optional): Notch filter frequency. Defaults to None.

    Returns:
        mne.io.Raw: Preprocessed MNE Raw object.
    """
    try:
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

        # --- Basic Filtering --- #
        # 1. High-pass filter (always applied)
        raw.filter(l_freq=1, h_freq=None, verbose=False)
        print(f"Applied 1 Hz high-pass filter to {Path(file_path).name}")

        # 2. Optional Notch filter
        if notch_freq:
            try:
                raw.notch_filter(freqs=notch_freq, verbose=False)
                print(f"Applied {notch_freq} Hz notch filter (and harmonics) to {Path(file_path).name}")
            except Exception as e_notch:
                 print(f"⚠️ Could not apply notch filter ({notch_freq} Hz): {e_notch}")

        # 3. Optional Low-pass filter
        if lpf_freq:
            try:
                raw.filter(l_freq=None, h_freq=lpf_freq, verbose=False)
                print(f"Applied {lpf_freq} Hz low-pass filter to {Path(file_path).name}")
            except Exception as e_lpf:
                 print(f"⚠️ Could not apply low-pass filter ({lpf_freq} Hz): {e_lpf}")

        # --- Referencing --- #
        try:
            raw.set_eeg_reference("average", projection=True, verbose=False)
            print(f"Applied common average reference to {Path(file_path).name}")
        except Exception as e_ref:
             print(f"⚠️ Could not apply average reference: {e_ref}")

        # --- Channel Handling --- #
        # Standardize channel names (strip 'EEG ' and common suffixes)
        rename_dict = {}
        for ch in raw.ch_names:
            ch_clean = clean_channel_name_dynamic(ch)
            if ch_clean and ch_clean != ch:
                rename_dict[ch] = ch_clean
        if rename_dict:
            for old, new in rename_dict.items():
                print(f"Renamed channel '{old}' -> '{new}'")
            raw.rename_channels(rename_dict)
        # Inject channel positions from montage_tools if available
        try:
            inject_ch_pos(raw)
            print("Applied channel-position injection via montage_tools.")
        except Exception:
            pass

        # Inject default positions for specific channels if missing
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
        # First, check the current positions in raw
        for ch in raw.ch_names:
            ch_upper = ch.upper()
            if ch_upper in default_pos:
                idx = raw.ch_names.index(ch)
                current_pos = raw.info["chs"][idx]["loc"][:3]
                # Inject position if current position is zero or nan
                if np.all(current_pos == 0) or np.all(np.isnan(current_pos)):
                    raw.info["chs"][idx]["loc"][:3] = default_pos[ch_upper]
                    print(f"Injected position for {ch}: {default_pos[ch_upper]}")

        # Try to set a standard 10-20 montage and inject positions for all channels
        try:
            montage = mne.channels.make_standard_montage("standard_1020")
            raw.set_montage(montage, match_case=False, on_missing="warn")
            # Inject custom positions from montage
            for ch_info in raw.info['chs']:
                name = ch_info['ch_name'].upper()
                if name in montage.get_positions()['ch_pos']:
                    ch_info['loc'][:3] = montage.get_positions()['ch_pos'][name]
        except Exception as e:
            print(f"Failed to set standard_1020 montage: {e}")
            try_alternative_montages(raw)
        # Apply any user-defined position injections (e.g., from metadata)
        try:
            inject_metadata_positions(raw)
        except Exception:
            pass
        # Drop channels with invalid/no positions
        raw = remove_invalid_channels(raw)

        print(f"\n=== Channels and Positions Before Filtering for {Path(file_path).name} ===")
        for ch in raw.info["chs"]:
            print(f"{ch['ch_name']}: {ch['loc'][:3]}")

        raw.filter(l_freq=1, h_freq=None, verbose=False)
        print(f"Applied 1 Hz high-pass filter to {Path(file_path).name}")

        print(f"\n=== Channels and Positions After Filtering for {Path(file_path).name} ===")
        for ch in raw.info["chs"]:
            print(f"{ch['ch_name']}: {ch['loc'][:3]}")

        return raw

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        raise


# --- Utility: Group EDF Files by Subject ---
def find_subject_edf_files(directory: str) -> dict:
    """
    Finds top-level EDF files and groups them by subject ID and EO/EC condition.
    Handles case-insensitive EO/EC designations and provides robust naming logic.

    Args:
        directory (str): Path to directory containing EDF files.

    Returns:
        dict: {subject_id: {'EO': str|None, 'EC': str|None}, ...}
              Only includes subjects with at least one valid EO or EC file.
    """
    # List only .edf files (case-insensitive) directly under the input directory
    edf_files = [f for f in os.listdir(directory) if f.lower().endswith('.edf')]
    subjects: dict[str, dict[str, str | None]] = {}
    condition_patterns = [
        (r'\beo\b', 'EO'),  # Matches 'eo' (case-insensitive) as a word
        (r'\bec\b', 'EC'),  # Matches 'ec' (case-insensitive) as a word
    ]

    for fname in edf_files:
        stem = Path(fname).stem
        low_stem = stem.lower().strip()

        # Extract subject ID (digits or alphanumeric prefix)
        m_id = re.match(r'^(\d+)', low_stem) or re.match(r'^([a-z0-9]+)', low_stem)
        if not m_id:
            print(f"⚠️ Cannot extract subject ID from '{fname}'; skipping.")
            continue
        sid = m_id.group(1).upper()
        subjects.setdefault(sid, {'EO': None, 'EC': None})

        # Check for EO/EC conditions (case-insensitive)
        matched = False
        for pattern, condition in condition_patterns:
            if re.search(pattern, low_stem, re.IGNORECASE):
                if subjects[sid][condition] is None:
                    subjects[sid][condition] = fname
                    print(f"Assigned '{fname}' as {condition} for subject {sid}")
                else:
                    print(f"⚠️ Multiple {condition} files for {sid}; keeping first: {subjects[sid][condition]}")
                matched = True
                break

        # Future-proof fallback: handle ambiguous or misnamed files
        if not matched:
            # Check for partial matches or alternative naming (e.g., 'eyesopen', 'eyesclosed')
            if 'eyesopen' in low_stem or 'open' in low_stem:
                condition = 'EO'
            elif 'eyesclosed' in low_stem or 'closed' in low_stem:
                condition = 'EC'
            else:
                print(f"⚠️ '{fname}' lacks clear EO/EC designation; skipping.")
                continue

            if subjects[sid][condition] is None:
                subjects[sid][condition] = fname
                print(f"Assigned '{fname}' as {condition} (fallback) for subject {sid}")
            else:
                print(f"⚠️ Multiple {condition} files for {sid}; keeping first: {subjects[sid][condition]}")

    # Filter subjects with at least one valid file
    valid_subjects = {
        sid: rec for sid, rec in subjects.items()
        if rec.get('EO') or rec.get('EC')
    }
    print(f"Found {len(valid_subjects)} subjects with valid EDF files")
    return valid_subjects


def setup_output_directories(project_dir: str, subject: str) -> dict:
    """Creates the necessary output directory structure for a subject.

    Generates a standard set of subdirectories within 'output/[subject_id]'
    for storing plots, reports, and data.

    Args:
        project_dir (str): The root directory of the project.
        subject (str): The subject identifier.

    Returns:
        dict: A dictionary mapping descriptive keys (e.g., 'subject', 'plots',
              'plots_topo', 'reports', 'data') to their absolute paths (str).
              Creates directories if they don't exist.
    """
    subject_folder = Path(project_dir) / "output" / subject
    plots_folder = subject_folder / "plots"
    reports_folder = subject_folder / "reports"
    data_folder = subject_folder / "data" # For CSVs, intermediate results

    folders = {
        "subject": subject_folder,
        "plots": plots_folder,
        "reports": reports_folder,
        "data": data_folder,
        # Add subfolders for specific plot types
        "plots_topo": plots_folder / "topomaps",
        "plots_wave": plots_folder / "waveforms",
        "plots_erp": plots_folder / "erp",
        "plots_coh": plots_folder / "coherence",
        "plots_zscore": plots_folder / "zscores",
        "plots_var": plots_folder / "variance",
        "plots_tfr": plots_folder / "tfr",
        "plots_ica": plots_folder / "ica",
        "plots_src": plots_folder / "source_localization",
        "plots_vigilance": plots_folder / "vigilance",
        "plots_site": plots_folder / "site_reports",
    }

    # Create all directories, including subdirectories
    for path in folders.values():
        # Ensure path is a Path object before calling mkdir
        if isinstance(path, str):
            path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

    print(f"📂 Set up output directories for subject {subject} in {subject_folder}")
    # Return paths as strings for compatibility if needed elsewhere
    return {key: str(path) for key, path in folders.items()}
