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
            match = re.match(r"ch_pos:\s*([A-Z0-9]+),\s*([-.\d]+),\s*([-.\d]+),\s*([-.\d]+)", desc)
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
        return raw  # Don’t drop any channels if none are valid

    if len(valid_idx) < len(ch_names):
        valid_channels = [ch_names[i] for i in valid_idx]
        raw.pick(valid_channels)  # Use modern pick method
        print(f"✅ Kept {len(valid_idx)}/{len(ch_names)} channels after removing invalid positions.")
    return raw


def load_eeg_data(file_path: str, use_csd: bool = False) -> mne.io.Raw:
    try:
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        raw.set_eeg_reference("average", projection=True, verbose=False)
        raw.filter(l_freq=1, h_freq=None, verbose=False)
        print(f"Applied 1 Hz high-pass filter to {Path(file_path).name}")

        # Standardize channel names
        ch_names = raw.ch_names
        rename_dict = {}
        for ch in ch_names:
            ch_clean = clean_channel_name_dynamic(ch)
            if ch_clean != ch:
                rename_dict[ch] = ch_clean
        if rename_dict:
            for old_name, new_name in rename_dict.items():
                print(f"Renamed legacy channel {old_name} -> {new_name}")
            raw.rename_channels(rename_dict)

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

        # Try to set a standard montage to fill in any remaining positions
        try:
            montage = mne.channels.make_standard_montage("standard_1020")
            raw.set_montage(montage, match_case=False, on_missing="warn")
        except Exception as e:
            print(f"Failed to set standard_1020 montage: {e}")
            # Optionally try alternative montages if needed
            try_alternative_montages(raw)

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
