"""
montage_tools.py - Utilities for EEG montage management in The Squiggle Interpreter

Provides:
- Channel listings for standard montages
- Tools to inject 3D electrode positions
- Validation for CSD and source localization readiness
"""

import mne
import numpy as np
from mne.io.constants import FIFF


def get_channels_for_montage(montage_name: str) -> list[str]:
    """
    Returns a list of channel names for a given montage.

    Args:
        montage_name (str): Name of the montage (e.g., 'standard_1020', 'biosemi64')

    Returns:
        list[str]: List of channel names included in the montage

    Raises:
        ValueError: If montage_name is not recognized
    """
    try:
        montage = mne.channels.make_standard_montage(montage_name)
        channels = list(montage.get_positions()["ch_pos"].keys())
        return channels
    except ValueError:
        print(
            f"⚠️ Montage '{montage_name}' not found in MNE. Available montages: {mne.channels.get_builtin_montages()}")
        raise ValueError(f"Unknown montage: {montage_name}")


def print_montage_table():
    """Prints a table of common montages and their partial channel lists."""
    montages = {
        "standard_1020": "Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, Fz, Cz, Pz, T7, T8, P7, P8",
        "standard_1005": "AF3, AF4, FC1–FC6, CP1–CP6, PO3–PO4 + 1020",
        "biosemi64": "A1, A2, POz, TP7, PO7, CPz, FCz + 64-site grid",
        "GSN-HydroCel-129": "E1–E129 (dense geodesic grid)",
        "easycap-M10": "F9, F10, FT9, FT10 + 61 EasyCap sites",
        "egi-128": "E1–E128 (EGI geodesic net)"
    }
    print("\n=== Montage Coverage Table ===")
    print(f"{'Montage':<15} {'Channels (Partial)':<40} {'Typical Use':<30}")
    print("-" * 85)
    for name, chans in montages.items():
        use = {
            "standard_1020": "Clinical EEG, qEEG (21/10–20 cap)",
            "standard_1005": "High-res qEEG, source analysis",
            "biosemi64": "BioSemi research setups",
            "GSN-HydroCel-129": "sLORETA, source modeling",
            "easycap-M10": "EEG-TMS integration",
            "egi-128": "EGI geodesic systems"
        }[name]
        print(f"{name:<15} {chans:<40} {use:<30}")


def inject_ch_pos(raw: mne.io.Raw, positions_dict: dict[str, tuple[float, float, float]]) -> mne.io.Raw:
    """
    Injects 3D coordinates into the Raw object for specified channels.

    Args:
        raw (mne.io.Raw): The raw EEG data object
        positions_dict (dict): Mapping of channel names to (x, y, z) coordinates

    Returns:
        mne.io.Raw: Updated Raw object with injected positions
    """
    for ch_name, pos in positions_dict.items():
        if ch_name in raw.ch_names:
            idx = raw.ch_names.index(ch_name)
            raw.info["chs"][idx]["loc"][:3] = pos
            print(f"📍 Injected position for {ch_name}: {pos}")
        else:
            print(f"⚠️ Channel {ch_name} not found in raw data; skipping position injection.")
    return raw


def validate_montage_for_csd_loreta(raw):
    status = {
        "csd_ready": True,
        "loreta_ready": True,
        "missing_channels": [],
        "overlapping_positions": [],
        "has_dig_points": False,
        "has_fiducials": False,
        "has_valid_positions": False,
    }

    # Check for digitization points
    dig = raw.info.get("dig", []) or []  # Ensure dig is a list, even if None
    status["has_dig_points"] = any(d["kind"] in (FIFF.FIFFV_POINT_EEG, FIFF.FIFFV_POINT_EXTRA) for d in dig)
    status["has_fiducials"] = any(d["kind"] == FIFF.FIFFV_POINT_CARDINAL for d in dig)

    # Check channel positions
    ch_pos = [ch["loc"][:3] for ch in raw.info["chs"]]
    ch_names = raw.ch_names

    # Check for missing or invalid positions
    invalid_pos = []
    for i, (pos, name) in enumerate(zip(ch_pos, ch_names)):
        if not np.all(np.isfinite(pos)) or np.all(pos == 0):
            invalid_pos.append(name)
    status["missing_channels"] = invalid_pos
    if invalid_pos:
        status["csd_ready"] = False
        status["loreta_ready"] = False

    # Check for overlapping positions
    seen_pos = set()
    overlapping = []
    for i, (pos, name) in enumerate(zip(ch_pos, ch_names)):
        if name in invalid_pos:
            continue
        pos_tuple = tuple(pos.round(3))
        if pos_tuple in seen_pos:
            overlapping.append(name)
        seen_pos.add(pos_tuple)
    status["overlapping_positions"] = overlapping
    if overlapping:
        status["csd_ready"] = False

    # Check if there are enough valid positions for CSD/LORETA
    valid_pos_count = sum(1 for pos in ch_pos if np.all(np.isfinite(pos)) and not np.all(pos == 0))
    status["has_valid_positions"] = valid_pos_count >= 4  # Minimum for CSD/LORETA
    if valid_pos_count < 4:
        status["csd_ready"] = False
        status["loreta_ready"] = False

    return status["csd_ready"] and status["loreta_ready"], status


if __name__ == "__main__":
    # Example usage
    print_montage_table()
