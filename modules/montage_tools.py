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


def validate_montage_for_csd_loreta(raw: mne.io.Raw, min_channels: int = 16, tol: float = 0.001) -> tuple[bool, dict]:
    """
    Validates if the montage is suitable for CSD and LORETA analysis.

    Args:
        raw (mne.io.Raw): The raw EEG data object
        min_channels (int): Minimum number of channels required (default: 16)
        tol (float): Tolerance for detecting overlapping positions (default: 0.001)

    Returns:
        tuple[bool, dict]: (is_valid, status_dict)
        - is_valid: True if montage is ready for CSD/LORETA
        - status_dict: Details on digitization, positions, and issues
    """
    status = {
        "has_dig_points": False,
        "has_unique_positions": True,
        "valid_channels": 0,
        "missing_channels": [],
        "overlapping_positions": [],
        "csd_ready": False,
        "loreta_ready": False
    }

    # Check digitization points
    dig = raw.info.get("dig", [])
    status["has_dig_points"] = any(d["kind"] in (FIFF.FIFFV_POINT_EEG, FIFF.FIFFV_POINT_EXTRA) for d in dig)

    # Check channel positions
    pos = np.array([ch["loc"][:3] for ch in raw.info["chs"]])
    ch_names = raw.ch_names
    valid_idx = []
    seen_pos = set()

    for i, (p, name) in enumerate(zip(pos, ch_names)):
        if not np.all(np.isfinite(p)) or np.all(p == 0):
            status["missing_channels"].append(f"{name}: Invalid/zero position {p}")
            continue
        pos_tuple = tuple(p.round(int(-np.log10(tol))))
        if pos_tuple in seen_pos:
            status["overlapping_positions"].append(f"{name} overlaps at {p}")
            status["has_unique_positions"] = False
        else:
            seen_pos.add(pos_tuple)
            valid_idx.append(i)

    status["valid_channels"] = len(valid_idx)

    # CSD readiness: Need digitization and enough valid channels
    status["csd_ready"] = status["has_dig_points"] and status["valid_channels"] >= min_channels

    # LORETA readiness: Need unique positions and enough channels
    status["loreta_ready"] = status["has_unique_positions"] and status["valid_channels"] >= min_channels

    is_valid = status["csd_ready"] and status["loreta_ready"]

    if not is_valid:
        print(f"⚠️ Montage validation failed for {raw.info['meas_date']}:")
        for key, value in status.items():
            if key in ["missing_channels", "overlapping_positions"] and value:
                print(f"  {key}: {value}")
            elif key not in ["csd_ready", "loreta_ready"]:
                print(f"  {key}: {value}")

    return is_valid, status


if __name__ == "__main__":
    # Example usage
    print_montage_table()