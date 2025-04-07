import os
import re
import mne
from mne.io.constants import FIFF
from pathlib import Path
import numpy as np

CRITICAL_SITES = {"F3", "F4", "CZ", "PZ", "O1", "O2", "T3", "T4", "FZ"}

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
            raw.set_montage(m, match_case=False, on_missing="ignore")
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
    """Remove channels with zero, infinite, or overlapping positions."""
    ch_names = raw.ch_names
    pos = np.array([ch["loc"][:3] for ch in raw.info["chs"]])
    valid_idx = []
    seen_pos = set()

    for i, (p, name) in enumerate(zip(pos, ch_names)):
        # Check for zero, infinite, or NaN positions
        if not np.all(np.isfinite(p)) or np.all(p == 0):
            print(f"⚠️ Dropping {name}: Invalid position {p}")
            continue
        # Check for overlapping positions
        pos_tuple = tuple(p.round(3))  # Round to avoid floating-point noise
        if pos_tuple in seen_pos:
            print(f"⚠️ Dropping {name}: Overlaps with another channel at {p}")
            continue
        seen_pos.add(pos_tuple)
        valid_idx.append(i)

    if not valid_idx:
        raise ValueError("No channels with valid, unique positions remain.")
    if len(valid_idx) < len(ch_names):
        raw.pick(valid_idx)
        print(f"✅ Kept {len(valid_idx)}/{len(ch_names)} channels after removing invalid/overlapping positions.")
    return raw

def load_eeg_data(edf_path: str | Path, use_csd: bool = False, apply_filter: bool = True, strict_montage: bool = False) -> mne.io.Raw:
    edf_path = Path(edf_path)
    try:
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)

        # Normalize channel names
        raw.rename_channels({ch: clean_channel_name_dynamic(ch) for ch in raw.ch_names})

        # Standard montage
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, match_case=False, on_missing="ignore")

        # Inject fallback montage if locs missing
        montage_pos = montage.get_positions().get("ch_pos", {})
        missing_locs = []
        for ch in raw.info["chs"]:
            name = ch["ch_name"].upper()
            if name in montage_pos:
                ch["loc"][:3] = montage_pos[name]
            else:
                missing_locs.append(name)

        if missing_locs:
            fallback = try_alternative_montages(raw)
            if fallback:
                montage_pos = fallback.get_positions().get("ch_pos", {})
                for ch in raw.info["chs"]:
                    name = ch["ch_name"].upper()
                    if name in montage_pos:
                        ch["loc"][:3] = montage_pos[name]

        inject_metadata_positions(raw)

        # Remove channels with invalid or overlapping positions
        raw = remove_invalid_channels(raw)

        # Log missing locs after filtering
        missing_after = [ch["ch_name"] for ch in raw.info["chs"] if not ch["loc"][:3].any()]
        if missing_after:
            with open("missing_positions_log.txt", "a") as log:
                log.write(f"\n=== {edf_path.name} ===\n")
                log.write("Missing 3D electrode positions after filtering:\n")
                for ch in missing_after:
                    log.write(f"  - {ch}\n")

        # Abort if critical electrodes are missing and strict mode is on
        missing_critical = CRITICAL_SITES - set(raw.ch_names)
        if missing_critical:
            print(f"[!] Missing critical clinical electrodes: {missing_critical}")
            if strict_montage:
                raise RuntimeError(f"Missing critical electrodes: {missing_critical}")

        raw.set_eeg_reference("average", projection=True)

        if apply_filter:
            raw.filter(l_freq=1.0, h_freq=None, verbose=False)
            print(f"🔍 Applied 1 Hz high-pass filter to {edf_path.name}")

        if use_csd:
            dig = raw.info.get("dig", [])
            has_dig = any(d['kind'] in (FIFF.FIFFV_POINT_EEG, FIFF.FIFFV_POINT_EXTRA) for d in dig)
            if has_dig and not any(np.all(ch["loc"][:3] == 0) or not np.all(np.isfinite(ch["loc"][:3])) for ch in raw.info["chs"]):
                try:
                    raw = mne.preprocessing.compute_current_source_density(raw)
                    print(f"🧠 CSD applied to {edf_path.name}")
                except Exception as csd_error:
                    print(f"[!] CSD failed on {edf_path.name}: {csd_error}")
                    with open("missing_channels_log.txt", "a") as log:
                        log.write(f"\n=== {edf_path.name} ===\n")
                        log.write(f"CSD failed: {csd_error}\n")
            else:
                print(f"[!] Skipping CSD for {edf_path.name}: No digitization points or invalid positions remain.")
                with open("missing_channels_log.txt", "a") as log:
                    log.write(f"\n=== {edf_path.name} ===\n")
                    log.write("CSD skipped: Insufficient valid positions.\n")

        return raw

    except Exception as e:
        print(f"❌ Error loading {edf_path}: {e}")
        raise
