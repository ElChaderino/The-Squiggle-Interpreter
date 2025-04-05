# feature_extraction.py

import numpy as np
from scipy.signal import welch
from .processing import compute_band_power, BANDS


def compute_paf(signal, sfreq, band=(7, 13)):
    f, Pxx = welch(signal, fs=sfreq, nperseg=sfreq*2)
    mask = (f >= band[0]) & (f <= band[1])
    return f[mask][np.argmax(Pxx[mask])]


def extract_classification_features(raw, vigilance_sequence=None, eyes_open_raw=None, eyes_closed_raw=None, csd_raw=None, sloreta_data=None):
    sfreq = raw.info['sfreq']
    ch_names = raw.ch_names
    data = raw.get_data() * 1e6

    def mean_band(band):
        return np.mean([compute_band_power(data[i], sfreq, band) for i in range(data.shape[0])])

    features = {
        'delta': mean_band(BANDS['Delta']),
        'theta': mean_band(BANDS['Theta']),
        'low_alpha': mean_band((8, 10)),
        'high_alpha': mean_band((10, 12)),
        'alpha_power': mean_band(BANDS['Alpha']),
        'smr': mean_band(BANDS['SMR']),
        'beta': mean_band(BANDS['Beta']),
        'hi_beta': mean_band(BANDS['HighBeta'])
    }

    features['theta_beta'] = features['theta'] / features['beta'] if features['beta'] else 0
    features['hi_alpha_to_lo_alpha'] = features['high_alpha'] / features['low_alpha'] if features['low_alpha'] else 0
    features['hi_beta_to_beta'] = features['hi_beta'] / features['beta'] if features['beta'] else 0

    features['vigilance_sequence'] = vigilance_sequence
    features['coherence_global'] = 0.4  # Replace with real calculation if available

    # Compute PAF across all channels and average
    paps = [compute_paf(data[i], sfreq) for i in range(data.shape[0])]
    features['paf'] = np.mean(paps)

    # EO vs EC alpha reactivity and PAF
    if eyes_open_raw and eyes_closed_raw:
        eo_data = eyes_open_raw.get_data() * 1e6
        ec_data = eyes_closed_raw.get_data() * 1e6
        alpha_eo = np.mean([compute_band_power(eo_data[i], sfreq, BANDS['Alpha']) for i in range(eo_data.shape[0])])
        alpha_ec = np.mean([compute_band_power(ec_data[i], sfreq, BANDS['Alpha']) for i in range(ec_data.shape[0])])
        features['alpha_shift'] = ((alpha_ec - alpha_eo) / alpha_eo) * 100 if alpha_eo else 0
        paf_ec = [compute_paf(ec_data[i], sfreq) for i in range(ec_data.shape[0])]
        paf_eo = [compute_paf(eo_data[i], sfreq) for i in range(eo_data.shape[0])]
        features['paf_ec'] = np.mean(paf_ec)
        features['paf_eo'] = np.mean(paf_eo)

    # Mu suppression ratio (C3/C4 average)
    if eyes_open_raw and eyes_closed_raw:
        mu_band = (8, 12)
        c3_idx = ch_names.index("C3") if "C3" in ch_names else None
        c4_idx = ch_names.index("C4") if "C4" in ch_names else None
        if c3_idx is not None and c4_idx is not None:
            mu_ec = (compute_band_power(eyes_closed_raw.get_data()[c3_idx], sfreq, mu_band) +
                     compute_band_power(eyes_closed_raw.get_data()[c4_idx], sfreq, mu_band)) / 2
            mu_eo = (compute_band_power(eyes_open_raw.get_data()[c3_idx], sfreq, mu_band) +
                     compute_band_power(eyes_open_raw.get_data()[c4_idx], sfreq, mu_band)) / 2
            features['mu_suppression_ratio'] = mu_eo / mu_ec if mu_ec else 0

    # Frontal asymmetry (log F4 - log F3 in alpha)
    if "F3" in ch_names and "F4" in ch_names:
        f3_idx = ch_names.index("F3")
        f4_idx = ch_names.index("F4")
        alpha_f3 = compute_band_power(data[f3_idx], sfreq, BANDS['Alpha'])
        alpha_f4 = compute_band_power(data[f4_idx], sfreq, BANDS['Alpha'])
        features['alpha_asymmetry_f3_f4'] = np.log10(alpha_f4 + 1e-6) - np.log10(alpha_f3 + 1e-6)

    # CSD data input
    if csd_raw:
        csd_data = csd_raw.get_data() * 1e6
        frontal_idxs = [i for i, ch in enumerate(csd_raw.ch_names) if ch.startswith("F")]
        parietal_idxs = [i for i, ch in enumerate(csd_raw.ch_names) if ch.startswith("P")]
        if frontal_idxs:
            features['csd_frontal_smr'] = np.mean([compute_band_power(csd_data[i], sfreq, BANDS['SMR']) for i in frontal_idxs])
        if parietal_idxs:
            features['csd_parietal_alpha'] = np.mean([compute_band_power(csd_data[i], sfreq, BANDS['Alpha']) for i in parietal_idxs])

    # Source (sLORETA) features
    if sloreta_data:
        features['sloreta_frontal_hibeta'] = sloreta_data.get('frontal_hi_beta', 0)
        features['sloreta_parietal_alpha'] = sloreta_data.get('parietal_alpha', 0)

    return features
