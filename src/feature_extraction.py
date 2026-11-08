"""
feature_extraction.py
─────────────────────
Task 3: Identify and list all speech/acoustic parameters used for
        speaker classification.

Extracts a 50+ dimensional feature vector per audio file covering:
  1. MFCCs (Mel-Frequency Cepstral Coefficients)
  2. Pitch / Fundamental Frequency (F0)
  3. Voice Quality (Jitter, Shimmer, HNR)
  4. Spectral Features
  5. Temporal / Prosodic Features
  6. Energy Features
"""

import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# FEATURE CATALOGUE  (Task 3 documentation)
# ─────────────────────────────────────────────────────────────
FEATURE_CATALOGUE = {
    "MFCCs": {
        "description": "Mel-Frequency Cepstral Coefficients capture the timbral texture of speech.",
        "features": ["mfcc_1..13_mean", "mfcc_1..13_std", "mfcc_delta_1..13_mean"],
        "relevance": "Primary features for speaker identity, age, and gender classification.",
        "count": 39
    },
    "Pitch (F0)": {
        "description": "Fundamental frequency of vocal fold vibration.",
        "features": ["f0_mean_hz", "f0_std_hz", "f0_min_hz", "f0_max_hz", "f0_range_hz"],
        "relevance": "Males ~85-180 Hz, Females ~165-255 Hz, Children ~250-400 Hz.",
        "count": 5
    },
    "Jitter": {
        "description": "Cycle-to-cycle variation in pitch period (vocal instability).",
        "features": ["jitter_local_%", "jitter_rap_%", "jitter_ppq5_%"],
        "relevance": "Elevated in Atypical speech (dysarthria, Parkinson's). Threshold: >1.04%",
        "count": 3
    },
    "Shimmer": {
        "description": "Cycle-to-cycle variation in amplitude (voice roughness).",
        "features": ["shimmer_local_db", "shimmer_apq3_db", "shimmer_apq5_db"],
        "relevance": "Elevated in Atypical speech. Threshold: >0.35 dB",
        "count": 3
    },
    "HNR": {
        "description": "Harmonics-to-Noise Ratio — voice clarity vs breathiness.",
        "features": ["hnr_mean_db"],
        "relevance": "Lower HNR in Atypical speech. Normal: >20 dB",
        "count": 1
    },
    "Spectral Features": {
        "description": "Frequency-domain characteristics of the speech signal.",
        "features": [
            "spectral_centroid_mean", "spectral_centroid_std",
            "spectral_bandwidth_mean", "spectral_rolloff_mean",
            "spectral_contrast_mean_6bands", "zero_crossing_rate_mean"
        ],
        "relevance": "Distinguish age groups and typical vs atypical articulation.",
        "count": 10
    },
    "Formants": {
        "description": "Resonant frequencies of the vocal tract (F1, F2, F3).",
        "features": ["f1_mean_hz", "f2_mean_hz", "f3_mean_hz"],
        "relevance": "Children have shorter vocal tracts → higher formants. Key for Adult/Child split.",
        "count": 3
    },
    "Energy / RMS": {
        "description": "Overall loudness and energy distribution.",
        "features": ["rms_mean", "rms_std", "energy_entropy"],
        "relevance": "Speech intensity patterns differ by age and disorder type.",
        "count": 3
    },
    "Temporal Features": {
        "description": "Time-domain speech characteristics.",
        "features": ["speech_rate_syllables_per_sec", "voiced_fraction", "pause_count", "avg_pause_duration_sec"],
        "relevance": "Atypical speakers show slower rate, more pauses, lower voiced fraction.",
        "count": 4
    },
}

TOTAL_FEATURES = sum(v["count"] for v in FEATURE_CATALOGUE.values())  # ~71 features


# ─────────────────────────────────────────────────────────────
# EXTRACTION FUNCTIONS
# ─────────────────────────────────────────────────────────────

def extract_mfcc_features(y: np.ndarray, sr: int) -> dict:
    """Extract 39 MFCC features (13 mean + 13 std + 13 delta mean)."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    features = {}
    for i in range(13):
        features[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc_{i+1}_std"] = float(np.std(mfcc[i]))
        features[f"mfcc_delta_{i+1}_mean"] = float(np.mean(mfcc_delta[i]))
    return features


def extract_pitch_features(y: np.ndarray, sr: int) -> dict:
    """Extract F0 (pitch) statistics."""
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=50, fmax=500, sr=sr,
        frame_length=2048, hop_length=512
    )
    f0_voiced = f0[voiced_flag & ~np.isnan(f0)]
    if len(f0_voiced) == 0:
        return {k: 0.0 for k in ["f0_mean_hz", "f0_std_hz", "f0_min_hz", "f0_max_hz", "f0_range_hz", "voiced_fraction"]}

    return {
        "f0_mean_hz": float(np.mean(f0_voiced)),
        "f0_std_hz": float(np.std(f0_voiced)),
        "f0_min_hz": float(np.min(f0_voiced)),
        "f0_max_hz": float(np.max(f0_voiced)),
        "f0_range_hz": float(np.max(f0_voiced) - np.min(f0_voiced)),
        "voiced_fraction": float(np.sum(voiced_flag) / len(voiced_flag))
    }


def extract_voice_quality_features(audio_path: str) -> dict:
    """Extract Jitter, Shimmer, HNR using Praat via parselmouth."""
    try:
        sound = parselmouth.Sound(audio_path)
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)

        jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_rap   = call(point_process, "Get jitter (rap)",   0, 0, 0.0001, 0.02, 1.3)
        jitter_ppq5  = call(point_process, "Get jitter (ppq5)",  0, 0, 0.0001, 0.02, 1.3)

        shimmer_local = call([sound, point_process], "Get shimmer (local)",    0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq3  = call([sound, point_process], "Get shimmer (apq3)",     0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq5  = call([sound, point_process], "Get shimmer (apq5)",     0, 0, 0.0001, 0.02, 1.3, 1.6)

        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)

        return {
            "jitter_local_pct": float(jitter_local * 100) if jitter_local else 0.0,
            "jitter_rap_pct":   float(jitter_rap * 100)   if jitter_rap   else 0.0,
            "jitter_ppq5_pct":  float(jitter_ppq5 * 100)  if jitter_ppq5  else 0.0,
            "shimmer_local_db": float(shimmer_local) if shimmer_local else 0.0,
            "shimmer_apq3_db":  float(shimmer_apq3)  if shimmer_apq3  else 0.0,
            "shimmer_apq5_db":  float(shimmer_apq5)  if shimmer_apq5  else 0.0,
            "hnr_mean_db":      float(hnr)            if hnr           else 0.0,
        }
    except Exception as e:
        print(f"  Voice quality extraction failed: {e}")
        return {k: 0.0 for k in [
            "jitter_local_pct", "jitter_rap_pct", "jitter_ppq5_pct",
            "shimmer_local_db", "shimmer_apq3_db", "shimmer_apq5_db", "hnr_mean_db"
        ]}


def extract_spectral_features(y: np.ndarray, sr: int) -> dict:
    """Extract spectral centroid, bandwidth, rolloff, contrast, ZCR."""
    centroid   = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth  = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff    = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    contrast   = librosa.feature.spectral_contrast(y=y, sr=sr)  # 7 bands
    zcr        = librosa.feature.zero_crossing_rate(y)[0]

    feats = {
        "spectral_centroid_mean": float(np.mean(centroid)),
        "spectral_centroid_std":  float(np.std(centroid)),
        "spectral_bandwidth_mean": float(np.mean(bandwidth)),
        "spectral_rolloff_mean":  float(np.mean(rolloff)),
        "zcr_mean":               float(np.mean(zcr)),
        "zcr_std":                float(np.std(zcr)),
    }
    for i in range(contrast.shape[0]):
        feats[f"spectral_contrast_band{i+1}_mean"] = float(np.mean(contrast[i]))

    return feats


def extract_formant_features(audio_path: str) -> dict:
    """Extract first 3 formant frequencies using Praat."""
    try:
        sound = parselmouth.Sound(audio_path)
        formants = call(sound, "To Formant (burg)", 0, 5, 5500, 0.025, 50)
        duration = call(sound, "Get total duration")

        f1_vals, f2_vals, f3_vals = [], [], []
        time_step = 0.01
        t = time_step
        while t < duration:
            f1 = call(formants, "Get value at time", 1, t, "Hertz", "Linear")
            f2 = call(formants, "Get value at time", 2, t, "Hertz", "Linear")
            f3 = call(formants, "Get value at time", 3, t, "Hertz", "Linear")
            if f1 and not (f1 != f1): f1_vals.append(f1)  # nan check
            if f2 and not (f2 != f2): f2_vals.append(f2)
            if f3 and not (f3 != f3): f3_vals.append(f3)
            t += time_step

        return {
            "f1_mean_hz": float(np.mean(f1_vals)) if f1_vals else 0.0,
            "f2_mean_hz": float(np.mean(f2_vals)) if f2_vals else 0.0,
            "f3_mean_hz": float(np.mean(f3_vals)) if f3_vals else 0.0,
        }
    except Exception as e:
        print(f"  Formant extraction failed: {e}")
        return {"f1_mean_hz": 0.0, "f2_mean_hz": 0.0, "f3_mean_hz": 0.0}


def extract_energy_features(y: np.ndarray, sr: int) -> dict:
    """Extract RMS energy features."""
    rms = librosa.feature.rms(y=y)[0]
    return {
        "rms_mean": float(np.mean(rms)),
        "rms_std":  float(np.std(rms)),
        "rms_max":  float(np.max(rms)),
    }


def extract_temporal_features(y: np.ndarray, sr: int) -> dict:
    """Extract pause count, speech rate estimates."""
    # Voice activity via RMS thresholding
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    threshold = np.mean(rms) * 0.3
    voiced = rms > threshold

    # Count pauses (transitions from voiced to silence)
    transitions = np.diff(voiced.astype(int))
    pause_count = int(np.sum(transitions == -1))

    # Estimate speaking duration
    hop_duration = 512 / sr
    voiced_duration = float(np.sum(voiced) * hop_duration)
    total_duration = float(len(y) / sr)

    return {
        "pause_count": pause_count,
        "voiced_duration_sec": round(voiced_duration, 3),
        "total_duration_sec": round(total_duration, 3),
        "voiced_fraction": round(voiced_duration / total_duration, 4) if total_duration > 0 else 0.0,
        "speech_rate_approx": round(pause_count / total_duration, 4) if total_duration > 0 else 0.0,
    }


def extract_all_features(audio_path: str, y: np.ndarray = None, sr: int = 16000) -> dict:
    """
    Full feature extraction pipeline for a single audio file.
    Returns a flat dict of all ~71 features.
    """
    if y is None:
        import librosa as _lb
        y, sr = _lb.load(audio_path, sr=16000, mono=True)

    features = {"filename": audio_path}
    features.update(extract_mfcc_features(y, sr))
    features.update(extract_pitch_features(y, sr))
    features.update(extract_voice_quality_features(audio_path))
    features.update(extract_spectral_features(y, sr))
    features.update(extract_formant_features(audio_path))
    features.update(extract_energy_features(y, sr))
    features.update(extract_temporal_features(y, sr))

    return features


def print_feature_catalogue():
    """Print Task 3 documentation: all acoustic parameters used."""
    print("\n" + "="*60)
    print("ACOUSTIC PARAMETERS USED FOR SPEAKER CLASSIFICATION")
    print("="*60)
    for category, info in FEATURE_CATALOGUE.items():
        print(f"\n{'-'*50}")
        print(f"  {category}  ({info['count']} features)")
        print(f"  {info['description']}")
        print(f"  Relevance: {info['relevance']}")
        print(f"  Features: {', '.join(info['features'])}")
    print(f"\n{'='*60}")
    print(f"  TOTAL: {TOTAL_FEATURES} features per speaker segment")
    print("="*60 + "\n")


if __name__ == "__main__":
    print_feature_catalogue()
    import sys
    if len(sys.argv) > 1:
        feats = extract_all_features(sys.argv[1])
        for k, v in feats.items():
            print(f"  {k}: {v}")
