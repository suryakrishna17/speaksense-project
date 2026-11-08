# Acoustic Parameters — SpeakSense
**Task 3 Documentation: All speech/acoustic parameters used for classification**

---

## Overview

SpeakSense extracts **71 acoustic features** per audio sample, organized into 9 categories.
These features power three binary classification tasks:
- Adult vs Child
- Male vs Female  
- Typical vs Atypical speech

---

## 1. MFCCs (Mel-Frequency Cepstral Coefficients) — 39 features

| Feature | Description |
|---|---|
| `mfcc_1_mean` .. `mfcc_13_mean` | Mean of each of the 13 MFCC coefficients across the recording |
| `mfcc_1_std` .. `mfcc_13_std` | Standard deviation of each MFCC coefficient |
| `mfcc_delta_1_mean` .. `mfcc_delta_13_mean` | Mean of delta (velocity) MFCCs |

**How computed:** Using `librosa.feature.mfcc(n_mfcc=13)` on 16kHz mono audio.  
**Relevance:** Captures the timbral fingerprint of a speaker. Primary features for speaker identity, age group, and gender classification. Children and women have higher-frequency energy patterns reflected in upper MFCC coefficients.

---

## 2. Fundamental Frequency / Pitch (F0) — 5 features

| Feature | Description | Typical range |
|---|---|---|
| `f0_mean_hz` | Mean pitch over voiced frames | Males: 85–180 Hz, Females: 165–255 Hz, Children: 250–400 Hz |
| `f0_std_hz` | Standard deviation of pitch (prosodic variability) | |
| `f0_min_hz` | Minimum pitch in voiced frames | |
| `f0_max_hz` | Maximum pitch in voiced frames | |
| `f0_range_hz` | Difference between max and min pitch | |

**How computed:** `librosa.pyin()` with fmin=50 Hz, fmax=500 Hz.  
**Relevance:** Most discriminative feature for Adult/Child and Male/Female classification. Children's shorter vocal tracts produce systematically higher F0.

---

## 3. Jitter — 3 features

| Feature | Description | Atypical threshold |
|---|---|---|
| `jitter_local_pct` | Cycle-to-cycle pitch period variation (%) | > 1.04% |
| `jitter_rap_pct` | Relative Average Perturbation (%) | > 0.68% |
| `jitter_ppq5_pct` | 5-point Period Perturbation Quotient (%) | > 0.84% |

**How computed:** Praat via `parselmouth` — `To PointProcess (periodic, cc)` → `Get jitter`.  
**Relevance:** Elevated jitter indicates vocal fold irregularity. Key marker for Atypical speech (dysarthria, Parkinson's disease, laryngeal pathology).

---

## 4. Shimmer — 3 features

| Feature | Description | Atypical threshold |
|---|---|---|
| `shimmer_local_db` | Cycle-to-cycle amplitude variation (dB) | > 0.35 dB |
| `shimmer_apq3_db` | 3-point Amplitude Perturbation Quotient (dB) | |
| `shimmer_apq5_db` | 5-point Amplitude Perturbation Quotient (dB) | |

**How computed:** Praat via `parselmouth` — `Get shimmer (local)`, `(apq3)`, `(apq5)`.  
**Relevance:** Elevated shimmer indicates rough or breathy voice quality. Co-occurs with elevated jitter in Atypical speakers.

---

## 5. Harmonics-to-Noise Ratio (HNR) — 1 feature

| Feature | Description | Normal range |
|---|---|---|
| `hnr_mean_db` | Ratio of harmonic energy to noise energy (dB) | > 20 dB (healthy voice) |

**How computed:** Praat via `parselmouth` — `To Harmonicity (cc)` → `Get mean`.  
**Relevance:** Low HNR indicates a noisy, breathy, or hoarse voice. Strong indicator of Atypical speech. Normal adults typically show > 20 dB.

---

## 6. Spectral Features — 10 features

| Feature | Description |
|---|---|
| `spectral_centroid_mean` | Weighted mean frequency (brightness of sound) |
| `spectral_centroid_std` | Variability of spectral centroid |
| `spectral_bandwidth_mean` | Mean spectral bandwidth (spread of frequencies) |
| `spectral_rolloff_mean` | Frequency below which 85% of energy is concentrated |
| `zcr_mean` | Zero Crossing Rate mean (noisiness estimate) |
| `zcr_std` | Zero Crossing Rate standard deviation |
| `spectral_contrast_band1..7_mean` | Energy contrast across 7 frequency sub-bands |

**How computed:** `librosa.feature.spectral_centroid/bandwidth/rolloff/contrast/zero_crossing_rate`.  
**Relevance:** Spectral centroid distinguishes age groups (children: higher centroid). ZCR distinguishes voiced vs unvoiced segments. Spectral contrast captures formant prominence.

---

## 7. Formants (F1, F2, F3) — 3 features

| Feature | Description | Age pattern |
|---|---|---|
| `f1_mean_hz` | First formant frequency (jaw opening, vowel height) | Children: 600–1000 Hz |
| `f2_mean_hz` | Second formant frequency (tongue advancement) | Children: 1500–3000 Hz |
| `f3_mean_hz` | Third formant frequency (lip rounding) | |

**How computed:** Praat via `parselmouth` — `To Formant (burg)`, max formant = 5500 Hz.  
**Relevance:** Children have shorter vocal tracts (~12 cm vs adult ~17 cm), producing systematically higher formant frequencies. Critical for Adult vs Child classification.

---

## 8. Energy / RMS — 3 features

| Feature | Description |
|---|---|
| `rms_mean` | Mean Root Mean Square energy (overall loudness) |
| `rms_std` | Standard deviation of RMS (loudness variability) |
| `rms_max` | Peak energy |

**How computed:** `librosa.feature.rms(frame_length=2048, hop_length=512)`.  
**Relevance:** Atypical speakers may show reduced or irregular loudness patterns. Also used for Voice Activity Detection (VAD) within the pipeline.

---

## 9. Temporal / Prosodic Features — 5 features

| Feature | Description | Atypical pattern |
|---|---|---|
| `pause_count` | Number of detected pauses in speech | Elevated in Atypical |
| `voiced_duration_sec` | Total duration of voiced speech | Reduced in Atypical |
| `total_duration_sec` | Full recording duration | |
| `voiced_fraction` | Proportion of recording that is voiced speech | Reduced in Atypical |
| `speech_rate_approx` | Approximate speech rate (pauses/sec) | Slower in Atypical |

**How computed:** RMS-based VAD thresholding at 30% of mean RMS.  
**Relevance:** Atypical speakers (dysarthria, apraxia) show reduced speech rate, more frequent pauses, and lower voiced fraction compared to Typical speakers.

---

## Classification Mapping

| Task | Primary Features Used |
|---|---|
| **Adult vs Child** | `f0_mean_hz`, `f1_mean_hz`, `f2_mean_hz`, `f3_mean_hz`, all MFCCs |
| **Male vs Female** | `f0_mean_hz`, `f0_range_hz`, `f1_mean_hz`, `f2_mean_hz`, `spectral_centroid_mean`, MFCCs 1-13 |
| **Typical vs Atypical** | `jitter_local_pct`, `shimmer_local_db`, `hnr_mean_db`, `voiced_fraction`, `pause_count`, `speech_rate_approx` |

## Classifier Architecture

Each task uses a **soft-voting ensemble**:
- Random Forest (200 trees, max_depth=10)
- XGBoost (150 estimators, lr=0.05)
- SVM (RBF kernel, C=1.0)

Additionally, **clinical rule-based thresholds** override ML predictions for Typicality when:
- Jitter > 1.04% **OR** Shimmer > 0.35 dB **OR** HNR < 15 dB
