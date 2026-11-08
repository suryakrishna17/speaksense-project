"""
visualize.py
────────────
Generates professional graphical insights (pitch, intensity, and MFCC spectrograms)
from a speech sample for clinical and diagnostic reporting.
"""

import os
import sys
import numpy as np
import librosa
import parselmouth
import matplotlib.pyplot as plt

def generate_visualizations(audio_path: str, output_dir: str = "outputs") -> str:
    """
    Extracts pitch, intensity, and MFCCs from the audio file and plots them.
    Saves the output visualization to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    out_img_path = os.path.join(output_dir, f"{base_name}_insights.png")

    print(f"Generating visualizations for: {audio_path}")

    # 1. Load Audio
    snd = parselmouth.Sound(audio_path)
    y, sr = librosa.load(audio_path, sr=None)
    duration = len(y) / sr
    times = np.linspace(0, duration, len(y))

    # 2. Extract Pitch (Parselmouth autocorrelation)
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values == 0] = np.nan  # hide unvoiced periods
    pitch_times = pitch.xs()

    # 3. Extract Intensity (Parselmouth)
    intensity = snd.to_intensity()
    intensity_values = intensity.values[0]
    intensity_times = intensity.xs()

    # 4. Extract MFCCs (Librosa)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Plotting Configuration
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=False)
    plt.subplots_adjust(hspace=0.45)

    # Subplot 1: Waveform & Pitch
    ax1 = axes[0]
    ax1.plot(times, y, color="#cbd5e1", alpha=0.7, label="Waveform")
    ax1.set_title("Speech Waveform & Fundamental Frequency (Pitch / F0)", fontsize=11, fontweight="bold", pad=8)
    ax1.set_ylabel("Amplitude", color="#64748b")
    ax1.tick_params(axis='y', labelcolor="#64748b")
    ax1.set_xlim(0, duration)
    ax1.grid(True, linestyle="--", alpha=0.3)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(pitch_times, pitch_values, color="#4f46e5", linewidth=2.2, label="Pitch (F0)")
    ax1_twin.set_ylabel("Frequency (Hz)", color="#4f46e5")
    ax1_twin.tick_params(axis='y', labelcolor="#4f46e5")
    valid_pitch = pitch_values[~np.isnan(pitch_values)]
    if len(valid_pitch) > 0:
        ax1_twin.set_ylim(max(0, np.min(valid_pitch) - 20), np.max(valid_pitch) + 20)

    # Subplot 2: Intensity (Energy profile)
    ax2 = axes[1]
    ax2.plot(intensity_times, intensity_values, color="#ea580c", linewidth=2, label="Intensity")
    ax2.set_title("Acoustic Intensity (Loudness Profile)", fontsize=11, fontweight="bold", pad=8)
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Intensity (dB)", color="#ea580c")
    ax2.tick_params(axis='y', labelcolor="#ea580c")
    ax2.set_xlim(0, duration)
    ax2.grid(True, linestyle="--", alpha=0.3)
    if len(intensity_values) > 0:
        ax2.set_ylim(max(0, np.min(intensity_values) - 5), np.max(intensity_values) + 5)

    # Subplot 3: Mel-Frequency Cepstral Coefficients (MFCCs)
    ax3 = axes[2]
    img = librosa.display.specshow(mfccs, x_axis='time', sr=sr, ax=ax3, cmap='viridis')
    ax3.set_title("Mel-Frequency Cepstral Coefficients (MFCCs 1-13)", fontsize=11, fontweight="bold", pad=8)
    ax3.set_xlabel("Time (seconds)")
    ax3.set_ylabel("MFCC Index")
    fig.colorbar(img, ax=ax3, format="%+2.0f dB", pad=0.02, aspect=10)

    # Save
    plt.savefig(out_img_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Successfully saved diagnostic visualization -> {out_img_path}")
    return out_img_path

if __name__ == "__main__":
    test_audio = "sample.wav"
    if not os.path.exists(test_audio):
        test_audio = "datasets/audio_samples/Sample1C.wav"
    
    if os.path.exists(test_audio):
        generate_visualizations(test_audio)
    else:
        print("ERROR: Sample audio file not found. Place a 'sample.wav' or run from the speaksense root.")
