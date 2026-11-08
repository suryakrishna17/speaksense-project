"""
augment.py
──────────
Boosts the 48-sample dataset to 200+ samples using audio augmentation.
Augmented samples preserve speaker characteristics while adding variation.

Techniques used:
  1. Pitch Shift    ±2 semitones (preserves speaker identity)
  2. Time Stretch   ±10%         (simulates speaking rate variation)
  3. Gaussian Noise σ=0.003      (simulates mic/room noise)
  4. RIR Simulation              (room impulse response — optional)

Run: python src/augment.py --input datasets/processed --output datasets/augmented
"""

import os
import argparse
import numpy as np
import librosa
import soundfile as sf

AUGMENTATIONS = [
    "pitch_up",
    "pitch_down",
    "time_stretch_fast",
    "time_stretch_slow",
    "add_noise",
    "pitch_up_noise",
    "pitch_down_noise",
]


def pitch_shift(y: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)


def time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    return librosa.effects.time_stretch(y, rate=rate)


def add_gaussian_noise(y: np.ndarray, sigma: float = 0.003) -> np.ndarray:
    noise = np.random.normal(0, sigma, y.shape)
    return np.clip(y + noise, -1.0, 1.0)


def augment_one(y: np.ndarray, sr: int, aug_type: str) -> np.ndarray:
    """Apply a single augmentation to an audio array."""
    if aug_type == "pitch_up":
        return pitch_shift(y, sr, semitones=2.0)
    elif aug_type == "pitch_down":
        return pitch_shift(y, sr, semitones=-2.0)
    elif aug_type == "time_stretch_fast":
        return time_stretch(y, rate=1.1)
    elif aug_type == "time_stretch_slow":
        return time_stretch(y, rate=0.9)
    elif aug_type == "add_noise":
        return add_gaussian_noise(y, sigma=0.003)
    elif aug_type == "pitch_up_noise":
        y2 = pitch_shift(y, sr, semitones=1.5)
        return add_gaussian_noise(y2, sigma=0.002)
    elif aug_type == "pitch_down_noise":
        y2 = pitch_shift(y, sr, semitones=-1.5)
        return add_gaussian_noise(y2, sigma=0.002)
    else:
        return y


def augment_dataset(input_dir: str, output_dir: str, augmentations: list[str] = None) -> int:
    """
    Augment all WAV files in input_dir and save to output_dir.
    Returns total number of files created.
    """
    os.makedirs(output_dir, exist_ok=True)
    if augmentations is None:
        augmentations = AUGMENTATIONS

    files = [f for f in sorted(os.listdir(input_dir)) if f.lower().endswith(".wav")]
    total_created = 0

    print(f"\nAugmenting {len(files)} files × {len(augmentations)} techniques")
    print(f"Output: {output_dir}\n")

    for fname in files:
        src_path = os.path.join(input_dir, fname)
        base = os.path.splitext(fname)[0]

        # Copy original
        y, sr = librosa.load(src_path, sr=16000, mono=True)
        sf.write(os.path.join(output_dir, fname), y, sr)
        total_created += 1

        for aug in augmentations:
            try:
                y_aug = augment_one(y, sr, aug)
                out_name = f"{base}_{aug}.wav"
                sf.write(os.path.join(output_dir, out_name), y_aug, sr, subtype="PCM_16")
                total_created += 1
            except Exception as e:
                print(f"  ✗ {fname} / {aug}: {e}")

        print(f"  ✓ {fname} -> {len(augmentations) + 1} variants")

    print(f"\nTotal files created: {total_created}")
    print(f"Dataset size: {len(files)} -> {total_created} (+{total_created - len(files)} augmented)")
    return total_created


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="datasets/processed",  help="Input dir of processed WAVs")
    parser.add_argument("--output", default="datasets/augmented",  help="Output dir for augmented WAVs")
    args = parser.parse_args()
    augment_dataset(args.input, args.output)
