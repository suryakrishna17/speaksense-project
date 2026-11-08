"""
preprocess.py
─────────────
Normalizes all audio files to a consistent format:
  - Mono channel
  - 16,000 Hz sample rate (required by Whisper & pyannote)
  - PCM 16-bit WAV
"""

import os
import subprocess
import tempfile
import librosa
import soundfile as sf
import numpy as np

TARGET_SR = 16000


def load_and_normalize(audio_path: str) -> tuple[np.ndarray, int]:
    """
    Load any WAV/MP3 file and normalize to mono 16kHz.
    Returns (audio_array, sample_rate)
    """
    try:
        y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    except Exception:
        y, sr = _load_with_ffmpeg(audio_path)
    # Peak normalization to prevent clipping
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y)) * 0.95
    return y, TARGET_SR


def _load_with_ffmpeg(audio_path: str) -> tuple[np.ndarray, int]:
    """Decode mislabeled or compressed audio through imageio-ffmpeg."""
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "Audio is not a standard WAV file. Install imageio-ffmpeg or convert it to WAV first."
        ) from exc

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp_path = tmp.name

    try:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i", audio_path,
                "-ac", "1",
                "-ar", str(TARGET_SR),
                "-acodec", "pcm_s16le",
                tmp_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return librosa.load(tmp_path, sr=TARGET_SR, mono=True)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def preprocess_dataset(input_dir: str, output_dir: str) -> list[dict]:
    """
    Batch preprocess all WAV files in input_dir.
    Saves normalized files to output_dir.
    Returns metadata list with label info decoded from filename.
    """
    os.makedirs(output_dir, exist_ok=True)
    metadata = []

    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith(".wav"):
            continue

        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, fname)

        try:
            y, sr = load_and_normalize(input_path)
            sf.write(output_path, y, sr, subtype="PCM_16")

            # Decode label from filename: SampleNNX.wav
            # C = Child (age < 15), W = Woman/Adult (age > 18)
            name = fname.replace(".wav", "").replace(".WAV", "")
            suffix = name[-1].upper() if name[-1].upper() in ("C", "W") else name[-2].upper()
            age_label = "child" if suffix == "C" else "adult"
            # W files are female adult speakers in this dataset
            gender_label = "female" if suffix == "W" else "unknown"

            metadata.append({
                "filename": fname,
                "processed_path": output_path,
                "age_label": age_label,        # child | adult
                "gender_label": gender_label,  # female | unknown
                "duration_sec": round(len(y) / sr, 2),
            })
            print(f"  OK {fname} -> {age_label} ({len(y)/sr:.1f}s)")

        except Exception as e:
            print(f"  ERROR {fname}: {e}")

    print(f"\nPreprocessed {len(metadata)} files -> {output_dir}")
    return metadata


if __name__ == "__main__":
    preprocess_dataset(
        input_dir="datasets/audio_samples",
        output_dir="datasets/processed"
    )
