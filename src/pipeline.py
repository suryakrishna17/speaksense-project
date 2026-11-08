"""
pipeline.py
───────────
End-to-end runner for the Automated Speech Diagnostic Engine.

Usage:
  # Process a single file:
  python src/pipeline.py --file datasets/processed/Sample1C.wav

  # Train all classifiers on the full dataset:
  python src/pipeline.py --train --data datasets/processed

  # Full pipeline (train + process one file):
  python src/pipeline.py --train --data datasets/processed --file datasets/processed/Sample1C.wav

  # Set your HuggingFace token for diarization:
  export HUGGINGFACE_TOKEN=hf_xxxxx
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import librosa

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from preprocess import load_and_normalize, preprocess_dataset
from feature_extraction import extract_all_features, print_feature_catalogue
from classification import train_all, classify_speaker


def run_training_pipeline(data_dir: str) -> pd.DataFrame:
    """
    Extract features from all processed audio files and train classifiers.
    Returns feature DataFrame.
    """
    print(f"\n{'='*60}")
    print("PHASE 1: FEATURE EXTRACTION")
    print(f"{'='*60}")

    records = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.lower().endswith(".wav"):
            continue

        fpath = os.path.join(data_dir, fname)
        print(f"\nExtracting: {fname}")

        # Decode label from filename
        name = fname.replace(".wav", "").replace(".WAV", "")
        last_char = name[-1].upper()
        if last_char not in ("C", "W"):
            last_char = name[-2].upper()

        age_label    = "child"  if last_char == "C" else "adult"
        gender_label = "female" if last_char == "W" else "unknown"

        try:
            feats = extract_all_features(fpath)
            feats["filename"]     = fname
            feats["age_label"]    = age_label
            feats["gender_label"] = gender_label
            records.append(feats)
        except Exception as e:
            print(f"  ERROR: {e}")

    df = pd.DataFrame(records)
    df.to_csv("outputs/features.csv", index=False)
    print(f"\nExtracted features for {len(df)} files -> outputs/features.csv")

    print(f"\n{'='*60}")
    print("PHASE 2: CLASSIFIER TRAINING")
    print(f"{'='*60}")
    results = train_all(df)

    for r in results:
        if r.get("skipped"):
            print(f"\n  - {r['task']}: skipped ({r['reason']})")
        else:
            print(f"\n  OK {r['task']}: CV F1 = {r['cv_f1_mean']:.3f} +/- {r['cv_f1_std']:.3f}")

    return df


def run_inference_pipeline(
    audio_path: str,
    hf_token: str = None,
    skip_diarization: bool = False
) -> dict:
    """
    Full inference on a single audio file.
    Returns structured output with diarization, transcription, and classification.
    """
    print(f"\n{'='*60}")
    print(f"PROCESSING: {os.path.basename(audio_path)}")
    print(f"{'='*60}")

    output = {
        "file": os.path.basename(audio_path),
        "speakers": [],
        "classification": {},
        "transcript": ""
    }

    # ── Step 1: Speaker Diarization ──────────────────────────
    diarization_segments = []
    if not skip_diarization and hf_token:
        print("\n[1/4] Running speaker diarization...")
        from diarization import get_diarization_pipeline, diarize_audio, format_diarization_output
        pipeline = get_diarization_pipeline(hf_token)
        diarization_segments = diarize_audio(audio_path, pipeline)
        print(format_diarization_output(diarization_segments))
    else:
        print("\n[1/4] Diarization skipped (no HF token). Treating file as single speaker.")
        diarization_segments = [{
            "speaker": "speaker-1",
            "start": 0.0,
            "end": librosa.get_duration(path=audio_path),
            "duration": librosa.get_duration(path=audio_path),
            "segment_file": audio_path
        }]

    # ── Step 2: Kannada Transcription ────────────────────────
    print("\n[2/4] Transcribing (Kannada)...")
    try:
        from transcription import load_whisper_model, transcribe_all_speakers, save_transcript
        whisper = load_whisper_model("medium")
        enriched = transcribe_all_speakers(diarization_segments, whisper)
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        paths = save_transcript(enriched, base_name)
        output["transcript_files"] = paths
        print(f"  Transcript saved: {paths['txt']}")
    except ImportError:
        print("  faster-whisper not installed — skipping transcription")
        enriched = diarization_segments

    # ── Step 3: Feature Extraction ───────────────────────────
    print("\n[3/4] Extracting acoustic features...")
    features = extract_all_features(audio_path)
    output["acoustic_features"] = {
        k: v for k, v in features.items()
        if k != "filename" and isinstance(v, (int, float))
    }

    # Key feature summary
    print(f"  F0 (pitch): {features.get('f0_mean_hz', 0):.1f} Hz")
    print(f"  Jitter: {features.get('jitter_local_pct', 0):.3f}%")
    print(f"  Shimmer: {features.get('shimmer_local_db', 0):.3f} dB")
    print(f"  HNR: {features.get('hnr_mean_db', 0):.1f} dB")
    print(f"  Voiced fraction: {features.get('voiced_fraction', 0):.2%}")

    # ── Step 4: Classification ───────────────────────────────
    print("\n[4/4] Classifying speaker...")
    classification = classify_speaker(features)
    output["classification"] = classification

    print(f"\n  +-- RESULT ----------------------------")
    print(f"  |  Age:        {classification['age']['prediction'].upper()}"
          f"  (confidence: {classification['age']['confidence']:.0%})")
    print(f"  |  Gender:     {classification['gender']['prediction'].upper()}"
          f"  (confidence: {classification['gender']['confidence']:.0%})")
    print(f"  |  Typicality: {classification['typicality']['prediction'].upper()}"
          f"  (confidence: {classification['typicality']['confidence']:.0%})")
    print(f"  +--------------------------------------")

    # Save output JSON
    os.makedirs("outputs", exist_ok=True)
    out_path = f"outputs/{os.path.splitext(os.path.basename(audio_path))[0]}_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        # Convert numpy types
        def convert(obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            return obj
        json.dump(output, f, ensure_ascii=False, indent=2, default=convert)
    print(f"\n  Output saved -> {out_path}")

    return output


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    parser = argparse.ArgumentParser(description="SpeakSense: Automated Speech Diagnostic Engine")
    parser.add_argument("--file",  help="Path to audio file for inference")
    parser.add_argument("--data",  help="Directory of processed audio files for training")
    parser.add_argument("--train", action="store_true", help="Run training pipeline")
    parser.add_argument("--catalogue", action="store_true", help="Print feature catalogue")
    parser.add_argument("--no-diarize", action="store_true", help="Skip diarization")
    args = parser.parse_args()

    hf_token = os.environ.get("HUGGINGFACE_TOKEN", "")

    if args.catalogue:
        print_feature_catalogue()

    if args.train and args.data:
        run_training_pipeline(args.data)

    if args.file:
        run_inference_pipeline(
            args.file,
            hf_token=hf_token,
            skip_diarization=args.no_diarize or not hf_token
        )

    if not any([args.file, args.train, args.catalogue]):
        parser.print_help()
