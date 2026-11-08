"""
diarization.py
──────────────
Task 1: Identify, label, and display unique speakers from an audio sample.

Uses pyannote.audio 3.x pipeline for state-of-the-art speaker diarization.
Each speaker segment is returned with:
  - speaker label (speaker-1, speaker-2, ...)
  - start and end timestamps
  - audio segment path (saved to outputs/)

Requires a HuggingFace token with access to:
  https://huggingface.co/pyannote/speaker-diarization-3.1
  https://huggingface.co/pyannote/segmentation-3.0
Set env var: HUGGINGFACE_TOKEN=hf_xxx
"""

import os
import json
import torch
import numpy as np
import soundfile as sf
from pyannote.audio import Pipeline


def get_diarization_pipeline(hf_token: str) -> Pipeline:
    """Load pyannote diarization pipeline from HuggingFace.

    Works across pyannote.audio 3.x (use_auth_token=) and 4.x (token=),
    since huggingface_hub 1.x removed the older `use_auth_token` keyword.
    """
    model_id = "pyannote/speaker-diarization-3.1"
    try:
        # pyannote.audio 4.x / huggingface_hub 1.x
        pipeline = Pipeline.from_pretrained(model_id, token=hf_token)
    except TypeError:
        # pyannote.audio 3.x fallback
        pipeline = Pipeline.from_pretrained(model_id, use_auth_token=hf_token)
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)
    print(f"Diarization pipeline loaded on {device}")
    return pipeline


def diarize_audio(
    audio_path: str,
    pipeline: Pipeline,
    output_dir: str = "outputs/segments",
    min_speakers: int = 1,
    max_speakers: int = 5
) -> list[dict]:
    """
    Run speaker diarization on an audio file.

    Returns list of segments:
    [
      {
        "speaker": "speaker-1",
        "start": 0.0,
        "end": 4.32,
        "duration": 4.32,
        "segment_file": "outputs/segments/sample1_speaker-1_0.0-4.32.wav"
      }, ...
    ]
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(audio_path))[0]

    # Run diarization
    diarization = pipeline(
        audio_path,
        min_speakers=min_speakers,
        max_speakers=max_speakers
    )

    # Load full audio for segment extraction
    audio, sr = sf.read(audio_path)

    segments = []
    speaker_map = {}   # internal pyannote label → anonymized label
    speaker_counter = 1

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        # Anonymize speaker label
        if speaker not in speaker_map:
            speaker_map[speaker] = f"speaker-{speaker_counter}"
            speaker_counter += 1
        anon_label = speaker_map[speaker]

        start_sample = int(turn.start * sr)
        end_sample = int(turn.end * sr)
        segment_audio = audio[start_sample:end_sample]

        # Save segment to disk
        seg_fname = f"{base_name}_{anon_label}_{turn.start:.2f}-{turn.end:.2f}.wav"
        seg_path = os.path.join(output_dir, seg_fname)
        sf.write(seg_path, segment_audio, sr)

        segments.append({
            "speaker": anon_label,
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "duration": round(turn.end - turn.start, 3),
            "segment_file": seg_path
        })

    # Print summary
    unique_speakers = list(set(s["speaker"] for s in segments))
    print(f"\n{'='*50}")
    print(f"File: {base_name}")
    print(f"Unique speakers detected: {len(unique_speakers)}")
    for sp in sorted(unique_speakers):
        sp_segs = [s for s in segments if s["speaker"] == sp]
        total_dur = sum(s["duration"] for s in sp_segs)
        print(f"  {sp}: {len(sp_segs)} segments, {total_dur:.1f}s total")
    print(f"{'='*50}\n")

    return segments


def format_diarization_output(segments: list[dict]) -> str:
    """Format diarization output as readable text with timestamps."""
    lines = []
    for seg in sorted(segments, key=lambda x: x["start"]):
        start_fmt = _format_time(seg["start"])
        end_fmt = _format_time(seg["end"])
        lines.append(f"[{start_fmt} -> {end_fmt}]  {seg['speaker']}")
    return "\n".join(lines)


def _format_time(seconds: float) -> str:
    """Convert seconds to MM:SS.mmm format."""
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:06.3f}"


if __name__ == "__main__":
    import sys
    hf_token = os.environ.get("HUGGINGFACE_TOKEN", "")
    if not hf_token:
        print("ERROR: Set HUGGINGFACE_TOKEN environment variable")
        sys.exit(1)

    audio_path = sys.argv[1] if len(sys.argv) > 1 else "datasets/processed/Sample1C.wav"
    pipeline = get_diarization_pipeline(hf_token)
    segments = diarize_audio(audio_path, pipeline)

    print(format_diarization_output(segments))

    # Save JSON output
    out_json = f"outputs/{os.path.basename(audio_path).replace('.wav', '_diarization.json')}"
    with open(out_json, "w") as f:
        json.dump(segments, f, indent=2)
    print(f"\nSaved: {out_json}")
