"""
transcription.py
────────────────
Task 2: Generate Kannada transcriptions for each unique speaker
        with timestamps.

Uses faster-whisper (CTranslate2 backend) with the large-v3 model
which supports Kannada (kn) natively via OpenAI's multilingual training.

Output format per speaker segment:
  [speaker-1]  [00:00.000 → 00:04.320]  ನಮಸ್ಕಾರ, ನನ್ನ ಹೆಸರು...
"""

import os
import json
from faster_whisper import WhisperModel


def load_whisper_model(model_size: str = "large-v3", device: str = "auto") -> WhisperModel:
    """
    Load faster-whisper model.
    model_size options: tiny, base, small, medium, large-v2, large-v3
    large-v3 gives best Kannada accuracy; use medium if GPU memory is limited.
    """
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    compute_type = "float16" if device == "cuda" else "int8"
    print(f"Loading Whisper {model_size} on {device} ({compute_type})...")

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    return model


def transcribe_segment(
    audio_path: str,
    model: WhisperModel,
    language: str = "kn"   # kn = Kannada
) -> list[dict]:
    """
    Transcribe a single audio segment.

    Returns list of word/sentence-level chunks:
    [{"start": 0.0, "end": 2.4, "text": "ನಮಸ್ಕಾರ"}, ...]
    """
    segments, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,                # filter silence
        vad_parameters={"min_silence_duration_ms": 500}
    )

    chunks = []
    for seg in segments:
        chunks.append({
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip()
        })

    return chunks


def transcribe_all_speakers(
    diarization_segments: list[dict],
    model: WhisperModel,
    output_dir: str = "outputs"
) -> list[dict]:
    """
    Transcribe each speaker's segments and merge with diarization output.

    Input: diarization_segments from diarization.py
    Output: enriched list with "transcription" field added
    """
    enriched = []

    for seg in diarization_segments:
        seg_path = seg.get("segment_file")
        if not seg_path or not os.path.exists(seg_path):
            seg["transcription"] = ""
            enriched.append(seg)
            continue

        try:
            chunks = transcribe_segment(seg_path, model)
            full_text = " ".join(c["text"] for c in chunks)
            seg["transcription"] = full_text
            seg["word_chunks"] = chunks
        except Exception as e:
            print(f"  ✗ Transcription failed for {seg_path}: {e}")
            seg["transcription"] = ""
            seg["word_chunks"] = []

        enriched.append(seg)

    return enriched


def format_transcript(enriched_segments: list[dict]) -> str:
    """
    Format the full transcript as a readable document.

    [speaker-1] [00:00.000 → 00:04.320]
    ನಮಸ್ಕಾರ, ನನ್ನ ಹೆಸರು ರವಿ.
    """
    lines = []
    for seg in sorted(enriched_segments, key=lambda x: x["start"]):
        start_fmt = _fmt(seg["start"])
        end_fmt = _fmt(seg["end"])
        speaker = seg["speaker"]
        text = seg.get("transcription", "")

        lines.append(f"[{speaker}]  [{start_fmt} -> {end_fmt}]")
        if text:
            lines.append(text)
        lines.append("")  # blank line between segments

    return "\n".join(lines)


def _fmt(sec: float) -> str:
    m = int(sec // 60)
    s = sec % 60
    return f"{m:02d}:{s:06.3f}"


def save_transcript(
    enriched_segments: list[dict],
    base_name: str,
    output_dir: str = "outputs"
) -> dict[str, str]:
    """Save transcript as both JSON and human-readable TXT."""
    os.makedirs(output_dir, exist_ok=True)

    txt_path = os.path.join(output_dir, f"{base_name}_transcript.txt")
    json_path = os.path.join(output_dir, f"{base_name}_transcript.json")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(format_transcript(enriched_segments))

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(enriched_segments, f, ensure_ascii=False, indent=2)

    print(f"Saved transcript -> {txt_path}")
    return {"txt": txt_path, "json": json_path}


if __name__ == "__main__":
    import sys
    # Quick test: transcribe a single file directly
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "datasets/processed/Sample1C.wav"
    model = load_whisper_model("medium")  # use medium for speed in testing

    chunks = transcribe_segment(audio_path, model)
    for c in chunks:
        print(f"[{_fmt(c['start'])} -> {_fmt(c['end'])}]  {c['text']}")
