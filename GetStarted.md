# Getting Started with SpeakSense

## Requirements

- Python 3.10 or newer
- Node.js 18 or newer
- 8 GB RAM minimum, 16 GB recommended
- Optional CUDA GPU for faster Whisper and pyannote inference
- Hugging Face token for speaker diarization

## Backend

```bash
cd speaksense
pip install -r requirements.txt
uvicorn src.api:app --reload --port 8000
```

Check the backend at:

```text
http://localhost:8000/health
http://localhost:8000/docs
```

## Frontend

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

## Hugging Face Setup for Diarization

Diarization uses `pyannote/speaker-diarization-3.1`.

1. Create a Hugging Face account.
2. Accept access terms for `pyannote/speaker-diarization-3.1`.
3. Accept access terms for `pyannote/segmentation-3.0`.
4. Create an access token.
5. Set it before running the full diarization pipeline.

PowerShell:

```powershell
$env:HUGGINGFACE_TOKEN="hf_your_token_here"
```

Bash:

```bash
export HUGGINGFACE_TOKEN=hf_your_token_here
```

Without this token, SpeakSense can still extract features, classify, transcribe, and generate reports, but diarization is skipped.

## Train Models

Place WAV files in:

```text
datasets/audio_samples/
```

Then run:

```bash
python src/preprocess.py
python src/pipeline.py --train --data datasets/processed
```

Generated model files are saved in `models/`.

## Analyze One File from the Command Line

```bash
python src/pipeline.py --file sample.wav --no-diarize
```

With diarization:

```bash
python src/pipeline.py --file sample.wav
```

## Output Files

Generated files appear in `outputs/`, including transcripts, JSON results, speaker segments, and PDF reports.
