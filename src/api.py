"""
api.py
──────
FastAPI backend for SpeakSense.
Exposes REST endpoints for the React frontend.

Run: uvicorn src.api:app --reload --port 8000

Configuration (environment variables):
  SPEAKSENSE_MAX_UPLOAD_MB   Max accepted upload size in MB        (default 50)
  SPEAKSENSE_WHISPER_MODEL   faster-whisper model size             (default medium)
  SPEAKSENSE_CORS_ORIGINS    Comma-separated allowed origins, or * (default *)
  SPEAKSENSE_LOG_LEVEL       Logging level                         (default INFO)
  HUGGINGFACE_TOKEN          Token for pyannote diarization        (required for /diarize)
"""

import os
import sys
import time
import logging
import tempfile
import importlib.util

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(__file__))
from feature_extraction import extract_all_features, FEATURE_CATALOGUE, TOTAL_FEATURES
from classification import classify_speaker
from explainability import explain_all

# ── configuration ────────────────────────────────────────────
MAX_UPLOAD_MB    = int(os.environ.get("SPEAKSENSE_MAX_UPLOAD_MB", "150"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
WHISPER_MODEL    = os.environ.get("SPEAKSENSE_WHISPER_MODEL", "medium")
HF_TOKEN         = os.environ.get("HUGGINGFACE_TOKEN", "")
CORS_ORIGINS     = [o.strip() for o in os.environ.get("SPEAKSENSE_CORS_ORIGINS", "*").split(",") if o.strip()]
LOG_LEVEL        = os.environ.get("SPEAKSENSE_LOG_LEVEL", "INFO").upper()

ALLOWED_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".aac"}

# ── logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
)
log = logging.getLogger("speaksense")


# ── capability detection (which optional deps are installed) ──
def _installed(*modules: str) -> bool:
    return all(importlib.util.find_spec(m) is not None for m in modules)

HAS_REPORT      = _installed("reportlab")
HAS_WHISPER     = _installed("faster_whisper")
HAS_DIARIZATION = _installed("torch", "pyannote.audio")


def capabilities() -> dict:
    """Report which optional features are currently available."""
    return {
        "analyze":     True,
        "report":      HAS_REPORT,
        "transcribe":  HAS_WHISPER,
        "diarize":     HAS_DIARIZATION and bool(HF_TOKEN),
        "diarize_deps_installed": HAS_DIARIZATION,
        "hf_token_set": bool(HF_TOKEN),
    }


# ── lazy, cached heavy models ────────────────────────────────
_whisper_model = None
def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        from transcription import load_whisper_model
        log.info("Loading Whisper model '%s' (first use, may take a while)…", WHISPER_MODEL)
        _whisper_model = load_whisper_model(WHISPER_MODEL)
    return _whisper_model

_diar_pipeline = None
def get_diarizer():
    global _diar_pipeline
    if _diar_pipeline is None:
        from diarization import get_diarization_pipeline
        log.info("Loading pyannote diarization pipeline (first use, may take a while)…")
        _diar_pipeline = get_diarization_pipeline(HF_TOKEN)
    return _diar_pipeline


# ── upload validation helper ─────────────────────────────────
async def save_upload(file: UploadFile) -> str:
    """Validate an uploaded audio file and persist it to a temp path.

    Raises HTTPException on invalid type / empty / oversized uploads.
    Returns the temp file path (caller is responsible for cleanup).
    """
    name = file.filename or "upload"
    ext = os.path.splitext(name)[-1].lower()
    if ext and ext not in ALLOWED_EXT:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXT))}",
        )

    content = await file.read()
    size = len(content)
    if size == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if size > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size / 1024 / 1024:.1f} MB). Maximum is {MAX_UPLOAD_MB} MB.",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext or ".wav") as tmp:
        tmp.write(content)
        log.info("Accepted upload '%s' (%.1f KB) -> %s", name, size / 1024, tmp.name)
        return tmp.name


class ReportPayload(BaseModel):
    filename: str = "analysis.wav"
    classification: dict
    key_features: dict
    mfccs: list[float] = []
    transcript_segments: list[dict] = []
    explanation: dict = None


# ── app ──────────────────────────────────────────────────────
app = FastAPI(
    title="SpeakSense API",
    description="Automated Speech Diagnostic Engine — AblePro × BIG Foundation Hackathon",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup():
    log.info("SpeakSense API starting. Capabilities: %s", capabilities())


@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-ms"] = f"{elapsed_ms:.0f}"
    if request.url.path not in ("/health", "/"):
        log.info("%s %s -> %s (%.0f ms)", request.method, request.url.path, response.status_code, elapsed_ms)
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    log.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


@app.get("/")
def root():
    return {
        "name": "SpeakSense",
        "description": "Automated Speech Diagnostic Engine",
        "version": "1.1.0",
        "endpoints": ["/analyze", "/diarize", "/transcribe", "/report", "/features/catalogue", "/health"],
        "capabilities": capabilities(),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "total_features": TOTAL_FEATURES,
        "capabilities": capabilities(),
        "limits": {"max_upload_mb": MAX_UPLOAD_MB},
    }


@app.get("/features/catalogue")
def feature_catalogue():
    """Return the full acoustic feature catalogue (Task 3)."""
    return {
        "total_features": TOTAL_FEATURES,
        "categories": FEATURE_CATALOGUE,
    }


@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    """
    Upload a WAV file and get acoustic features + classifications
    (age, gender, typicality). Diarization requires HUGGINGFACE_TOKEN.
    """
    tmp_path = await save_upload(file)
    try:
        features = extract_all_features(tmp_path)
        classification = classify_speaker(features)
        explanation = explain_all(features, classification)

        key_features = {
            "f0_mean_hz":         round(features.get("f0_mean_hz", 0), 2),
            "f0_std_hz":          round(features.get("f0_std_hz", 0), 2),
            "jitter_local_pct":   round(features.get("jitter_local_pct", 0), 4),
            "shimmer_local_db":   round(features.get("shimmer_local_db", 0), 4),
            "hnr_mean_db":        round(features.get("hnr_mean_db", 0), 2),
            "voiced_fraction":    round(features.get("voiced_fraction", 0), 4),
            "pause_count":        int(features.get("pause_count", 0)),
            "speech_rate_approx": round(features.get("speech_rate_approx", 0), 4),
            "f1_mean_hz":         round(features.get("f1_mean_hz", 0), 2),
            "f2_mean_hz":         round(features.get("f2_mean_hz", 0), 2),
            "spectral_centroid_mean": round(features.get("spectral_centroid_mean", 0), 2),
        }
        mfccs = [round(features.get(f"mfcc_{i}_mean", 0), 4) for i in range(1, 14)]

        return JSONResponse({
            "filename": file.filename,
            "classification": {
                "age": classification["age"],
                "gender": classification["gender"],
                "typicality": classification["typicality"],
                "summary": classification["summary"],
            },
            "explanation": explanation,
            "key_features": key_features,
            "mfccs": mfccs,
            "model_note": (
                "Predictions use trained ensemble models when available. "
                "If model files are missing, SpeakSense falls back to transparent clinical heuristics."
            ),
            "total_features_extracted": TOTAL_FEATURES,
        })
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Feature extraction / classification failed")
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        _cleanup(tmp_path)


@app.post("/diarize")
async def diarize_audio_endpoint(
    file: UploadFile = File(...),
    min_speakers: int = 1,
    max_speakers: int = 5,
):
    """
    Speaker diarization (Task 1). Identifies and labels unique speakers
    with timestamps. Requires torch + pyannote.audio and a HUGGINGFACE_TOKEN.
    """
    if not HAS_DIARIZATION:
        return JSONResponse(
            {"error": "Diarization dependencies not installed (torch + pyannote.audio)."},
            status_code=501,
        )
    if not HF_TOKEN:
        return JSONResponse(
            {"error": "HUGGINGFACE_TOKEN is not set. Diarization requires a Hugging Face access token "
                      "with access to pyannote/speaker-diarization-3.1."},
            status_code=503,
        )
    if min_speakers < 1 or max_speakers < min_speakers:
        raise HTTPException(status_code=422, detail="Require 1 <= min_speakers <= max_speakers.")

    tmp_path = await save_upload(file)
    try:
        from diarization import diarize_audio, format_diarization_output
        pipeline = get_diarizer()
        segments = diarize_audio(tmp_path, pipeline, min_speakers=min_speakers, max_speakers=max_speakers)
        speakers = sorted({s["speaker"] for s in segments})
        return JSONResponse({
            "filename": file.filename,
            "num_speakers": len(speakers),
            "speakers": speakers,
            "segments": segments,
            "timeline": format_diarization_output(segments),
        })
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Diarization failed")
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        _cleanup(tmp_path)


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...), language: str = "kn"):
    """
    Transcribe audio to text (default Kannada) using Whisper (Task 2).
    Requires faster-whisper.
    """
    if not HAS_WHISPER:
        return JSONResponse({"error": "faster-whisper is not installed."}, status_code=501)

    tmp_path = await save_upload(file)
    try:
        from transcription import transcribe_segment
        model = get_whisper()
        chunks = transcribe_segment(tmp_path, model, language=language)
        full_text = " ".join(c["text"] for c in chunks)
        return JSONResponse({
            "filename": file.filename,
            "language": language,
            "transcription": full_text,
            "segments": chunks,
        })
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Transcription failed")
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        _cleanup(tmp_path)


@app.post("/report")
async def generate_pdf_report(payload: ReportPayload):
    """Create a downloadable diagnostic PDF from the current analysis result."""
    if not HAS_REPORT:
        return JSONResponse(
            {"error": "Report dependencies are missing (reportlab). Install with: pip install reportlab"},
            status_code=501,
        )
    try:
        from report import generate_report
        from explainability import explain_all

        features = dict(payload.key_features)
        features["_total_features"] = TOTAL_FEATURES
        classification = payload.classification
        
        if payload.explanation:
            explanation = payload.explanation
        else:
            explanation = explain_all(features, classification)

        pdf_path = generate_report(
            features=features,
            classification=classification,
            explanation=explanation,
            transcript_segments=payload.transcript_segments,
            audio_filename=payload.filename,
        )
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=os.path.basename(pdf_path),
        )
    except Exception as e:
        log.exception("Report generation failed")
        return JSONResponse({"error": str(e)}, status_code=500)


# ── training status ──────────────────────────────────────────
import threading
training_status = {"status": "idle", "message": "Console initialized.", "progress": 1.0}

def train_worker():
    global training_status
    training_status = {"status": "running", "message": "Initializing training pipeline...", "progress": 0.1}
    try:
        from pipeline import run_training_pipeline
        from preprocess import preprocess_dataset
        training_status = {"status": "running", "message": "Step 1/2: Preprocessing raw files...", "progress": 0.3}
        preprocess_dataset("datasets/audio_samples", "datasets/processed")
        
        training_status = {"status": "running", "message": "Step 2/2: Extracting acoustic features & training classifiers...", "progress": 0.6}
        run_training_pipeline("datasets/processed")
        training_status = {"status": "idle", "message": "Models trained successfully!", "progress": 1.0}
    except Exception as e:
        log.exception("Asynchronous training failed")
        training_status = {"status": "error", "message": f"Training failed: {str(e)}", "progress": 0.0}

def augment_worker():
    global training_status
    training_status = {"status": "running", "message": "Initializing dataset augmentation...", "progress": 0.1}
    try:
        from augment import augment_dataset
        from preprocess import preprocess_dataset
        import shutil
        
        training_status = {"status": "running", "message": "Step 1/3: Normalizing raw dataset...", "progress": 0.3}
        preprocess_dataset("datasets/audio_samples", "datasets/processed")
        
        training_status = {"status": "running", "message": "Step 2/3: Applying Pitch Shift, Time Stretch, and Gaussian Noise (8x expansion)...", "progress": 0.6}
        augment_dataset("datasets/processed", "datasets/augmented")
        
        training_status = {"status": "running", "message": "Step 3/3: Deploying augmented samples to processed directory...", "progress": 0.8}
        for f in os.listdir("datasets/augmented"):
            if f.lower().endswith(".wav"):
                shutil.copy(os.path.join("datasets/augmented", f), os.path.join("datasets/processed", f))
                
        training_status = {"status": "idle", "message": "Dataset augmented successfully! Active training pool expanded.", "progress": 1.0}
    except Exception as e:
        log.exception("Asynchronous augmentation failed")
        training_status = {"status": "error", "message": f"Augmentation failed: {str(e)}", "progress": 0.0}

@app.post("/train/run")
def trigger_training():
    global training_status
    if training_status["status"] == "running":
        raise HTTPException(status_code=409, detail="A training or augmentation process is already active.")
    
    t = threading.Thread(target=train_worker)
    t.start()
    return {"message": "Classifier training initiated."}

@app.post("/augment/run")
def trigger_augmentation():
    global training_status
    if training_status["status"] == "running":
        raise HTTPException(status_code=409, detail="A training or augmentation process is already active.")
    
    t = threading.Thread(target=augment_worker)
    t.start()
    return {"message": "Dataset augmentation initiated."}

@app.get("/train/status")
def get_training_status():
    return training_status

@app.get("/train/info")
def get_train_info():
    processed_count = 0
    augmented_count = 0
    samples_count = 0
    
    if os.path.exists("datasets/audio_samples"):
        samples_count = len([f for f in os.listdir("datasets/audio_samples") if f.endswith(".wav")])
    if os.path.exists("datasets/processed"):
        processed_count = len([f for f in os.listdir("datasets/processed") if f.endswith(".wav")])
    if os.path.exists("datasets/augmented"):
        augmented_count = len([f for f in os.listdir("datasets/augmented") if f.endswith(".wav")])
        
    models_status = {}
    for task in ["age_classification", "gender_classification", "typicality_classification"]:
        model_path = os.path.join("models", f"{task}_model.pkl")
        exists = os.path.exists(model_path)
        stats = {}
        if exists:
            try:
                import joblib
                saved = joblib.load(model_path)
                stats["classes"] = saved.get("classes", [])
                stats["features"] = len(saved.get("feature_cols", []))
            except Exception:
                pass
        models_status[task] = {
            "exists": exists,
            "stats": stats
        }
        
    return {
        "samples_count": samples_count,
        "processed_count": processed_count,
        "augmented_count": augmented_count,
        "models": models_status
    }

@app.post("/transcribe/diarized")
async def transcribe_diarized_audio(file: UploadFile = File(...), language: str = "kn"):
    """
    Unified diarized transcription.
    Returns speaker-labeled text.
    """
    if not HAS_WHISPER:
        return JSONResponse({"error": "faster-whisper is not installed."}, status_code=501)
        
    tmp_path = await save_upload(file)
    try:
        whisper_model = get_whisper()
        
        # Fallback if diarization is not available
        if not HAS_DIARIZATION or not HF_TOKEN:
            from transcription import transcribe_segment
            chunks = transcribe_segment(tmp_path, whisper_model, language=language)
            segments = []
            for c in chunks:
                segments.append({
                    "speaker": "speaker-1",
                    "start": c["start"],
                    "end": c["end"],
                    "transcription": c["text"],
                    "duration": round(c["end"] - c["start"], 2)
                })
            return JSONResponse({
                "filename": file.filename,
                "segments": segments,
                "diarized": False,
                "num_speakers": 1
            })
            
        from diarization import diarize_audio
        from transcription import transcribe_all_speakers
        
        diar_pipeline = get_diarizer()
        
        # Diarize and transcribe
        diar_segments = diarize_audio(tmp_path, diar_pipeline)
        enriched = transcribe_all_speakers(diar_segments, whisper_model)
        
        speakers = sorted({s["speaker"] for s in enriched})
        
        # Cleanup segments
        for s in enriched:
            fpath = s.get("segment_file")
            if fpath and os.path.exists(fpath):
                try:
                    os.unlink(fpath)
                except Exception:
                    pass
                    
        return JSONResponse({
            "filename": file.filename,
            "segments": enriched,
            "diarized": True,
            "num_speakers": len(speakers),
            "speakers": speakers
        })
    except Exception as e:
        log.exception("Diarized transcription failed")
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        _cleanup(tmp_path)


def _cleanup(path: str):
    try:
        if path and os.path.exists(path):
            os.unlink(path)
    except OSError as e:
        log.warning("Could not delete temp file %s: %s", path, e)
