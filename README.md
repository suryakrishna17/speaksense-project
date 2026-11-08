# SpeakSense: Automated Speech Diagnostic Engine

SpeakSense is an AI-assisted speech analysis, speaker diarization, Kannada ASR transcription, and clinical speech classification platform built for the **AblePro Solutions x BIG Foundation Hackathon**.

This repository contains the complete working system, architectural documentation, and evaluation deliverables.

---

## 🚀 Key Hackathon Deliverables

This repository contains all 5 required deliverables:

1. **Source Code**: Fully modular, well-documented Python/FastAPI backend and modern React dashboard under the `speaksense` directory.
2. **Working POC**: End-to-end command-line and REST API system that processes audio data, normalizes formats, transcribes Kannada speakers, and executes classifications.
3. **Visualization Output**: Script for diagnostic plots showing waveforms, fundamental frequency ($F_0$ / pitch), loudness intensity profiles, and MFCC spectrograms.
   * Tool: `src/visualize.py`
4. **Technical Documentation**: Deep dive into acoustic engineering, feature selection, and voting ensemble justifications.
   * See [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md).
5. **Ensemble Classifiers & Heuristics**: Soft-voting ensemble (Random Forest + XGBoost + SVM) with deterministic clinical perturbation fallbacks.

---

## 🛠️ Project Architecture

```text
speaksense/
  src/
    api.py                 FastAPI backend (manages CORS, uploads, endpoints, and .env loading)
    pipeline.py            CLI runner (processes single files or trains models)
    preprocess.py          Decibel-level amplitude and sampling normalization
    diarization.py         pyannote.audio speaker segmentation
    transcription.py       faster-whisper Kannada ASR segmenter
    feature_extraction.py  36-dim Parselmouth & Librosa feature extractor
    classification.py      RF + XGBoost + SVM Voting Ensemble logic
    report.py              ReportLab PDF diagnostic report generator
    visualize.py           Plotting script for pitch, intensity, and MFCCs
  docs/
    acoustic_parameters.md Detailed descriptions of vocal features
  models/                  Saved ML classifier model weights (.pkl)
  outputs/                 Diagnostic reports, feature CSVs, and visualization outputs
```

---

## 🏃 Setup & Execution Guide

### Prerequisites
* Python 3.10 or newer (tested on 3.13)
* Node.js 18 or newer

### 1. Backend Setup & Run
Create a virtual environment, install dependencies, and run the API server:
```bash
# Navigate to project root
cd speaksense

# Install dependencies
pip install -r requirements.txt

# Start FastAPI server
uvicorn src.api:app --reload --port 8000
```
API endpoints are interactive at: `http://localhost:8000/docs`.

### 2. Frontend Setup & Run
Run the interactive React dashboard:
```bash
cd frontend
npm install
npm run dev
```
Open: `http://localhost:5173`.

---

## 📊 Running Tasks via CLI

### Run End-to-End Inference (Diarization, ASR, Feature Extraction, Classification, and Reports):
```bash
# Runs full pipeline and generates output artifacts under outputs/
python src/pipeline.py --file sample.wav
```

### Generate Pitch and Intensity Visual Insights:
```bash
# Generates pitch, intensity, and MFCC plots
python src/visualize.py
```

### Run Model Training:
```bash
# Trains models using files under datasets/processed/
python src/pipeline.py --train --data datasets/processed
```

---

## 💡 Modeling & Algorithms Rationale

* **ASR & Diarization**: Combines the highly efficient `faster-whisper` transformer (configured for Kannada `kn`) with `pyannote/speaker-diarization-3.1` to align translations with corresponding speaker labels.
* **ML Soft-Voting Ensemble**: Combines **Random Forest** (decision tree aggregation), **XGBoost** (gradient boosting residuals), and **SVM** (RBF kernel projection). Probability averaging ensures robust and generalized classification on small datasets.
* **Clinical Perturbation Fallback**: Evaluates pitch and perturbation thresholds (Jitter $> 1.04\%$, Shimmer $> 0.35\text{ dB}$, HNR $< 15\text{ dB}$) when machine learning weights are not trained on specific categories, ensuring clinical accuracy.
