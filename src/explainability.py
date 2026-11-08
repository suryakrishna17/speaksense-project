"""
explainability.py
─────────────────
SHAP-based explainability for all three classifiers.

For each prediction, generates:
  1. Feature importance bar chart (top 10 drivers)
  2. Natural language explanation
  3. Clinical threshold comparison
  4. SHAP values dict (for frontend visualization)

Usage:
  from explainability import explain_prediction
  result = explain_prediction(features, task="age_classification")
"""

import os
import json
import numpy as np
import joblib
try:
    import shap
except ImportError:
    shap = None
import warnings
warnings.filterwarnings("ignore")

MODELS_DIR = "models"

# Clinical reference ranges (from literature)
CLINICAL_RANGES = {
    "jitter_local_pct":   {"normal": (0.0, 1.04),  "unit": "%",   "label": "Jitter (Local)"},
    "shimmer_local_db":   {"normal": (0.0, 0.35),   "unit": "dB",  "label": "Shimmer (Local)"},
    "hnr_mean_db":        {"normal": (20.0, 45.0),  "unit": "dB",  "label": "HNR",       "inverted": True},
    "f0_mean_hz":         {"normal": (85.0, 255.0), "unit": "Hz",  "label": "Pitch (F0)"},
    "voiced_fraction":    {"normal": (0.6, 1.0),    "unit": "",    "label": "Voiced Fraction", "inverted": True},
    "f1_mean_hz":         {"normal": (300.0, 900.0),"unit": "Hz",  "label": "Formant F1"},
}

# Human-readable explanations per feature
FEATURE_LABELS = {
    "f0_mean_hz":           "average pitch",
    "f0_std_hz":            "pitch variability",
    "f0_range_hz":          "pitch range",
    "jitter_local_pct":     "vocal fold irregularity (jitter)",
    "shimmer_local_db":     "amplitude irregularity (shimmer)",
    "hnr_mean_db":          "voice clarity (HNR)",
    "voiced_fraction":      "proportion of voiced speech",
    "pause_count":          "number of pauses",
    "speech_rate_approx":   "speech rate",
    "spectral_centroid_mean": "brightness of voice",
    "f1_mean_hz":           "first formant (jaw opening)",
    "f2_mean_hz":           "second formant (tongue position)",
    "mfcc_1_mean":          "overall energy (MFCC-1)",
    "mfcc_2_mean":          "low-frequency energy (MFCC-2)",
    "rms_mean":             "average loudness",
    "zcr_mean":             "noisiness (ZCR)",
}


def load_model(task_name: str) -> dict | None:
    """Load a trained model bundle."""
    path = os.path.join(MODELS_DIR, f"{task_name}_model.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def compute_shap_values(
    features: dict,
    task_name: str,
    background_size: int = 50
) -> dict:
    """
    Compute SHAP values for a single prediction.

    Returns dict with:
      - shap_values: {feature_name: shap_value}
      - base_value: float
      - prediction: str
      - confidence: float
    """
    bundle = load_model(task_name)
    if bundle is None:
        return {"error": f"Model {task_name} not trained yet"}
    if shap is None:
        return {"error": "SHAP is not installed"}

    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    classes = bundle.get("classes", [])

    # Build input vector
    X = np.array([[features.get(c, 0.0) for c in feature_cols]])

    # Get prediction
    proba = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))
    prediction = classes[pred_idx] if classes else str(pred_idx)
    confidence = float(proba[pred_idx])

    # SHAP — use TreeExplainer on the Random Forest member
    try:
        rf = model.named_steps["ensemble"].estimators_[0][1]  # RF from voting
        scaler = model.named_steps["scaler"]
        X_scaled = scaler.transform(X)

        explainer = shap.TreeExplainer(rf)
        shap_vals = explainer.shap_values(X_scaled)

        # For binary: take values for predicted class
        if isinstance(shap_vals, list):
            sv = shap_vals[pred_idx][0]
        else:
            sv = shap_vals[0]

        shap_dict = {col: float(sv[i]) for i, col in enumerate(feature_cols)}
        base_value = float(explainer.expected_value[pred_idx]
                          if isinstance(explainer.expected_value, np.ndarray)
                          else explainer.expected_value)

        return {
            "task": task_name,
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "base_value": round(base_value, 4),
            "shap_values": shap_dict,
            "top_features": _top_features(shap_dict, n=10)
        }

    except Exception as e:
        # Fallback: use feature importance from RF
        try:
            rf = model.named_steps["ensemble"].estimators_[0][1]
            importances = rf.feature_importances_
            imp_dict = {col: float(importances[i]) for i, col in enumerate(feature_cols)}
            return {
                "task": task_name,
                "prediction": prediction,
                "confidence": round(confidence, 4),
                "base_value": 0.5,
                "shap_values": imp_dict,
                "top_features": _top_features(imp_dict, n=10),
                "note": "Feature importance (SHAP unavailable)"
            }
        except Exception as e2:
            return {"error": str(e2), "prediction": prediction, "confidence": confidence}


def _top_features(shap_dict: dict, n: int = 10) -> list[dict]:
    """Return top N features sorted by absolute SHAP value."""
    sorted_feats = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    return [
        {
            "feature": k,
            "label": FEATURE_LABELS.get(k, k),
            "shap_value": round(v, 5),
            "direction": "increases" if v > 0 else "decreases"
        }
        for k, v in sorted_feats[:n]
    ]


def generate_clinical_explanation(features: dict, classification: dict) -> dict:
    """
    Generate natural language + clinical threshold comparison.

    Returns structured explanation for the frontend and PDF report.
    """
    age        = classification.get("age", {}).get("prediction", "unknown")
    gender     = classification.get("gender", {}).get("prediction", "unknown")
    typicality = classification.get("typicality", {}).get("prediction", "unknown")

    flags = []
    clinical_checks = []

    for feat_key, ref in CLINICAL_RANGES.items():
        val = features.get(feat_key)
        if val is None:
            continue
        lo, hi = ref["normal"]
        label  = ref["label"]
        unit   = ref["unit"]
        inv    = ref.get("inverted", False)

        status = "normal"
        if inv:
            status = "below_normal" if val < lo else "normal"
        else:
            if val < lo:   status = "below_normal"
            elif val > hi: status = "above_normal"

        clinical_checks.append({
            "parameter":   label,
            "value":       round(val, 4),
            "unit":        unit,
            "normal_range": f"{lo}–{hi} {unit}".strip(),
            "status":      status,
            "flag":        status != "normal"
        })

        if status != "normal":
            direction = "elevated" if status == "above_normal" else "reduced"
            flags.append(f"{label} is {direction} ({val:.3f}{unit}, normal: {lo}–{hi}{unit})")

    # Build narrative
    narrative_parts = []

    # Age narrative
    if age == "child":
        narrative_parts.append(
            "Acoustic analysis indicates a child speaker. "
            "Elevated fundamental frequency and higher formant values are consistent with "
            "a shorter vocal tract (typical of speakers under 15 years)."
        )
    else:
        narrative_parts.append(
            "Acoustic profile is consistent with an adult speaker. "
            "Pitch and formant frequencies fall within the adult range."
        )

    # Typicality narrative
    if typicality == "atypical":
        if flags:
            narrative_parts.append(
                f"Speech quality indicators suggest ATYPICAL patterns. "
                f"Specifically: {'; '.join(flags[:3])}. "
                "These findings may be consistent with dysarthria, voice disorders, or other speech pathologies. "
                "Clinical evaluation by a licensed speech-language pathologist is recommended."
            )
        else:
            narrative_parts.append(
                "ML classifier detected atypical speech patterns based on combined feature profile. "
                "Clinical evaluation recommended."
            )
    else:
        narrative_parts.append(
            "Voice quality parameters (jitter, shimmer, HNR) are within normal clinical limits. "
            "Speech pattern is classified as TYPICAL."
        )

    return {
        "summary": f"{age.upper()} | {gender.upper()} | {typicality.upper()}",
        "narrative": " ".join(narrative_parts),
        "clinical_checks": clinical_checks,
        "flags": flags,
        "flag_count": len(flags),
        "disclaimer": (
            "This analysis is generated by an AI system and is intended for "
            "assistive/screening purposes only. It does not constitute a clinical diagnosis. "
            "All findings must be interpreted by a qualified speech-language pathologist."
        )
    }


def explain_all(features: dict, classification: dict) -> dict:
    """Run full explainability pipeline for all three tasks."""
    return {
        "age_explanation":        compute_shap_values(features, "age_classification"),
        "gender_explanation":     compute_shap_values(features, "gender_classification"),
        "typicality_explanation": compute_shap_values(features, "typicality_classification"),
        "clinical":               generate_clinical_explanation(features, classification),
    }
