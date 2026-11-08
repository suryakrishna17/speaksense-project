"""
classification.py
─────────────────
Task 4: Classify speech samples using ML ensembles.

Three binary classifiers trained on acoustic features:
  (a) Adult (age > 18) vs Child (age < 15)
  (b) Male vs Female
  (c) Typical vs Atypical speech

Each classifier uses an ensemble of:
  - Random Forest
  - XGBoost
  - Support Vector Machine (RBF kernel)
  Voting: soft probability average → final prediction
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# FEATURE COLUMNS used per task
# ─────────────────────────────────────────────────────────────

# Adult/Child: pitch is the most discriminative + formants + spectral
AGE_FEATURES = [
    "f0_mean_hz", "f0_std_hz", "f0_min_hz", "f0_max_hz",
    "f1_mean_hz", "f2_mean_hz", "f3_mean_hz",
    "spectral_centroid_mean", "spectral_centroid_std",
    "spectral_bandwidth_mean",
    *[f"mfcc_{i}_mean" for i in range(1, 14)],
    *[f"mfcc_{i}_std" for i in range(1, 14)],
]

# Male/Female: pitch + formants + spectral + MFCCs
GENDER_FEATURES = [
    "f0_mean_hz", "f0_std_hz", "f0_min_hz", "f0_max_hz", "f0_range_hz",
    "f1_mean_hz", "f2_mean_hz",
    "spectral_centroid_mean", "zcr_mean",
    *[f"mfcc_{i}_mean" for i in range(1, 14)],
]

# Typical/Atypical: voice quality + prosody + temporal
ATYPICAL_FEATURES = [
    "jitter_local_pct", "jitter_rap_pct", "jitter_ppq5_pct",
    "shimmer_local_db", "shimmer_apq3_db", "shimmer_apq5_db",
    "hnr_mean_db",
    "voiced_fraction", "pause_count", "speech_rate_approx",
    "f0_std_hz", "f0_range_hz",
    "rms_std",
    *[f"mfcc_{i}_mean" for i in range(1, 8)],
]


def build_ensemble(task_name: str, random_state: int = 42) -> Pipeline:
    """Build a soft-voting ensemble wrapped in a StandardScaler pipeline."""
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        min_samples_split=5, random_state=random_state, n_jobs=-1
    )
    xgb = XGBClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=random_state,
        use_label_encoder=False
    )
    svm = SVC(
        kernel="rbf", C=1.0, gamma="scale",
        probability=True, random_state=random_state
    )

    voting = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb), ("svm", svm)],
        voting="soft"
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ensemble", voting)
    ])
    return pipeline


def train_classifier(
    features_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    task_name: str,
    cv_folds: int = 5
) -> dict:
    """
    Train ensemble classifier and evaluate with cross-validation.

    Returns dict with model, scores, and classification report.
    """
    # Filter valid rows
    valid = features_df[label_col].notna() & features_df[label_col].ne("unknown")
    df = features_df[valid].copy()

    # Use only columns that exist
    available_cols = [c for c in feature_cols if c in df.columns]
    X = df[available_cols].fillna(0).values
    y = df[label_col].values
    classes, counts = np.unique(y, return_counts=True)

    print(f"\n{'-'*50}")
    print(f"Training: {task_name}")
    print(f"  Samples: {len(X)} | Classes: {classes}")
    print(f"  Features: {len(available_cols)}")

    if len(classes) < 2:
        print(f"  SKIPPED: {task_name} needs at least 2 classes, found {classes.tolist()}")
        return {
            "task": task_name,
            "skipped": True,
            "reason": "Need at least 2 classes for supervised training",
            "classes": classes.tolist(),
            "model_path": None,
            "cv_f1_mean": 0.0,
            "cv_f1_std": 0.0,
        }

    min_class_count = int(np.min(counts))
    if min_class_count < 2:
        print(f"  SKIPPED: smallest class has only {min_class_count} sample")
        return {
            "task": task_name,
            "skipped": True,
            "reason": "Need at least 2 samples per class for cross-validation",
            "classes": classes.tolist(),
            "model_path": None,
            "cv_f1_mean": 0.0,
            "cv_f1_std": 0.0,
        }

    model = build_ensemble(task_name)

    # Cross-validation
    cv = StratifiedKFold(n_splits=min(cv_folds, min_class_count),
                         shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1_weighted")
    print(f"  CV F1 (weighted): {scores.mean():.3f} ± {scores.std():.3f}")

    # Fit on full data
    model.fit(X, y)

    # Save model
    model_path = os.path.join(MODELS_DIR, f"{task_name}_model.pkl")
    joblib.dump({
        "model": model,
        "feature_cols": available_cols,
        "classes": model.classes_.tolist() if hasattr(model, "classes_") else []
    }, model_path)
    print(f"  Saved -> {model_path}")

    return {
        "task": task_name,
        "model": model,
        "feature_cols": available_cols,
        "cv_f1_mean": float(scores.mean()),
        "cv_f1_std": float(scores.std()),
        "model_path": model_path
    }


def predict_single(
    features: dict,
    task_name: str,
    model_path: str = None
) -> dict:
    """
    Predict class for a single feature vector.

    Returns: {"prediction": "adult", "confidence": 0.87, "probabilities": {...}}
    """
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, f"{task_name}_model.pkl")

    if not os.path.exists(model_path):
        return {
            "prediction": "unknown",
            "confidence": 0.0,
            "probabilities": {},
            "error": "Model not trained yet"
        }

    saved = joblib.load(model_path)
    model = saved["model"]
    feature_cols = saved["feature_cols"]
    classes = saved.get("classes", [])

    X = np.array([[features.get(c, 0.0) for c in feature_cols]])
    proba = model.predict_proba(X)[0]
    pred_idx = np.argmax(proba)
    prediction = classes[pred_idx] if classes else str(pred_idx)

    return {
        "prediction": prediction,
        "confidence": round(float(proba[pred_idx]), 4),
        "probabilities": {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
    }


def heuristic_classification(features: dict) -> dict:
    """Transparent fallback when trained model files are not available."""
    f0 = features.get("f0_mean_hz", 0)
    f1 = features.get("f1_mean_hz", 0)
    f2 = features.get("f2_mean_hz", 0)
    jitter = features.get("jitter_local_pct", 0)
    shimmer = features.get("shimmer_local_db", 0)
    hnr = features.get("hnr_mean_db", 99)
    voiced = features.get("voiced_fraction", 0)
    pauses = features.get("pause_count", 0)

    age_score = 0
    if f0 >= 240:
        age_score += 2
    elif f0 >= 210:
        age_score += 1
    if f1 >= 650:
        age_score += 1
    if f2 >= 1700:
        age_score += 1
    age_prediction = "child" if age_score >= 2 else "adult"
    age_confidence = min(0.9, 0.58 + age_score * 0.08)

    if age_prediction == "child":
        gender_prediction = "unknown"
        gender_confidence = 0.5
    elif f0 >= 165:
        gender_prediction = "female"
        gender_confidence = 0.72 if f0 >= 190 else 0.62
    elif f0 > 0:
        gender_prediction = "male"
        gender_confidence = 0.74 if f0 <= 145 else 0.64
    else:
        gender_prediction = "unknown"
        gender_confidence = 0.0

    atypical_flags = [
        jitter > 1.04,
        shimmer > 0.35,
        hnr < 15,
        0 < voiced < 0.55,
        pauses >= 30,
    ]
    atypical_score = sum(atypical_flags)
    typicality_prediction = "atypical" if atypical_score > 0 else "typical"
    typicality_confidence = min(0.92, 0.62 + atypical_score * 0.08)

    return {
        "age": {
            "prediction": age_prediction,
            "confidence": round(age_confidence, 4),
            "probabilities": {
                "adult": round(1 - age_confidence, 4) if age_prediction == "child" else round(age_confidence, 4),
                "child": round(age_confidence, 4) if age_prediction == "child" else round(1 - age_confidence, 4),
            },
            "source": "clinical heuristic fallback"
        },
        "gender": {
            "prediction": gender_prediction,
            "confidence": round(gender_confidence, 4),
            "probabilities": {},
            "source": "pitch/formant heuristic fallback"
        },
        "typicality": {
            "prediction": typicality_prediction,
            "confidence": round(typicality_confidence, 4),
            "probabilities": {},
            "source": "clinical threshold fallback"
        }
    }


def classify_speaker(features: dict) -> dict:
    """
    Run all three classifiers on a feature dict.
    Returns combined classification result.
    """
    age_result      = predict_single(features, "age_classification")
    gender_result   = predict_single(features, "gender_classification")
    atypical_result = predict_single(features, "typicality_classification")
    fallback = heuristic_classification(features)

    if age_result["prediction"] == "unknown":
        age_result = fallback["age"]
    if gender_result["prediction"] == "unknown":
        gender_result = fallback["gender"]
    if atypical_result["prediction"] == "unknown":
        atypical_result = fallback["typicality"]

    # Rule-based refinement for typicality
    # Clinical thresholds from literature
    jitter   = features.get("jitter_local_pct", 0)
    shimmer  = features.get("shimmer_local_db", 0)
    hnr      = features.get("hnr_mean_db", 99)

    rule_atypical = (jitter > 1.04 or shimmer > 0.35 or hnr < 15)
    if rule_atypical and atypical_result["prediction"] == "typical":
        atypical_result["prediction"] = "atypical"
        atypical_result["rule_override"] = True
        atypical_result["reason"] = f"Jitter={jitter:.3f}%, Shimmer={shimmer:.3f}dB, HNR={hnr:.1f}dB"

    return {
        "age":       age_result,
        "gender":    gender_result,
        "typicality": atypical_result,
        "summary": (
            f"{age_result['prediction'].upper()} | "
            f"{gender_result['prediction'].upper()} | "
            f"{atypical_result['prediction'].upper()}"
        )
    }


def train_all(features_df: pd.DataFrame) -> list[dict]:
    """Train all three classifiers from a features DataFrame."""
    results = []

    # Task 4a: Adult vs Child
    if "age_label" in features_df.columns:
        r = train_classifier(features_df, AGE_FEATURES, "age_label", "age_classification")
        results.append(r)

    # Task 4b: Male vs Female
    if "gender_label" in features_df.columns:
        r = train_classifier(features_df, GENDER_FEATURES, "gender_label", "gender_classification")
        results.append(r)

    # Task 4c: Typical vs Atypical
    # Rule-based labeling using clinical thresholds since dataset may not have explicit labels
    if "jitter_local_pct" in features_df.columns:
        features_df["typicality_label"] = features_df.apply(
            lambda row: "atypical" if (
                row.get("jitter_local_pct", 0) > 1.04 or
                row.get("shimmer_local_db", 0) > 0.35 or
                row.get("hnr_mean_db", 99) < 15
            ) else "typical",
            axis=1
        )
        r = train_classifier(features_df, ATYPICAL_FEATURES, "typicality_label", "typicality_classification")
        results.append(r)

    return results


if __name__ == "__main__":
    # Quick test with dummy data
    dummy = {
        "f0_mean_hz": 250.0, "f0_std_hz": 30.0,
        "f1_mean_hz": 800.0, "f2_mean_hz": 1800.0,
        "jitter_local_pct": 0.5, "hnr_mean_db": 22.0,
        **{f"mfcc_{i}_mean": float(i) for i in range(1, 14)}
    }
    print("Test prediction:", classify_speaker(dummy))
