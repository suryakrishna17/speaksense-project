"""
report.py
─────────
Generates a professional PDF diagnostic report using ReportLab.
Covers all 4 hackathon tasks in one downloadable document.

Usage:
  from report import generate_report
  pdf_path = generate_report(features, classification, explanation, transcript, audio_filename)
"""

import os
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)

# ── Brand colors ────────────────────────────────────────────
C_DARK    = colors.HexColor("#0f172a")
C_NAVY    = colors.HexColor("#1e293b")
C_INDIGO  = colors.HexColor("#4f46e5")
C_GREEN   = colors.HexColor("#059669")
C_RED     = colors.HexColor("#dc2626")
C_AMBER   = colors.HexColor("#d97706")
C_MUTED   = colors.HexColor("#64748b")
C_LIGHT   = colors.HexColor("#e2e8f0")
C_WHITE   = colors.white


def _styles():
    base = getSampleStyleSheet()
    custom = {
        "title": ParagraphStyle("title",
            fontSize=22, fontName="Helvetica-Bold",
            textColor=C_INDIGO, spaceAfter=4, alignment=TA_LEFT),
        "subtitle": ParagraphStyle("subtitle",
            fontSize=11, fontName="Helvetica",
            textColor=C_MUTED, spaceAfter=16, alignment=TA_LEFT),
        "section": ParagraphStyle("section",
            fontSize=13, fontName="Helvetica-Bold",
            textColor=C_DARK, spaceBefore=14, spaceAfter=6),
        "body": ParagraphStyle("body",
            fontSize=10, fontName="Helvetica",
            textColor=C_DARK, spaceAfter=6, leading=15),
        "small": ParagraphStyle("small",
            fontSize=8.5, fontName="Helvetica",
            textColor=C_MUTED, spaceAfter=4),
        "disclaimer": ParagraphStyle("disclaimer",
            fontSize=8, fontName="Helvetica-Oblique",
            textColor=C_MUTED, spaceAfter=4, leading=11),
        "badge_normal":   ParagraphStyle("badge_normal",
            fontSize=10, fontName="Helvetica-Bold",
            textColor=C_GREEN, alignment=TA_CENTER),
        "badge_atypical": ParagraphStyle("badge_atypical",
            fontSize=10, fontName="Helvetica-Bold",
            textColor=C_RED, alignment=TA_CENTER),
        "badge_label":    ParagraphStyle("badge_label",
            fontSize=8, fontName="Helvetica",
            textColor=C_MUTED, alignment=TA_CENTER),
        "mono": ParagraphStyle("mono",
            fontSize=9, fontName="Courier",
            textColor=C_DARK, spaceAfter=3, leading=13),
    }
    return custom


def _status_color(status: str):
    return {
        "normal":       C_GREEN,
        "above_normal": C_RED,
        "below_normal": C_AMBER,
    }.get(status, C_MUTED)


def generate_report(
    features: dict,
    classification: dict,
    explanation: dict,
    transcript_segments: list[dict] = None,
    audio_filename: str = "sample.wav",
    output_dir: str = "outputs"
) -> str:
    """
    Generate a professional PDF diagnostic report.
    Returns path to the saved PDF.
    """
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(audio_filename)[0]
    pdf_path = os.path.join(output_dir, f"{base}_diagnostic_report.pdf")

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=18*mm, bottomMargin=18*mm
    )

    S = _styles()
    story = []
    W = A4[0] - 40*mm   # usable width

    # ────────────────────────────────────────────────────────
    # HEADER
    # ────────────────────────────────────────────────────────
    story.append(Paragraph("SpeakSense", S["title"]))
    story.append(Paragraph(
        "Automated Speech Diagnostic Engine  |  AblePro x BIG Foundation",
        S["subtitle"]
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=C_INDIGO, spaceAfter=12))

    # Report metadata table
    now = datetime.now().strftime("%B %d, %Y  %H:%M")
    meta_data = [
        ["File", audio_filename, "Report Date", now],
        ["Features Extracted", str(features.get("_total_features", 71)),
         "Model", "Ensemble (RF + XGB + SVM)"],
    ]
    meta_table = Table(meta_data, colWidths=[35*mm, 65*mm, 35*mm, 55*mm])
    meta_table.setStyle(TableStyle([
        ("FONTNAME",  (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE",  (0,0), (-1,-1), 9),
        ("FONTNAME",  (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME",  (2,0), (2,-1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0,0), (-1,-1), C_DARK),
        ("TEXTCOLOR", (0,0), (0,-1), C_MUTED),
        ("TEXTCOLOR", (2,0), (2,-1), C_MUTED),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 12))

    # ────────────────────────────────────────────────────────
    # SECTION 1: CLASSIFICATION RESULTS
    # ────────────────────────────────────────────────────────
    story.append(Paragraph("1. Classification Results", S["section"]))

    age_r   = classification.get("age", {})
    gen_r   = classification.get("gender", {})
    typ_r   = classification.get("typicality", {})

    def conf_str(r): return f"{r.get('confidence', 0)*100:.0f}% confidence"
    def pred_str(r): return r.get("prediction", "unknown").upper()

    badge_data = [
        [
            Paragraph("AGE GROUP", S["badge_label"]),
            Paragraph("GENDER", S["badge_label"]),
            Paragraph("SPEECH TYPE", S["badge_label"]),
        ],
        [
            Paragraph(pred_str(age_r),
                S["badge_normal"] if age_r.get("prediction") == "adult" else S["badge_atypical"]),
            Paragraph(pred_str(gen_r), S["badge_normal"]),
            Paragraph(pred_str(typ_r),
                S["badge_normal"] if typ_r.get("prediction") == "typical" else S["badge_atypical"]),
        ],
        [
            Paragraph(conf_str(age_r), S["small"]),
            Paragraph(conf_str(gen_r), S["small"]),
            Paragraph(conf_str(typ_r), S["small"]),
        ]
    ]

    badge_table = Table(badge_data, colWidths=[W/3]*3)
    badge_table.setStyle(TableStyle([
        ("BOX",          (0,0), (-1,-1), 1, C_LIGHT),
        ("INNERGRID",    (0,0), (-1,-1), 0.5, C_LIGHT),
        ("BACKGROUND",   (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ("ALIGN",        (0,0), (-1,-1), "CENTER"),
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",   (0,0), (-1,-1), 10),
        ("BOTTOMPADDING",(0,0), (-1,-1), 10),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.HexColor("#f1f5f9"), C_WHITE, colors.HexColor("#f8fafc")]),
    ]))
    story.append(KeepTogether([badge_table]))
    story.append(Spacer(1, 10))

    # Clinical narrative
    clinical = explanation.get("clinical", {})
    narrative = clinical.get("narrative", "")
    if narrative:
        story.append(Paragraph(narrative, S["body"]))

    # ────────────────────────────────────────────────────────
    # SECTION 2: CLINICAL PARAMETER ANALYSIS
    # ────────────────────────────────────────────────────────
    story.append(Paragraph("2. Clinical Parameter Analysis", S["section"]))

    checks = clinical.get("clinical_checks", [])
    if checks:
        header = ["Parameter", "Value", "Normal Range", "Status"]
        rows = [header]
        for c in checks:
            status_text = {"normal": "Normal", "above_normal": "Elevated", "below_normal": "Reduced"}.get(c["status"], c["status"])
            rows.append([
                c["parameter"],
                f"{c['value']} {c['unit']}".strip(),
                c["normal_range"],
                status_text
            ])

        col_w = [60*mm, 30*mm, 45*mm, 35*mm]
        checks_table = Table(rows, colWidths=col_w)
        style = [
            ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",     (0,0), (-1,-1), 9),
            ("BACKGROUND",   (0,0), (-1,0), C_INDIGO),
            ("TEXTCOLOR",    (0,0), (-1,0), C_WHITE),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.HexColor("#f8fafc"), C_WHITE]),
            ("GRID",         (0,0), (-1,-1), 0.5, C_LIGHT),
            ("ALIGN",        (1,0), (-1,-1), "CENTER"),
            ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING",   (0,0), (-1,-1), 5),
            ("BOTTOMPADDING",(0,0), (-1,-1), 5),
        ]
        # Color-code status column
        for i, c in enumerate(checks, start=1):
            clr = _status_color(c["status"])
            style.append(("TEXTCOLOR", (3, i), (3, i), clr))
            style.append(("FONTNAME",  (3, i), (3, i), "Helvetica-Bold"))

        checks_table.setStyle(TableStyle(style))
        story.append(checks_table)
    story.append(Spacer(1, 8))

    # ────────────────────────────────────────────────────────
    # SECTION 3: KEY ACOUSTIC FEATURES
    # ────────────────────────────────────────────────────────
    story.append(Paragraph("3. Key Acoustic Features Extracted", S["section"]))

    KEY_DISPLAY = [
        ("f0_mean_hz",          "Fundamental Frequency (F0)",       "Hz"),
        ("f0_std_hz",           "Pitch Variability (F0 Std)",        "Hz"),
        ("jitter_local_pct",    "Jitter (Local)",                    "%"),
        ("shimmer_local_db",    "Shimmer (Local)",                   "dB"),
        ("hnr_mean_db",         "Harmonics-to-Noise Ratio (HNR)",    "dB"),
        ("voiced_fraction",     "Voiced Speech Fraction",            ""),
        ("pause_count",         "Pause Count",                       ""),
        ("speech_rate_approx",  "Speech Rate (approx.)",             "pauses/s"),
        ("f1_mean_hz",          "Formant F1",                        "Hz"),
        ("f2_mean_hz",          "Formant F2",                        "Hz"),
        ("spectral_centroid_mean","Spectral Centroid",               "Hz"),
        ("rms_mean",            "RMS Energy (mean)",                 ""),
    ]

    feat_rows = [["Feature", "Value", "Unit"]]
    for key, label, unit in KEY_DISPLAY:
        val = features.get(key, "N/A")
        feat_rows.append([label, f"{val:.4f}" if isinstance(val, float) else str(val), unit])

    feat_table = Table(feat_rows, colWidths=[90*mm, 40*mm, 40*mm])
    feat_table.setStyle(TableStyle([
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("BACKGROUND",    (0,0), (-1,0), C_NAVY),
        ("TEXTCOLOR",     (0,0), (-1,0), C_WHITE),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.HexColor("#f8fafc"), C_WHITE]),
        ("GRID",          (0,0), (-1,-1), 0.5, C_LIGHT),
        ("ALIGN",         (1,0), (-1,-1), "CENTER"),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(feat_table)
    story.append(Spacer(1, 8))

    # ────────────────────────────────────────────────────────
    # SECTION 4: EXPLAINABILITY - TOP DRIVERS
    # ────────────────────────────────────────────────────────
    story.append(Paragraph("4. Classification Drivers (Feature Importance)", S["section"]))

    for task_key, task_label in [
        ("typicality_explanation", "Typical vs Atypical"),
        ("age_explanation",        "Adult vs Child"),
    ]:
        exp = explanation.get(task_key, {})
        top = exp.get("top_features", [])[:5]
        if not top:
            continue

        story.append(Paragraph(f"{task_label} - Top 5 drivers:", S["body"]))
        rows = [["Feature", "Impact", "Direction"]]
        for f in top:
            rows.append([
                f.get("label", f.get("feature", "")),
                f"{abs(f['shap_value']):.5f}",
                f.get("direction", "")
            ])

        t = Table(rows, colWidths=[90*mm, 35*mm, 45*mm])
        t.setStyle(TableStyle([
            ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",     (0,0), (-1,-1), 9),
            ("BACKGROUND",   (0,0), (-1,0), colors.HexColor("#334155")),
            ("TEXTCOLOR",    (0,0), (-1,0), C_WHITE),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f1f5f9"), C_WHITE]),
            ("GRID",         (0,0), (-1,-1), 0.5, C_LIGHT),
            ("TOPPADDING",   (0,0), (-1,-1), 4),
            ("BOTTOMPADDING",(0,0), (-1,-1), 4),
        ]))
        # Color direction column
        for i, f in enumerate(top, start=1):
            clr = C_RED if f.get("direction") == "increases" else C_GREEN
            t.setStyle(TableStyle([("TEXTCOLOR", (2, i), (2, i), clr)]))

        story.append(t)
        story.append(Spacer(1, 8))

    # ────────────────────────────────────────────────────────
    # SECTION 5: TRANSCRIPT (if provided)
    # ────────────────────────────────────────────────────────
    if transcript_segments:
        story.append(Paragraph("5. Kannada Transcription (with Speaker Labels)", S["section"]))
        for seg in transcript_segments[:20]:  # cap at 20 segments
            spk   = seg.get("speaker", "?")
            start = seg.get("start", 0)
            end   = seg.get("end", 0)
            text  = seg.get("transcription", "")
            if not text:
                continue
            m_s, s_s = divmod(start, 60)
            m_e, s_e = divmod(end, 60)
            line = f"[{spk}]  [{int(m_s):02d}:{s_s:05.2f} -> {int(m_e):02d}:{s_e:05.2f}]"
            story.append(Paragraph(line, S["mono"]))
            story.append(Paragraph(text, S["body"]))
        story.append(Spacer(1, 8))

    # ────────────────────────────────────────────────────────
    # FOOTER - Disclaimer
    # ────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_MUTED, spaceBefore=10))
    disclaimer = clinical.get("disclaimer",
        "This report is generated by an AI system for assistive purposes only. "
        "Not a substitute for clinical evaluation by a qualified speech-language pathologist.")
    story.append(Paragraph(f"Note: {disclaimer}", S["disclaimer"]))
    story.append(Paragraph(
        f"Generated by SpeakSense v1.0  |  AblePro x BIG Foundation Hackathon 2026  |  {now}",
        S["disclaimer"]
    ))

    doc.build(story)
    print(f"PDF Report saved: {pdf_path}")
    return pdf_path
