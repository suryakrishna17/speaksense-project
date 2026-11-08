import { useState, useRef, useCallback, useEffect } from "react";
import {
  Activity, Upload, Mic, AlertTriangle,
  CheckCircle, XCircle, Download,
  Loader, Info, Waves, FileAudio2,
  Users, Languages, FileText, Sun, Moon,
  ChevronRight, BrainCircuit, Database, Play,
  RefreshCw, Sparkles, MessageSquare
} from "lucide-react";
import {
  RadarChart, PolarGrid, PolarAngleAxis, Radar,
  BarChart, Bar, XAxis, YAxis, Tooltip, Cell,
  ResponsiveContainer, ReferenceLine
} from "recharts";
import "./App.css";

const API = "http://127.0.0.1:8000";

/* ── helpers ─────────────────────────────────────────────── */
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

const fmtTime = (sec = 0) => {
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${String(m).padStart(2, "0")}:${s.toFixed(1).padStart(4, "0")}`;
};

const SPEAKER_COLORS = ["#818cf8", "#22d3ee", "#f472b6", "#34d399", "#fbbf24", "#f87171", "#a78bfa"];

/* ── components ──────────────────────────────────────────── */
const Pill = ({ label, value, color, bg, icon: Icon }) => (
  <div className="metric-pill-card ss-lift" style={{ borderLeft: `4px solid ${color}` }}>
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      {Icon && <Icon size={14} color={color} />}
      <span className="metric-pill-label" style={{ color: "var(--text-dim)" }}>{label}</span>
    </div>
    <span className="metric-pill-value" style={{ color }}>{value || "—"}</span>
  </div>
);

const PulseRing = ({ active }) => (
  <div style={{ position: "relative", width: 44, height: 44, display: "flex", alignItems: "center", justifyContent: "center" }}>
    {active && (
      <>
        <div style={{
          position: "absolute", width: 44, height: 44, borderRadius: "50%",
          border: "2px solid #f87171", animation: "ping 1.4s cubic-bezier(0,0,0.2,1) infinite", opacity: 0.6
        }} />
        <div style={{
          position: "absolute", width: 34, height: 34, borderRadius: "50%",
          border: "2px solid #f87171", animation: "ping 1.4s cubic-bezier(0,0,0.2,1) infinite 0.4s", opacity: 0.4
        }} />
      </>
    )}
    <Mic size={19} color={active ? "#f87171" : "#9fb2cd"} />
  </div>
);

const Gauge = ({ label, value, max, color, unit, gradId }) => {
  const pct = clamp(value / max, 0, 1);
  const r = 30, cx = 40, cy = 40, stroke = 6;
  const circ = 2 * Math.PI * r;
  const dash = pct * circ;
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 10, minWidth: 80 }}>
      <svg width={80} height={80} style={{ transform: "rotate(-90deg)" }}>
        <defs>
          <linearGradient id={gradId} x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor={color} stopOpacity={0.4} />
            <stop offset="100%" stopColor={color} />
          </linearGradient>
        </defs>
        <circle cx={cx} cy={cy} r={r} fill="none" stroke="var(--border-strong)" strokeWidth={stroke} />
        <circle cx={cx} cy={cy} r={r} fill="none" stroke={`url(#${gradId})`} strokeWidth={stroke}
          strokeDasharray={`${dash} ${circ}`} strokeLinecap="round"
          style={{ filter: `drop-shadow(0 0 3px ${color}33)`, transition: "stroke-dasharray 0.7s cubic-bezier(0.2,0.8,0.2,1)" }} />
        <text x={cx} y={cy} textAnchor="middle" dominantBaseline="central"
          fill="var(--text)" fontSize={13} fontWeight="700"
          className="mono"
          style={{ transform: "rotate(90deg)", transformOrigin: `${cx}px ${cy}px` }}>
          {typeof value === "number" ? value.toFixed(value < 10 ? 1 : 0) : value}
        </text>
      </svg>
      <span style={{ color: "var(--text-dim)", fontSize: 11, textAlign: "center", fontWeight: 600 }}>
        {label}{unit ? <span style={{ color: "var(--muted)" }}> ({unit})</span> : ""}
      </span>
    </div>
  );
};

/* ════════════════════════════════════════════════════════════
   MAIN APP
   ════════════════════════════════════════════════════════════ */
export default function App() {
  const [view,        setView]        = useState("dashboard"); // dashboard | training
  const [file,      setFile]      = useState(null);
  const [audioURL,  setAudioURL]  = useState(null);
  const [result,    setResult]    = useState(null);
  const [loading,   setLoading]   = useState(false);
  const [reporting, setReporting] = useState(false);
  const [error,     setError]     = useState(null);
  const [recording, setRecording] = useState(false);
  const [tab,       setTab]       = useState("overview");
  const [dragging,  setDragging]  = useState(false);
  const [apiHealth, setApiHealth] = useState("checking");
  const [caps,        setCaps]        = useState({});
  
  // Transcription States
  const [diarizedTranscript, setDiarizedTranscript] = useState(null);
  const [diarizedTranscribing, setDiarizedTranscribing] = useState(false);
  const [language,    setLanguage]    = useState("kn");

  // Model Training States
  const [trainInfo, setTrainInfo] = useState(null);
  const [trainStatus, setTrainStatus] = useState({ status: "idle", message: "Console initialized.", progress: 1.0 });
  const [pollingTrain, setPollingTrain] = useState(false);

  const [theme,       setTheme]       = useState("dark");
  const [shapTarget,  setShapTarget]  = useState("typicality_explanation");

  const mediaRef  = useRef(null);
  const chunksRef = useRef([]);
  const fileRef   = useRef(null);
  const audioRef  = useRef(null);
  const canvasRef = useRef(null);

  // Web Audio Refs
  const audioCtxRef = useRef(null);
  const analyserRef = useRef(null);
  const sourceRef   = useRef(null);
  const animRef     = useRef(null);

  const handleFile = (f) => {
    setFile(f);
    setAudioURL(URL.createObjectURL(f));
    setResult(null); setError(null);
    setDiarizedTranscript(null);
    cleanupAudioContext();
  };

  useEffect(() => {
    let alive = true;
    fetch(`${API}/health`)
      .then(res => res.ok ? res.json() : Promise.reject(new Error("offline")))
      .then(data => { if (alive) { setApiHealth("online"); setCaps(data.capabilities ?? {}); } })
      .catch(() => alive && setApiHealth("offline"));
      
    // Initial fetch of train info
    fetchTrainInfo();
    
    return () => { alive = false; };
  }, []);

  // Theme effect
  useEffect(() => {
    document.body.className = theme === "light" ? "light-theme" : "";
  }, [theme]);

  // Polling for training status
  useEffect(() => {
    let intervalId;
    if (pollingTrain) {
      intervalId = setInterval(() => {
        fetch(`${API}/train/status`)
          .then(res => res.json())
          .then(status => {
            setTrainStatus(status);
            if (status.status !== "running") {
              setPollingTrain(false);
              fetchTrainInfo();
            }
          })
          .catch(() => setPollingTrain(false));
      }, 2000);
    }
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [pollingTrain]);

  const fetchTrainInfo = async () => {
    try {
      const res = await fetch(`${API}/train/info`);
      if (res.ok) {
        const data = await res.json();
        setTrainInfo(data);
      }
    } catch (e) {
      console.error("Could not fetch training info", e);
    }
  };

  const startTraining = async () => {
    setError(null);
    try {
      const res = await fetch(`${API}/train/run`, { method: "POST" });
      if (!res.ok) throw new Error("Could not start model training.");
      setPollingTrain(true);
    } catch (e) {
      setError(e.message);
    }
  };

  const startAugmentation = async () => {
    setError(null);
    try {
      const res = await fetch(`${API}/augment/run`, { method: "POST" });
      if (!res.ok) throw new Error("Could not start dataset augmentation.");
      setPollingTrain(true);
    } catch (e) {
      setError(e.message);
    }
  };

  // Clean up Audio Context on unmount or file change
  const cleanupAudioContext = () => {
    if (animRef.current) cancelAnimationFrame(animRef.current);
    if (audioCtxRef.current) {
      audioCtxRef.current.close().catch(() => {});
      audioCtxRef.current = null;
      analyserRef.current = null;
      sourceRef.current = null;
    }
  };

  useEffect(() => {
    return () => cleanupAudioContext();
  }, []);

  // Set up real-time audio visualization
  const setupVisualizer = () => {
    if (!audioRef.current || !canvasRef.current) return;
    
    // Connect Web Audio nodes
    if (!audioCtxRef.current) {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      audioCtxRef.current = new AudioContextClass();
      analyserRef.current = audioCtxRef.current.createAnalyser();
      analyserRef.current.fftSize = 128;
      
      sourceRef.current = audioCtxRef.current.createMediaElementSource(audioRef.current);
      sourceRef.current.connect(analyserRef.current);
      analyserRef.current.connect(audioCtxRef.current.destination);
    }

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const analyser = analyserRef.current;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const draw = () => {
      if (!canvasRef.current || !analyserRef.current) return;
      animRef.current = requestAnimationFrame(draw);

      analyser.getByteFrequencyData(dataArray);

      // Canvas properties
      const width = canvas.width;
      const height = canvas.height;
      ctx.clearRect(0, 0, width, height);

      const barWidth = (width / bufferLength) * 1.5;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const value = dataArray[i];
        const percent = value / 255;
        const barHeight = percent * height * 0.95;

        // Indigo -> Cyan -> Teal glow gradient
        const grad = ctx.createLinearGradient(0, height, 0, height - barHeight);
        grad.addColorStop(0, "rgba(99, 102, 241, 0.15)");
        grad.addColorStop(0.5, "rgba(34, 211, 238, 0.7)");
        grad.addColorStop(1, "rgba(45, 212, 191, 1)");

        ctx.fillStyle = grad;
        // Rounded bars
        ctx.beginPath();
        ctx.roundRect(x, height - barHeight, barWidth - 3, barHeight, 2);
        ctx.fill();

        x += barWidth;
      }
    };

    if (animRef.current) cancelAnimationFrame(animRef.current);
    draw();
  };

  const handleAudioPlay = () => {
    // Resume context if suspended (browser security policy)
    if (audioCtxRef.current && audioCtxRef.current.state === "suspended") {
      audioCtxRef.current.resume();
    }
    setupVisualizer();
  };

  const handleAudioPause = () => {
    if (animRef.current) {
      cancelAnimationFrame(animRef.current);
      animRef.current = null;
    }
    // Clear canvas
    if (canvasRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  };

  const onDrop = useCallback((e) => {
    e.preventDefault(); setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, []);

  const startRec = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      chunksRef.current = [];
      mr.ondataavailable = e => chunksRef.current.push(e.data);
      mr.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/wav" });
        handleFile(new File([blob], "live_recording.wav", { type: "audio/wav" }));
      };
      mr.start(); mediaRef.current = mr; setRecording(true);
    } catch { setError("Microphone access denied."); }
  };
  const stopRec = () => { mediaRef.current?.stop(); setRecording(false); };

  const analyze = async () => {
    if (!file) return;
    setLoading(true); setError(null); setResult(null);
    const form = new FormData();
    form.append("file", file);
    try {
      const res  = await fetch(`${API}/analyze`, { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || data.detail || "Analysis failed.");
      setResult(data); setTab("overview");
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  const runDiarizedTranscription = async () => {
    if (!file) return;
    setDiarizedTranscribing(true); setError(null); setDiarizedTranscript(null);
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch(`${API}/transcribe/diarized?language=${language}`, { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || data.detail || "Diarized Transcription failed.");
      setDiarizedTranscript(data);
    } catch (e) { setError(e.message); }
    finally { setDiarizedTranscribing(false); }
  };

  const downloadReport = async () => {
    if (!result) return;
    setReporting(true); setError(null);
    try {
      // Pass the diarized transcripts if we have them
      const payload = {
        filename: file?.name,
        ...result,
        transcript_segments: diarizedTranscript?.segments?.map(s => ({
          speaker: s.speaker,
          start: s.start,
          end: s.end,
          transcription: s.transcription
        })) || []
      };

      const res  = await fetch(`${API}/report`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || "Report generation failed.");
      }
      const blob = await res.blob();
      const url  = URL.createObjectURL(blob);
      const a    = document.createElement("a");
      a.href = url; a.download = `${file?.name}_report.pdf`; a.click();
      URL.revokeObjectURL(url);
    } catch (e) { setError(e.message); }
    finally { setReporting(false); }
  };

  /* ── derived data ──────────────────────────────────────── */
  const kf     = result?.key_features ?? {};
  const cls    = result?.classification ?? {};
  const exp    = result?.explanation ?? {};
  const mfccs  = (result?.mfccs ?? []).map((v, i) => ({ name: `M${i+1}`, v: parseFloat(v?.toFixed(2) ?? 0) }));

  const radarData = result ? [
    { s: "Pitch Stability", v: clamp(100 - (kf.f0_std_hz ?? 0) * 1.5, 0, 100) },
    { s: "Voice Clarity",   v: clamp((kf.hnr_mean_db ?? 0) * 4, 0, 100) },
    { s: "Fluency",         v: clamp((kf.voiced_fraction ?? 0) * 100, 0, 100) },
    { s: "Jitter (inv)",    v: clamp(100 - (kf.jitter_local_pct ?? 0) * 20, 0, 100) },
    { s: "Shimmer (inv)",   v: clamp(100 - (kf.shimmer_local_db ?? 0) * 80, 0, 100) },
    { s: "Articulation",    v: clamp((kf.spectral_centroid_mean ?? 0) / 40, 0, 100) },
  ] : [];

  // SHAP feature driver list
  const shapDrivers = exp[shapTarget]?.top_features ?? [];
  const shapChartData = shapDrivers.map(d => ({
    name: d.label,
    impact: d.shap_value,
    direction: d.direction
  })).reverse(); // Reverse for horizontal layout

  const typPred   = cls.typicality?.prediction;
  const isAtyp    = typPred === "atypical";

  const isTrainingActive = trainStatus.status === "running";

  return (
    <div className="dashboard">
      {/* ── HEADER ─────────────────────────────────────── */}
      <header className="header-container">
        <div className="brand-section">
          <div className="brand-logo">
            <Waves size={24} color="#fff" strokeWidth={2.4} />
          </div>
          <div>
            <div className="brand-title">
              Speak<span style={{ color: "var(--cyan)" }}>Sense</span>
            </div>
            <div className="brand-subtitle">
              Automated Speech Diagnostic Engine
            </div>
          </div>
        </div>

        {/* Global Navigation Toggle */}
        <div style={{ display: "flex", gap: 8, background: "var(--inset)", padding: 4, borderRadius: 10, border: "1px solid var(--border)" }}>
          <button
            onClick={() => setView("dashboard")}
            style={{
              background: view === "dashboard" ? "var(--card-solid)" : "transparent",
              color: view === "dashboard" ? "var(--cyan)" : "var(--text-dim)",
              border: "none", borderRadius: 8, padding: "8px 16px", cursor: "pointer", fontWeight: 700, fontSize: 13
            }}
          >
            Diagnostics Dashboard
          </button>
          <button
            onClick={() => setView("training")}
            style={{
              background: view === "training" ? "var(--card-solid)" : "transparent",
              color: view === "training" ? "var(--cyan)" : "var(--text-dim)",
              border: "none", borderRadius: 8, padding: "8px 16px", cursor: "pointer", fontWeight: 700, fontSize: 13,
              display: "flex", alignItems: "center", gap: 6
            }}
          >
            <BrainCircuit size={14} />
            <span>Training Console</span>
            {isTrainingActive && <span className="api-dot online" style={{ width: 6, height: 6, display: "inline-block" }} />}
          </button>
        </div>
        
        <div className="header-actions">
          <button
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            className="theme-toggle-btn"
            aria-label="Toggle Light/Dark Theme"
          >
            {theme === "dark" ? <Sun size={17} /> : <Moon size={17} />}
          </button>
          
          <div className="api-badge">
            <span className={`api-dot ${apiHealth}`} />
            {apiHealth === "online" ? "API ONLINE" : apiHealth === "offline" ? "API OFFLINE" : "CHECKING API"}
          </div>
        </div>
      </header>

      {/* VIEW 1: DIAGNOSTIC DASHBOARD */}
      {view === "dashboard" && (
        <div className="dashboard-grid ss-fade">
          
          {/* LEFT COLUMN: CONTROLS */}
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            
            <div className="clinical-card">
              <span className="card-title-sec"><Upload size={14} /> Source Input</span>
              <div
                role="button" tabIndex={0} aria-label="Upload audio file"
                onDrop={onDrop}
                onDragOver={e => { e.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onClick={() => fileRef.current?.click()}
                onKeyDown={e => (e.key === "Enter" || e.key === " ") && fileRef.current?.click()}
                className={`upload-dropzone ${dragging ? "active" : ""}`}
              >
                <input ref={fileRef} type="file" accept="audio/*" style={{ display: "none" }}
                  onChange={e => e.target.files[0] && handleFile(e.target.files[0])} />
                
                <div style={{
                  width: 44, height: 44, borderRadius: 10,
                  background: "var(--inset)", border: "1px solid var(--border)",
                  display: "flex", alignItems: "center", justifyItems: "center", justifyContent: "center"
                }}>
                  {file ? <FileAudio2 size={20} color="var(--teal)" /> : <Upload size={18} color="var(--text-dim)" />}
                </div>
                
                <div style={{ minWidth: 0, width: "100%" }}>
                  <div style={{ color: file ? "var(--teal)" : "var(--text)", fontWeight: 700, fontSize: 14, textOverflow: "ellipsis", overflow: "hidden", whiteSpace: "nowrap" }}>
                    {file ? file.name : "Choose audio or drag here"}
                  </div>
                  <div style={{ color: "var(--muted)", fontSize: 11, marginTop: 4 }}>Standard audio files supported</div>
                </div>
              </div>

              <div className="record-btn-container">
                <button
                  onClick={recording ? stopRec : startRec}
                  aria-label={recording ? "Stop recording" : "Start recording"}
                  className={`btn ${recording ? "btn-danger" : ""}`}
                  style={{ flex: 1, height: 42 }}
                >
                  <PulseRing active={recording} />
                  <span>{recording ? "Stop Recording" : "Record Live"}</span>
                </button>
              </div>

              {file && (
                <div className="audio-bar">
                  <canvas ref={canvasRef} className="visualizer-canvas" width={300} height={60} />
                  {audioURL && (
                    <audio
                      ref={audioRef}
                      src={audioURL}
                      controls
                      onPlay={handleAudioPlay}
                      onPause={handleAudioPause}
                      onEnded={handleAudioPause}
                      style={{ height: 36, width: "100%", filter: theme === "dark" ? "invert(0.92) hue-rotate(180deg)" : "none" }}
                    />
                  )}
                </div>
              )}
            </div>

            {file && (
              <div className="clinical-card" style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                <span className="card-title-sec"><Activity size={14} /> Diagnostic Pipeline</span>
                
                <button
                  onClick={analyze} disabled={!file || loading}
                  className="btn btn-primary"
                >
                  {loading
                    ? <><Loader size={16} style={{ animation: "spin 0.9s linear infinite" }} /> Extracting parameters…</>
                    : <><Activity size={16} strokeWidth={2.4} /> Run Analysis</>
                  }
                </button>

                {/* Transcription options */}
                <div style={{ display: "flex", flexDirection: "column", gap: 8, borderTop: "1px solid var(--border)", paddingTop: 16 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <span style={{ fontSize: 11, fontWeight: 700, color: "var(--muted)" }}>TRANSCRIBE LANGUAGE</span>
                    <div style={{ display: "flex", background: "var(--inset)", borderRadius: 6, padding: 2, border: "1px solid var(--border)" }}>
                      {[["kn", "ಕನ್ನಡ"], ["en", "EN"]].map(([id, lbl]) => (
                        <button key={id} onClick={() => setLanguage(id)}
                          style={{
                            border: "none", borderRadius: 4, padding: "2px 6px", cursor: "pointer",
                            background: language === id ? "var(--cyan)" : "transparent",
                            color: language === id ? "#0f172a" : "var(--muted)",
                            fontWeight: 700, fontSize: 9.5,
                          }}>{lbl}</button>
                      ))}
                    </div>
                  </div>
                  <button onClick={runDiarizedTranscription} disabled={diarizedTranscribing || caps.transcribe === false}
                    className="btn"
                    style={{ width: "100%", padding: "12px 0", fontSize: 13, background: "rgba(99,102,241,0.06)", borderColor: "var(--indigo)" }}
                  >
                    {diarizedTranscribing 
                      ? <><Loader size={14} style={{ animation: "spin 0.9s linear infinite" }} /> Aligning transcripts…</> 
                      : <><Sparkles size={14} color="var(--indigo)" /> <span>Diarized Speech Transcription</span></>
                    }
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* RIGHT COLUMN: DIAGNOSTIC REPORT */}
          <div>
            {/* Error Banner */}
            {error && (
              <div className="alert-banner error ss-fade">
                <XCircle size={16} style={{ flexShrink: 0 }} />
                <span>{error}</span>
              </div>
            )}

            {result ? (
              <div className="ss-fade" style={{ display: "flex", flexDirection: "column", gap: 20 }}>
                
                {/* Classification Status banner */}
                <div className={`clinical-card ${isAtyp ? "highlight-danger" : "highlight-ok"}`}
                  style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 16, flexWrap: "wrap" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                    {isAtyp ? <AlertTriangle size={22} color="var(--red)" /> : <CheckCircle size={22} color="var(--green)" />}
                    <div>
                      <div style={{ fontWeight: 800, fontSize: 16, color: isAtyp ? "var(--red)" : "var(--green)" }}>
                        {isAtyp ? "Atypical Speech Patterns Detected" : "Typical Speech Patterns"}
                      </div>
                      <div style={{ color: "var(--text-dim)", fontSize: 12, marginTop: 2 }}>
                        {cls.typicality?.confidence
                          ? `Model confidence: ${(cls.typicality.confidence * 100).toFixed(0)}%`
                          : "Clinical indicators matched"}
                      </div>
                    </div>
                  </div>

                  <button
                    onClick={downloadReport} disabled={reporting}
                    className="btn"
                    style={{ height: 38 }}
                  >
                    {reporting ? <Loader size={13} style={{ animation: "spin 0.9s linear infinite" }} /> : <Download size={13} />}
                    <span>PDF Report</span>
                  </button>
                </div>

                {/* Heuristic Fallback Note */}
                {result.model_note && (
                  <div className="alert-banner info" style={{ padding: "12px 16px", marginBottom: 0 }}>
                    <Info size={14} color="var(--indigo)" style={{ flexShrink: 0 }} />
                    <span style={{ fontSize: 12, lineHeight: 1.5 }}>{result.model_note}</span>
                  </div>
                )}

                {/* Classification Pills */}
                <div className="pills-container">
                  <Pill label="Age Group" value={cls.age?.prediction} icon={Activity}
                    color={cls.age?.prediction === "child" ? "var(--amber)" : "var(--teal)"} />
                  <Pill label="Gender Profile" value={cls.gender?.prediction} icon={Mic}
                    color={cls.gender?.prediction === "female" ? "var(--pink)" : "var(--blue)"} />
                  <Pill label="Acoustic Speech" value={cls.typicality?.prediction} icon={Waves}
                    color={isAtyp ? "var(--red)" : "var(--green)"} />
                </div>

                {/* Gauges Panel */}
                <div className="gauges-panel">
                  <Gauge gradId="g-f0"  label="Pitch" value={kf.f0_mean_hz ?? 0} max={500} color="var(--cyan)" unit="Hz" />
                  <Gauge gradId="g-hnr" label="Clarity (HNR)" value={kf.hnr_mean_db ?? 0} max={40} color="var(--green)" unit="dB" />
                  <Gauge gradId="g-jit" label="Jitter" value={kf.jitter_local_pct ?? 0} max={3} color={isAtyp ? "var(--red)" : "var(--cyan)"} unit="%" />
                  <Gauge gradId="g-shm" label="Shimmer" value={kf.shimmer_local_db ?? 0} max={1} color={isAtyp ? "var(--red)" : "var(--cyan)"} unit="dB" />
                  <Gauge gradId="g-vf"  label="Voicing" value={(kf.voiced_fraction ?? 0) * 100} max={100} color="var(--indigo)" unit="%" />
                </div>

                {/* Segmented Control & Detail Tabs */}
                <div>
                  <div className="tab-group">
                    {[
                      ["overview", "Overview"],
                      ["clinical_checks", "Clinical Checks"],
                      ["shap", "Classification Drivers"],
                      ["mfccs", "MFCC Coefficients"],
                      ["features", "All Parameters"]
                    ].map(([id, lbl]) => (
                      <button key={id} onClick={() => setTab(id)}
                        className={`tab-btn ${tab === id ? "active" : ""}`}
                      >
                        {lbl}
                      </button>
                    ))}
                  </div>

                  {/* Tab Content */}
                  {tab === "overview" && (
                    <div className="ss-fade" style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 20 }}>
                      {/* Radar Chart */}
                      <div className="clinical-card" style={{ display: "flex", alignItems: "center", justifyContent: "center", minHeight: 250 }}>
                        <ResponsiveContainer width="100%" height={230}>
                          <RadarChart data={radarData} margin={{ top: 10, right: 24, bottom: 10, left: 24 }}>
                            <PolarGrid stroke="var(--border)" />
                            <PolarAngleAxis dataKey="s" tick={{ fill: "var(--text-dim)", fontSize: 10, fontWeight: 600 }} />
                            <Radar dataKey="v" stroke={isAtyp ? "var(--red)" : "var(--green)"} fill={isAtyp ? "var(--red)" : "var(--green)"} fillOpacity={0.15} strokeWidth={2} />
                          </RadarChart>
                        </ResponsiveContainer>
                      </div>

                      {/* Narrative Card */}
                      <div className="clinical-card" style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                        <span className="card-title-sec"><BrainCircuit size={13} /> Clinical Summary</span>
                        <div style={{ fontSize: 14, lineHeight: 1.6, color: "var(--text)" }}>
                          {exp.clinical?.narrative || "No narrative evaluation available."}
                        </div>
                        {exp.clinical?.flag_count > 0 && (
                          <div style={{ marginTop: "auto", display: "flex", alignItems: "center", gap: 6, color: "var(--amber)", fontSize: 12.5, fontWeight: 700 }}>
                            <AlertTriangle size={14} />
                            <span>{exp.clinical.flag_count} clinical parameters outside normal ranges.</span>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {tab === "clinical_checks" && (
                    <div className="ss-fade clinical-card">
                      <span className="card-title-sec"><Activity size={14} /> Clinical Threshold Reference</span>
                      <div className="table-grid">
                        {exp.clinical?.clinical_checks?.map((c, i) => (
                          <div key={i} className={`table-row ${c.flag ? "warning" : ""}`}>
                            <span className="row-label">
                              {c.flag ? <AlertTriangle size={14} color="var(--amber)" /> : <CheckCircle size={14} color="var(--green)" />}
                              {c.parameter}
                            </span>
                            <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
                              <span className="mono" style={{ fontSize: 12, color: "var(--muted)" }}>Range: {c.normal_range}</span>
                              <span className="row-value mono">{c.value} {c.unit}</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {tab === "shap" && (
                    <div className="ss-fade clinical-card">
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 12, marginBottom: 16 }}>
                        <span className="card-title-sec" style={{ marginBottom: 0 }}><BrainCircuit size={14} /> ML Prediction Drivers (SHAP)</span>
                        
                        {/* Prediction Target Select */}
                        <div style={{ display: "flex", background: "var(--inset)", borderRadius: 8, padding: 2, border: "1px solid var(--border)" }}>
                          {[
                            ["typicality_explanation", "Typicality"],
                            ["age_explanation", "Age Group"]
                          ].map(([id, lbl]) => (
                            <button key={id} onClick={() => setShapTarget(id)}
                              style={{
                                border: "none", borderRadius: 6, padding: "4px 10px", cursor: "pointer",
                                background: shapTarget === id ? "var(--cyan)" : "transparent",
                                color: shapTarget === id ? "#0f172a" : "var(--text-dim)",
                                fontWeight: 700, fontSize: 11,
                              }}>{lbl}</button>
                          ))}
                        </div>
                      </div>

                      {shapDrivers.length > 0 ? (
                        <>
                          <div style={{ color: "var(--text-dim)", fontSize: 12, marginBottom: 20 }}>
                            Features pushing prediction toward <strong>{exp[shapTarget]?.prediction?.toUpperCase()}</strong>.
                          </div>
                          <ResponsiveContainer width="100%" height={260}>
                            <BarChart data={shapChartData} layout="vertical" margin={{ left: 50, right: 20 }}>
                              <XAxis type="number" tick={{ fill: "var(--text-dim)", fontSize: 10 }} axisLine={{ stroke: "var(--border)" }} />
                              <YAxis type="category" dataKey="name" tick={{ fill: "var(--text)", fontSize: 10, fontWeight: 500 }} width={120} axisLine={false} tickLine={false} />
                              <Tooltip
                                contentStyle={{ background: "var(--card-solid)", border: "1px solid var(--border-strong)", borderRadius: 8, color: "var(--text)", fontFamily: "monospace", fontSize: 11 }}
                                cursor={{ fill: "rgba(255, 255, 255, 0.02)" }}
                              />
                              <Bar dataKey="impact">
                                {shapChartData.map((entry, i) => (
                                  <Cell key={i} fill={entry.impact >= 0 ? "var(--red)" : "var(--green)"} opacity={0.8} />
                                ))}
                              </Bar>
                            </BarChart>
                          </ResponsiveContainer>
                        </>
                      ) : (
                        <div style={{ textAlign: "center", padding: "32px 0", color: "var(--muted)", fontSize: 13 }}>
                          SHAP model explanations not available for this heuristic fallback.
                        </div>
                      )}
                    </div>
                  )}

                  {tab === "mfccs" && (
                    <div className="ss-fade clinical-card">
                      <span className="card-title-sec">MFCC Timbre Representation</span>
                      <div style={{ color: "var(--text-dim)", fontSize: 12, marginBottom: 16 }}>
                        Primary cepstral indicators representing acoustic voice shape and tone.
                      </div>
                      <ResponsiveContainer width="100%" height={230}>
                        <BarChart data={mfccs} margin={{ left: -18 }}>
                          <XAxis dataKey="name" tick={{ fill: "var(--text-dim)", fontSize: 11 }} axisLine={{ stroke: "var(--border)" }} tickLine={false} />
                          <YAxis tick={{ fill: "var(--text-dim)", fontSize: 10 }} axisLine={false} tickLine={false} />
                          <Tooltip
                            contentStyle={{ background: "var(--card-solid)", border: "1px solid var(--border-strong)", borderRadius: 8, color: "var(--text)", fontFamily: "monospace", fontSize: 11 }}
                            cursor={{ fill: "rgba(255, 255, 255, 0.02)" }}
                          />
                          <ReferenceLine y={0} stroke="var(--border)" />
                          <Bar dataKey="v" radius={[3, 3, 0, 0]}>
                            {mfccs.map((entry, i) => (
                              <Cell key={i} fill={entry.v >= 0 ? "var(--cyan)" : "var(--pink)"} opacity={0.8} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  )}

                  {tab === "features" && (
                    <div className="ss-fade clinical-card" style={{ maxHeight: 380, overflowY: "auto" }}>
                      <span className="card-title-sec">All Acoustic Parameters ({result.total_features_extracted} features)</span>
                      <div className="table-grid">
                        {Object.entries(kf).map(([k, v]) => (
                          <div key={k} className="table-row">
                            <span className="row-label mono" style={{ fontSize: 12 }}>{k}</span>
                            <span className="row-value mono" style={{ fontSize: 12 }}>
                              {typeof v === "number" ? v.toFixed(5) : v}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
              ) : (
                <div className="clinical-card" style={{ textAlign: "center", padding: "64px 24px", color: "var(--muted)" }}>
                  <Waves size={32} color="var(--border-strong)" style={{ marginBottom: 16 }} />
                  <div style={{ fontSize: 15, color: "var(--text)", fontWeight: 700 }}>Awaiting Diagnostic Data</div>
                  <div style={{ fontSize: 13, marginTop: 6, color: "var(--text-dim)" }}>Select and run analysis on an audio sample to extract markers.</div>
                </div>
              )}

              {/* Advanced Conversational Diarized Transcripts */}
              {diarizedTranscript && (
                <div className="ss-fade clinical-card" style={{ marginTop: 20 }}>
                  <span className="card-title-sec"><MessageSquare size={14} /> Diarized Conversation script</span>
                  
                  {diarizedTranscript.segments && diarizedTranscript.segments.length > 0 ? (
                    <div className="chat-container">
                      {diarizedTranscript.segments.map((seg, idx) => {
                        const avatarChar = seg.speaker.split("-")[1] || "A";
                        const speakerColor = SPEAKER_COLORS[parseInt(avatarChar) % SPEAKER_COLORS.length] || "var(--cyan)";
                        
                        return (
                          <div key={idx} className="chat-row ss-fade">
                            <div className="chat-avatar" style={{ border: `2px solid ${speakerColor}`, color: speakerColor }}>
                              S{avatarChar}
                            </div>
                            <div className="chat-bubble">
                              <div className="chat-bubble-header">
                                <span className="chat-speaker-name" style={{ color: speakerColor }}>
                                  {seg.speaker}
                                </span>
                                <span className="chat-timestamp mono">
                                  {fmtTime(seg.start)} → {fmtTime(seg.end)}
                                </span>
                              </div>
                              <div className="chat-text">
                                {seg.transcription?.trim() || <span style={{ color: "var(--muted)", fontStyle: "italic" }}>[silence / no speech]</span>}
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  ) : (
                    <div style={{ color: "var(--muted)", fontSize: 13, fontStyle: "italic", textAlign: "center", padding: "16px 0" }}>
                      No audio transcript available.
                    </div>
                  )}
                </div>
              )}
          </div>

        </div>
      )}

      {/* VIEW 2: MODEL TRAINING CONSOLE */}
      {view === "training" && (
        <div className="ss-fade" style={{ display: "flex", flexDirection: "column", gap: 24 }}>
          {/* Active Process Progress Card */}
          {isTrainingActive && (
            <div className="clinical-card highlight-ok ss-fade">
              <span className="card-title-sec" style={{ color: "var(--green)" }}>
                <Loader size={14} style={{ animation: "spin 0.9s linear infinite" }} /> Background Job Running
              </span>
              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                <div style={{ fontSize: 15, fontWeight: 700 }}>{trainStatus.message}</div>
                <div className="progress-container">
                  <div className="progress-fill" style={{ width: `${trainStatus.progress * 100}%` }} />
                </div>
                <span className="mono" style={{ fontSize: 11, color: "var(--muted)", alignSelf: "flex-end" }}>
                  {(trainStatus.progress * 100).toFixed(0)}% Complete
                </span>
              </div>
            </div>
          )}

          <div className="console-panel-grid">
            {/* Action Card 1: Data Augmentation */}
            <div className="clinical-card">
              <span className="card-title-sec"><Database size={14} /> Raw Dataset & Expansion</span>
              <div style={{ color: "var(--text-dim)", fontSize: 13.5, lineHeight: 1.5, marginBottom: 12 }}>
                Boost the active pool of files using audio transformations (pitch shift, time stretch, noise insertion) to prevent ML overfitting on small sample sizes.
              </div>

              <div className="console-action-card">
                <div className="console-stat-row">
                  <span>Source Audio Files:</span>
                  <span className="console-stat-value">{trainInfo?.samples_count ?? 0} WAVs</span>
                </div>
                <div className="console-stat-row">
                  <span>Normalized Pool:</span>
                  <span className="console-stat-value">{trainInfo?.processed_count ?? 0} files</span>
                </div>
                <div className="console-stat-row">
                  <span>Augmented Copies:</span>
                  <span className="console-stat-value">{trainInfo?.augmented_count ?? 0} files</span>
                </div>

                <button
                  onClick={startAugmentation}
                  disabled={isTrainingActive}
                  className="btn btn-primary"
                  style={{ background: "linear-gradient(135deg, var(--teal) 0%, var(--cyan) 100%)", boxShadow: "0 4px 15px rgba(45, 212, 191, 0.2)" }}
                >
                  <Sparkles size={14} /> Run Audio Augmentation (8x)
                </button>
              </div>
            </div>

            {/* Action Card 2: Ensemble Models */}
            <div className="clinical-card">
              <span className="card-title-sec"><BrainCircuit size={14} /> Ensemble Classifiers (RF + XGB + SVM)</span>
              <div style={{ color: "var(--text-dim)", fontSize: 13.5, lineHeight: 1.5, marginBottom: 12 }}>
                Train an ensemble of Random Forest, XGBoost, and SVM using stratified cross-validation. When models are active, SpeakSense uses them for explainable inference.
              </div>

              <div className="console-action-card" style={{ gap: 12 }}>
                {trainInfo?.models && Object.entries(trainInfo.models).map(([modelKey, info]) => (
                  <div key={modelKey} className="model-badge-row">
                    <span className="model-badge-name" style={{ textTransform: "capitalize" }}>
                      {modelKey.replace("_classification", "")} Classifier
                    </span>
                    <span className={`model-status-indicator ${info.exists ? "active" : "inactive"}`}>
                      {info.exists ? "Model Active" : "No Model File"}
                    </span>
                  </div>
                ))}

                <button
                  onClick={startTraining}
                  disabled={isTrainingActive}
                  className="btn btn-primary"
                  style={{ marginTop: 8 }}
                >
                  <Play size={14} /> Train Classifier Models
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
