"""
DunkR8 - Professional AI-Powered Slam Dunk Analysis
Beautiful, animated, step-by-step dunk evaluation system.
"""
import tempfile
from pathlib import Path
import os
import time
from html import escape
import base64

import streamlit as st
import streamlit.components.v1 as components
import cv2

from pose_processor import PoseProcessor, process_video
from physics_engine import PhysicsEngine, PhysicsResult
from dunk_analyzer import DunkAnalyzer
from judge_explainer import generate_judge_reasoning


def _load_deepseek_api_key() -> str:
    """
    Load DeepSeek key from environment, with lightweight .env fallback.
    """
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key

    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return ""

    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == "DEEPSEEK_API_KEY":
                parsed = v.strip().strip('"').strip("'")
                if parsed:
                    os.environ["DEEPSEEK_API_KEY"] = parsed
                    return parsed
    except OSError:
        return ""
    return ""


def _sanitize_commentary(text: str) -> str:
    """
    Remove code fences / JSON-ish wrappers, strip em dashes, and escape HTML before rendering.
    """
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```markdown", "").replace("```text", "").replace("```", "").strip()
    if cleaned.startswith("{") and cleaned.endswith("}") and len(cleaned) > 20:
        cleaned = "Judge commentary unavailable in plain-text format for this attempt."
    # No em dashes: replace with " - "
    cleaned = cleaned.replace("\u2014", " - ").replace("\u2013", " - ")
    return escape(cleaned)


def _cap_display_metrics(analysis) -> dict:
    """Cap displayed metrics to plausible ranges to avoid overestimates."""
    vertical = min(float(analysis.max_vertical_inches or 0), 48.0)
    hang = min(float(analysis.hang_time_s or 0), 1.2)
    ball_air = getattr(analysis, "ball_air_time_s", None) or 0.0
    if ball_air > 2.0:
        ball_air = 2.0
    return {"vertical": vertical, "hang_time_s": hang, "ball_air_time_s": ball_air}


REFERENCE_CLIPS_DIR = Path(__file__).resolve().parent / "reference_dunks" / "clips"
LANDER_VIDEO_PATH = Path(__file__).resolve().parent / "dunkmix.mp4"


@st.cache_data(ttl=3600)
def _get_lander_video_b64() -> str:
    """Load and base64-encode lander video for custom no-controls HTML rendering."""
    if not LANDER_VIDEO_PATH.is_file():
        return ""
    try:
        return base64.b64encode(LANDER_VIDEO_PATH.read_bytes()).decode("utf-8")
    except OSError:
        return ""




def write_overlay_video_to_bytes(frames: list, fps: float) -> bytes:
    """Encode skeleton overlay frames to MP4. Prefer H.264 so it plays in browsers."""
    if not frames:
        return b""
    h, w = frames[0].shape[:2]
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        out_path = tmp.name
    # Try H.264 first (browser-friendly); fall back to mp4v
    for codec in ("avc1", "X264", "H264", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if out.isOpened():
            for f in frames:
                bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
                out.write(bgr)
            out.release()
            break
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        for f in frames:
            bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            out.write(bgr)
        out.release()
    with open(out_path, "rb") as f:
        data = f.read()
    try:
        Path(out_path).unlink(missing_ok=True)
    except OSError:
        pass
    return data


def create_animated_progress_terminal(steps, current_step=0, status="idle"):
    """Create animated progress terminal with autoscroll"""
    progress_html = f"""
    <div id="progress-terminal" class="progress-terminal" style="
        display: {'block' if current_step > 0 else 'none'};
        opacity: {'1' if current_step > 0 else '0'};
    ">
        <div class="terminal-header">
            <div class="terminal-controls">
                <span class="control-btn red"></span>
                <span class="control-btn yellow"></span>
                <span class="control-btn green"></span>
            </div>
            <span class="terminal-title">DunkR8 Analysis Engine</span>
        </div>
        <div class="terminal-content">
    """
    
    for i, step in enumerate(steps):
        if i < current_step:
            progress_html += f'<div class="terminal-line completed">✅ {step}</div>\n'
        elif i == current_step:
            progress_html += f'<div class="terminal-line active">⏳ {step}<span class="cursor">_</span></div>\n'
        else:
            progress_html += f'<div class="terminal-line pending">⏸️ {step}</div>\n'
    
    progress_html += """
        </div>
    </div>
    """
    return progress_html


def main():
    st.set_page_config(
        page_title="DunkR8 - AI Slam Dunk Analysis",
        page_icon="D8",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Beautiful animated UI with no sidebar
    st.markdown("""
    <style>
    /* Import Google Fonts for thick, modern typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
    
    /* Hide sidebar completely */
    section[data-testid="stSidebar"] { display: none !important; }
    .main .block-container { padding-left: 2rem !important; max-width: 100% !important; }
    
    /* Base styling - dark/black theme */
    .stApp { 
        background: radial-gradient(circle at top, #111111 0%, #080808 45%, #030303 100%);
        min-height: 100vh;
        font-family: 'Inter', sans-serif !important;
        overflow-x: visible !important;
    }
    .main .block-container { overflow: visible !important; }
    
    /* Hide default header */
    [data-testid="stHeader"] { display: none !important; }
    
    /* Navbar: full width, narrow height, stuck to top */
    .app-header {
        position: sticky;
        top: 0;
        left: 0;
        right: 0;
        width: 100vw;
        margin-left: calc(-50vw + 50%);
        margin-right: calc(-50vw + 50%);
        margin-bottom: 1.5rem;
        background: linear-gradient(90deg, rgba(10,10,10,0.98), rgba(18,18,18,0.96));
        backdrop-filter: blur(12px);
        border-bottom: 1px solid rgba(0,255,255,0.2);
        padding: 0.4rem 2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        z-index: 999;
        animation: slideInDown 0.5s ease-out;
    }
    
    .logo-section {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .logo-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 38px;
        height: 38px;
        animation: glow 2s ease-in-out infinite alternate;
    }
    .logo-icon svg {
        width: 36px;
        height: 36px;
    }
    
    .logo-text {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 900;
        background: linear-gradient(45deg, #00ffff, #ffffff, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
    }
    
    .tagline {
        color: #8892b0;
        font-size: 0.8rem;
        font-weight: 500;
        opacity: 0.9;
    }
    
    /* Landing video: square box */
    .landing-video-wrap {
        position: relative;
        width: 100%;
        max-width: 560px;
        margin: 0 auto;
        padding-top: 100%; /* 1:1 square */
        background: #000;
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow: 0 8px 32px rgba(0,0,0,0.5);
    }
    .landing-video-wrap video {
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        width: 100% !important;
        height: 100% !important;
        object-fit: cover !important;
        border-radius: 16px !important;
        border: none !important;
        background: #000 !important;
    }
    
    .landing-split-right {
        padding-left: 1.5rem;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    /* Progress Terminal */
    .progress-terminal {
        background: linear-gradient(145deg, rgba(7,7,7,0.97), rgba(14,14,14,0.94));
        border: 1px solid rgba(0,255,255,0.3);
        border-radius: 12px;
        margin: 2rem 0;
        box-shadow: 0 10px 40px rgba(0,255,255,0.1);
        animation: terminalSlideIn 0.6s ease-out;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .terminal-header {
        background: linear-gradient(90deg, rgba(16,16,16,0.95), rgba(24,24,24,0.9));
        padding: 0.75rem 1.5rem;
        border-radius: 12px 12px 0 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid rgba(0,255,255,0.2);
    }
    
    .terminal-controls {
        display: flex;
        gap: 0.5rem;
    }
    
    .control-btn {
        width: 12px;
        height: 12px;
        border-radius: 50%;
    }
    
    .control-btn.red { background: #ff5f5f; }
    .control-btn.yellow { background: #ffd93d; }
    .control-btn.green { background: #28ca42; }
    
    .terminal-title {
        color: #00ffff;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .terminal-content {
        padding: 1.5rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        line-height: 1.6;
        max-height: 300px;
        overflow-y: auto;
    }
    
    .terminal-line {
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        animation: fadeInLeft 0.4s ease-out;
    }
    
    .terminal-line.completed {
        color: #28ca42;
        font-weight: 500;
    }
    
    .terminal-line.active {
        color: #ffd93d;
        font-weight: 600;
        text-shadow: 0 0 10px rgba(255,217,61,0.3);
    }
    
    .terminal-line.pending {
        color: #6b7280;
    }
    
    .cursor {
        animation: blink 1s infinite;
    }
    
    /* File Upload Area */
    .upload-zone {
        background: linear-gradient(145deg, rgba(12,12,12,0.85), rgba(20,20,20,0.72));
        border: 2px dashed rgba(0,255,255,0.3);
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        margin: 2rem 0;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeInUp 0.8s ease-out;
    }
    
    .upload-zone:hover {
        border-color: rgba(0,255,255,0.6);
        background: linear-gradient(145deg, rgba(18,18,18,0.92), rgba(28,28,28,0.82));
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0,255,255,0.1);
    }
    
    .upload-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        background: linear-gradient(45deg, #00ffff, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: pulse 2s ease-in-out infinite;
    }
    
    .upload-text {
        color: #ffffff;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .upload-subtext {
        color: #8892b0;
        font-size: 1rem;
        font-weight: 500;
    }
    
    /* Single upload zone – the file uploader IS the big box */
    [data-testid="stFileUploader"] {
        background: linear-gradient(145deg, rgba(12,12,12,0.85), rgba(20,20,20,0.72)) !important;
        border: 2px dashed rgba(0,255,255,0.35) !important;
        border-radius: 20px !important;
        padding: 2.5rem 2rem !important;
        margin: 0 auto !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(0,255,255,0.6) !important;
        background: linear-gradient(145deg, rgba(18,18,18,0.92), rgba(28,28,28,0.82)) !important;
        box-shadow: 0 16px 40px rgba(0,255,255,0.08) !important;
    }
    
    [data-testid="stFileUploader"] > div {
        background: transparent !important;
        border: none !important;
    }
    
    [data-testid="stFileUploader"] section {
        background: transparent !important;
        border: none !important;
    }
    
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
        background: transparent !important;
        border: none !important;
    }
    
    .landing-hero {
        text-align: center;
        margin-bottom: 2.5rem;
        animation: fadeInUp 0.8s ease-out;
    }
    
    .landing-hero h1 {
        font-size: clamp(1.8rem, 4vw, 2.5rem) !important;
        margin-bottom: 0.75rem !important;
    }
    
    .landing-hero p {
        font-size: 1.05rem;
        color: #94a3b8 !important;
        max-width: 560px;
        margin: 0 auto 1.5rem;
        line-height: 1.6;
    }
    
    .demo-section {
        background: linear-gradient(145deg, rgba(10,10,10,0.96), rgba(16,16,16,0.9));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 2rem 0;
        animation: fadeInUp 0.9s ease-out 0.1s both;
    }
    
    .demo-section h3 {
        font-size: 1.25rem !important;
        margin-bottom: 1rem !important;
    }
    
    /* Analysis Results */
    .analysis-container {
        background: linear-gradient(145deg, rgba(10,10,10,0.96), rgba(16,16,16,0.9));
        border-radius: 20px;
        border: 1px solid rgba(0,255,255,0.2);
        margin: 2rem 0;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        animation: slideInUp 0.8s ease-out;
    }
    
    /* Video Container */
    .video-container {
        position: relative;
        border-radius: 16px;
        overflow: hidden;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    }
    
    [data-testid="stVideo"] {
        border-radius: 16px !important;
        border: 2px solid rgba(0,255,255,0.2) !important;
        transition: all 0.3s ease !important;
        max-height: min(48vh, 500px) !important;
        overflow: hidden !important;
        background: #000 !important;
    }

    [data-testid="stVideo"] video {
        max-height: min(48vh, 500px) !important;
        width: 100% !important;
        object-fit: contain !important;
        background: #000 !important;
    }
    
    [data-testid="stVideo"]:hover {
        border-color: rgba(0,255,255,0.5) !important;
        box-shadow: 0 0 20px rgba(0,255,255,0.2) !important;
    }
    
    /* Typography improvements */
    h1, h2, h3, h4 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        letter-spacing: -0.025em !important;
    }
    
    h1 { font-size: 3rem !important; }
    h2 { font-size: 2rem !important; }
    h3 { font-size: 1.5rem !important; }
    
    p, .stMarkdown {
        color: #e2e8f0 !important;
        font-weight: 500 !important;
    }
    
    /* Animations */
    @keyframes slideInDown {
        from { transform: translateY(-100px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    @keyframes slideInUp {
        from { transform: translateY(50px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    @keyframes fadeInUp {
        from { transform: translateY(30px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    @keyframes fadeInLeft {
        from { transform: translateX(-20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes terminalSlideIn {
        from { transform: translateY(20px) scale(0.98); opacity: 0; }
        to { transform: translateY(0) scale(1); opacity: 1; }
    }
    
    @keyframes glow {
        from { text-shadow: 0 0 10px rgba(0,255,255,0.5); }
        to { text-shadow: 0 0 20px rgba(0,255,255,0.8), 0 0 30px rgba(255,107,107,0.3); }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
    }
    
    /* Analysis split layout */
    .analysis-report-title {
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 0.6rem;
        font-size: 1.05rem;
    }
    .analysis-video-title {
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 0.6rem;
        font-size: 1.05rem;
        text-align: center;
    }
    
    /* Smooth transitions everywhere */
    * {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Custom header
    st.markdown("""
    <div class="app-header">
        <div class="logo-section">
            <div class="logo-icon">
                <svg viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                    <defs>
                        <linearGradient id="ballGrad" x1="0" y1="0" x2="1" y2="1">
                            <stop offset="0%" stop-color="#ff9f43"/>
                            <stop offset="100%" stop-color="#ff6b6b"/>
                        </linearGradient>
                    </defs>
                    <!-- hoop -->
                    <ellipse cx="20" cy="44" rx="11" ry="3.5" fill="none" stroke="#ffd93d" stroke-width="2.5"/>
                    <path d="M10 46 C12 52, 16 56, 20 58 C24 56, 28 52, 30 46" fill="none" stroke="#ffd93d" stroke-width="1.4" opacity="0.75"/>
                    <!-- basketball -->
                    <circle cx="42" cy="24" r="14" fill="url(#ballGrad)" stroke="#b84a2f" stroke-width="1.6"/>
                    <path d="M28 24 Q42 16 56 24" fill="none" stroke="#b84a2f" stroke-width="1.2"/>
                    <path d="M28 24 Q42 32 56 24" fill="none" stroke="#b84a2f" stroke-width="1.2"/>
                    <path d="M42 10 C38 16, 38 32, 42 38" fill="none" stroke="#b84a2f" stroke-width="1.2"/>
                    <path d="M42 10 C46 16, 46 32, 42 38" fill="none" stroke="#b84a2f" stroke-width="1.2"/>
                </svg>
            </div>
            <div>
                <div class="logo-text">DunkR8</div>
                <div class="tagline">AI-Powered Slam Dunk Analysis</div>
            </div>
        </div>
        <div style="color: #8892b0; font-weight: 500;">
            Professional NBA-Style Judging System
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state for progress
    if 'analysis_step' not in st.session_state:
        st.session_state.analysis_step = 0
    if 'uploaded_file_data' not in st.session_state:
        st.session_state.uploaded_file_data = None
    # Analysis steps
    analysis_steps = [
        "Processing video frames",
        "Tracking player pose landmarks",
        "Detecting basketball trajectory",
        "Computing physics metrics",
        "Running AI dunk classification",
        "Generating professional scorecard",
        "Creating judge commentary",
        "Finalizing analysis results"
    ]

    # Landing page (no file uploaded yet): split screen — video left, upload + text right
    if st.session_state.uploaded_file_data is None:
        col_video, col_upload = st.columns([1, 1])
        
        with col_video:
            lander_b64 = _get_lander_video_b64()
            if lander_b64:
                st.markdown(
                    f"""
                    <div class="landing-video-wrap">
                        <video autoplay loop muted playsinline preload="metadata"
                            src="data:video/mp4;base64,{lander_b64}">
                        </video>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("""
                <div class="landing-video-wrap" style="display:flex; align-items: center; justify-content: center; color: #64748b; text-align: center;">
                    <p style="position:absolute; top:50%; transform:translateY(-50%);">Add dunkmix.mp4 to the project root.</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col_upload:
            st.markdown("""
            <h2 style="margin-top: 0;">Professional Dunk Analysis</h2>
            <p style="color: #94a3b8; line-height: 1.6; margin-bottom: 1.5rem;">
                Upload a dunk clip and get NBA-style scoring, pose and ball tracking, and AI judge commentary. Built with computer vision and a trained dunk taxonomy.
            </p>
            <p style="font-size: 1rem; font-weight: 600; color: #e2e8f0; margin-bottom: 0.5rem;">Upload your clip</p>
            <p style="color: #64748b; font-size: 0.9rem; margin-bottom: 1rem;">MP4, MOV, AVI, or WebM</p>
            """, unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Upload Your Dunk Video",
                type=["mp4", "mov", "avi", "webm"],
                help="Drag and drop or click to choose a dunk video.",
                label_visibility="collapsed",
                key="main_upload",
            )
                
        if uploaded_file is not None:
            st.session_state.uploaded_file_data = uploaded_file
            st.rerun()

        return

    # Process uploaded file with animated progress
    uploaded_file = st.session_state.uploaded_file_data
    deepseek_api_key = _load_deepseek_api_key()
    
    # Show progress terminal
    progress_placeholder = st.empty()
    video_placeholder = st.empty()
    results_placeholder = st.empty()
    
    upload_bytes = uploaded_file.getvalue()
    ext = Path(uploaded_file.name).suffix or ".mp4"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(upload_bytes)
        tmp_path = tmp.name

    try:
        # Step 1: Initialize processing
        st.session_state.analysis_step = 1
        progress_placeholder.markdown(
            create_animated_progress_terminal(analysis_steps, st.session_state.analysis_step),
            unsafe_allow_html=True
        )
        time.sleep(0.5)  # Smooth animation timing

        # Step 2: Process video
        st.session_state.analysis_step = 2
        progress_placeholder.markdown(
            create_animated_progress_terminal(analysis_steps, st.session_state.analysis_step),
            unsafe_allow_html=True
        )
        
        processor = PoseProcessor()
        pose_frames, skeleton_frames, fps, ball_detections, ball_air_time_s, lob_type = process_video(tmp_path, processor)
        processor.close()

        if not pose_frames:
            st.error("No person detected in the video. Please upload a clip with a visible player.")
            st.session_state.uploaded_file_data = None
            return

        if not skeleton_frames:
            st.error("Could not read video frames.")
            st.session_state.uploaded_file_data = None
            return

        # Step 3: Ball tracking
        st.session_state.analysis_step = 3
        progress_placeholder.markdown(
            create_animated_progress_terminal(analysis_steps, st.session_state.analysis_step),
            unsafe_allow_html=True
        )
        time.sleep(0.3)

        # Step 4: Physics computation
        st.session_state.analysis_step = 4
        progress_placeholder.markdown(
            create_animated_progress_terminal(analysis_steps, st.session_state.analysis_step),
            unsafe_allow_html=True
        )

        frame_height = skeleton_frames[0].shape[0]
        body_height_norm = next(
            (p.body_height_norm for p in pose_frames if getattr(p, "body_height_norm", None)),
            None,
        )
        physics = PhysicsEngine(
            fps=fps,
            frame_height=frame_height,
            body_height_norm=body_height_norm,
        )
        result = physics.compute(pose_frames)

        if result is None:
            result = PhysicsResult(
                hang_time_s=0.0,
                max_vertical_inches=0.0,
                rotation_degrees=0.0,
                frames_airborne=0,
                start_hip_y=0.0,
                min_hip_y=0.0,
            )

        # Step 5: AI Classification
        st.session_state.analysis_step = 5
        progress_placeholder.markdown(
            create_animated_progress_terminal(analysis_steps, st.session_state.analysis_step),
            unsafe_allow_html=True
        )

        analyzer = DunkAnalyzer()
        frame_height, frame_width = skeleton_frames[0].shape[0], skeleton_frames[0].shape[1]
        analysis = analyzer.analyze(
            physics=result,
            pose_frames=pose_frames,
            ball_detections=ball_detections,
            ball_air_time_s=ball_air_time_s,
            lob_type=lob_type,
            frame_width=frame_width,
            frame_height=frame_height,
            clip_name=Path(uploaded_file.name).stem,
            ai_api_key=deepseek_api_key,
        )

        # Step 6: Scorecard
        st.session_state.analysis_step = 6
        progress_placeholder.markdown(
            create_animated_progress_terminal(analysis_steps, st.session_state.analysis_step),
            unsafe_allow_html=True
        )
        time.sleep(0.3)

        # Step 7: Judge Commentary
        st.session_state.analysis_step = 7
        progress_placeholder.markdown(
            create_animated_progress_terminal(analysis_steps, st.session_state.analysis_step),
            unsafe_allow_html=True
        )

        reasoning_key = (
            f"{uploaded_file.name}:{len(upload_bytes)}:{analysis.is_dunk}:"
            f"{analysis.dunk_type}:{analysis.final_contest_score:.1f}:{analysis.score_confidence:.3f}:"
            f"{bool(deepseek_api_key)}"
        )
        if st.session_state.get("judge_reasoning_key") != reasoning_key:
            st.session_state["judge_reasoning"] = generate_judge_reasoning(analysis, deepseek_api_key)
            st.session_state["judge_reasoning_key"] = reasoning_key
        
        judge_commentary = _sanitize_commentary(st.session_state.get("judge_reasoning", ""))

        # Step 8: Finalize
        st.session_state.analysis_step = 8
        progress_placeholder.markdown(
            create_animated_progress_terminal(analysis_steps, st.session_state.analysis_step),
            unsafe_allow_html=True
        )
        time.sleep(0.5)

        # Clear progress terminal with fade animation
        time.sleep(1)
        progress_placeholder.markdown("""
        <style>
        .progress-terminal { 
            animation: fadeOut 0.8s ease-out forwards; 
        }
        @keyframes fadeOut {
            from { opacity: 1; transform: scale(1); }
            to { opacity: 0; transform: scale(0.95); }
        }
        </style>
        """, unsafe_allow_html=True)
        
        time.sleep(0.8)
        progress_placeholder.empty()

        overlay_bytes = write_overlay_video_to_bytes(skeleton_frames, fps)

        # Analysis results with beautiful styling
        if analysis.is_dunk:
            header_color = "#00ff88" if analysis.final_contest_score >= 47 else "#ffd93d" if analysis.final_contest_score >= 44 else "#ff6b6b"
            label_text = analysis.dunk_type
            aux_text = analysis.primary_category
        else:
            header_color = "#ff4757"
            label_text = f"NOT A DUNK ({analysis.non_dunk_type})"
            aux_text = analysis.rejection_reason

        comps = analysis.score_components
        caps = _cap_display_metrics(analysis)
        aux_text_clean = (aux_text or "").replace("\u2014", " - ").replace("\u2013", " - ")
        # Escape for safe use inside HTML attributes/text (judge_commentary already escaped by _sanitize_commentary)
        label_safe = escape(label_text)
        aux_safe = escape(aux_text_clean)

        # Create comprehensive animated report (Inter font throughout). Rendered via iframe so HTML displays correctly.
        results_html = f"""
        <div class="analysis-container" style="font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;">
            <div style="
                text-align: center; 
                background: linear-gradient(145deg, rgba(20,20,40,0.8), rgba(30,30,60,0.6)); 
                padding: 2rem; 
                border-radius: 20px 20px 0 0;
                border-bottom: 2px solid {header_color}40;
                font-family: inherit;
            ">
                <div style="font-size: 1rem; color: #8892b0; font-weight: 600; margin-bottom: 1rem; font-family: inherit;">
                    {'OFFICIAL NBA SLAM DUNK CONTEST ANALYSIS' if deepseek_api_key else 'PROFESSIONAL DUNK ANALYSIS'}
                </div>
                <div style="font-size: clamp(1.8rem, 5vw, 3rem); font-weight: 900; color: {header_color}; margin-bottom: 0.5rem; text-shadow: 0 0 20px {header_color}40; font-family: inherit;">
                    {label_safe}
                </div>
                <div style="font-size: 1.2rem; color: #e2e8f0; margin-bottom: 1.5rem; font-weight: 500; font-family: inherit;">
                    {aux_safe}
                </div>
                <div style="
                    display: inline-block;
                    font-size: 4rem; 
                    font-weight: 900; 
                    color: {header_color}; 
                    background: linear-gradient(145deg, rgba(0,0,0,0.4), rgba(20,20,20,0.6));
                    padding: 1rem 2rem;
                    border-radius: 20px;
                    border: 2px solid {header_color}40;
                    text-shadow: 0 0 30px {header_color}60;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.5);
                ">
                    {analysis.final_contest_score:.1f}<span style="font-size: 2rem; color: #8892b0;">/50</span>
                </div>
                <div style="font-size: 1rem; color: #8892b0; margin-top: 1rem; font-weight: 500;">
                    {'AI-Enhanced Professional Scoring' if deepseek_api_key else 'Algorithmic Analysis'} - Confidence: {analysis.score_confidence * 100:.0f}%
                </div>
            </div>
            
            <!-- Judge Commentary Section -->
            <div style="
                margin: 0; 
                padding: 2rem; 
                background: linear-gradient(145deg, rgba(15,15,30,0.9), rgba(25,25,50,0.8)); 
                border-left: 4px solid {header_color};
            ">
                <div style="
                    display: flex; 
                    align-items: center; 
                    font-size: 1.3rem; 
                    color: #ffffff; 
                    font-weight: 700; 
                    margin-bottom: 1rem;
                    gap: 0.5rem;
                ">
                    {'Professional Judge Commentary' if deepseek_api_key else 'Analysis Summary'}
                </div>
                <div style="
                    font-size: 1.2rem; 
                    line-height: 1.7; 
                    color: #e2e8f0; 
                    font-style: italic;
                    font-weight: 500;
                    padding: 1.5rem;
                    background: rgba(10,10,20,0.6);
                    border-radius: 12px;
                    border: 1px solid rgba(255,255,255,0.1);
                    font-family: inherit;
                ">
                    {judge_commentary}
                </div>
            </div>
            
            <div style="padding: 2rem; background: linear-gradient(145deg, rgba(10,10,25,0.9), rgba(20,20,40,0.8)); font-family: inherit;">
                <div style="font-size: 1.5rem; color: #ffffff; font-weight: 800; margin-bottom: 2rem; text-align: center; text-shadow: 0 2px 10px rgba(0,0,0,0.5); font-family: inherit;">
                    Official NBA Judging Criteria
                </div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1.5rem; margin-bottom: 2rem;">
                    <div style="
                        text-align: center; 
                        padding: 1.5rem 1rem; 
                        background: linear-gradient(145deg, rgba(30,30,50,0.8), rgba(40,40,70,0.6)); 
                        border-radius: 16px; 
                        border: 2px solid {header_color}30;
                        transition: transform 0.3s ease;
                    ">
                        <div style="font-size: 1rem; color: #8892b0; margin-bottom: 0.5rem; font-weight: 600;">DIFFICULTY</div>
                        <div style="font-size: 2.5rem; font-weight: 900; color: {header_color}; text-shadow: 0 0 15px {header_color}40;">{comps.judge_difficulty:.1f}</div>
                        <div style="font-size: 0.9rem; color: #64748b; font-weight: 500;">/ 10</div>
                    </div>
                    <div style="
                        text-align: center; 
                        padding: 1.5rem 1rem; 
                        background: linear-gradient(145deg, rgba(30,30,50,0.8), rgba(40,40,70,0.6)); 
                        border-radius: 16px; 
                        border: 2px solid {header_color}30;
                    ">
                        <div style="font-size: 1rem; color: #8892b0; margin-bottom: 0.5rem; font-weight: 600;">EXECUTION</div>
                        <div style="font-size: 2.5rem; font-weight: 900; color: {header_color}; text-shadow: 0 0 15px {header_color}40;">{comps.judge_execution:.1f}</div>
                        <div style="font-size: 0.9rem; color: #64748b; font-weight: 500;">/ 10</div>
                    </div>
                    <div style="
                        text-align: center; 
                        padding: 1.5rem 1rem; 
                        background: linear-gradient(145deg, rgba(30,30,50,0.8), rgba(40,40,70,0.6)); 
                        border-radius: 16px; 
                        border: 2px solid {header_color}30;
                    ">
                        <div style="font-size: 1rem; color: #8892b0; margin-bottom: 0.5rem; font-weight: 600;">CREATIVITY</div>
                        <div style="font-size: 2.5rem; font-weight: 900; color: {header_color}; text-shadow: 0 0 15px {header_color}40;">{comps.judge_creativity:.1f}</div>
                        <div style="font-size: 0.9rem; color: #64748b; font-weight: 500;">/ 10</div>
                    </div>
                    <div style="
                        text-align: center; 
                        padding: 1.5rem 1rem; 
                        background: linear-gradient(145deg, rgba(30,30,50,0.8), rgba(40,40,70,0.6)); 
                        border-radius: 16px; 
                        border: 2px solid {header_color}30;
                    ">
                        <div style="font-size: 1rem; color: #8892b0; margin-bottom: 0.5rem; font-weight: 600;">ATHLETICISM</div>
                        <div style="font-size: 2.5rem; font-weight: 900; color: {header_color}; text-shadow: 0 0 15px {header_color}40;">{comps.judge_athleticism:.1f}</div>
                        <div style="font-size: 0.9rem; color: #64748b; font-weight: 500;">/ 10</div>
                    </div>
                    <div style="
                        text-align: center; 
                        padding: 1.5rem 1rem; 
                        background: linear-gradient(145deg, rgba(30,30,50,0.8), rgba(40,40,70,0.6)); 
                        border-radius: 16px; 
                        border: 2px solid {header_color}30;
                    ">
                        <div style="font-size: 1rem; color: #8892b0; margin-bottom: 0.5rem; font-weight: 600;">STYLE</div>
                        <div style="font-size: 2.5rem; font-weight: 900; color: {header_color}; text-shadow: 0 0 15px {header_color}40;">{comps.judge_style:.1f}</div>
                        <div style="font-size: 0.9rem; color: #64748b; font-weight: 500;">/ 10</div>
                    </div>
                </div>
                
                <!-- Technical Metrics Grid -->
                <div style="
                    display: grid; 
                    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); 
                    gap: 1rem; 
                    margin-bottom: 2rem;
                    padding: 1.5rem;
                    background: rgba(5,5,15,0.6);
                    border-radius: 12px;
                    border: 1px solid rgba(255,255,255,0.1);
                ">
                    <div style="text-align: center; padding: 1rem;">
                        <div style="font-size: 0.9rem; color: #64748b; margin-bottom: 0.5rem; font-weight: 600;">HANG TIME</div>
                        <div style="font-size: 1.8rem; font-weight: 800; color: #ffffff;">{caps['hang_time_s']:.2f}s</div>
                    </div>
                    <div style="text-align: center; padding: 1rem;">
                        <div style="font-size: 0.9rem; color: #64748b; margin-bottom: 0.5rem; font-weight: 600;">VERTICAL LEAP</div>
                        <div style="font-size: 1.8rem; font-weight: 800; color: #ffffff;">{caps['vertical']:.0f}"</div>
                    </div>
                    <div style="text-align: center; padding: 1rem;">
                        <div style="font-size: 0.9rem; color: #64748b; margin-bottom: 0.5rem; font-weight: 600;">BODY ROTATION</div>
                        <div style="font-size: 1.8rem; font-weight: 800; color: #ffffff;">{analysis.rotation_degrees:.0f}°</div>
                    </div>
                    <div style="text-align: center; padding: 1rem;">
                        <div style="font-size: 0.9rem; color: #64748b; margin-bottom: 0.5rem; font-weight: 600;">TAKEOFF DISTANCE</div>
                        <div style="font-size: 1.8rem; font-weight: 800; color: #ffffff;">{analysis.takeoff_distance_ft:.1f}ft</div>
                    </div>
                    <div style="text-align: center; padding: 1rem;">
                        <div style="font-size: 0.9rem; color: #64748b; margin-bottom: 0.5rem; font-weight: 600;">BALL AIR TIME</div>
                        <div style="font-size: 1.8rem; font-weight: 800; color: #ffffff;">{"%.2fs" % caps['ball_air_time_s'] if caps['ball_air_time_s'] and caps['ball_air_time_s'] > 0 else "-"}</div>
                    </div>
                    <div style="text-align: center; padding: 1rem;">
                        <div style="font-size: 0.9rem; color: #64748b; margin-bottom: 0.5rem; font-weight: 600;">LOB TYPE</div>
                        <div style="font-size: 1.8rem; font-weight: 800; color: #ffffff;">{analysis.lob_type or "Standard"}</div>
                    </div>
                </div>
                
                <!-- Confidence Metrics -->
                <div style="
                    display: flex; 
                    justify-content: center; 
                    gap: 3rem; 
                    padding: 1.5rem;
                    background: rgba(0,0,0,0.3);
                    border-radius: 12px;
                    border-top: 2px solid {header_color}40;
                ">
                    <div style="text-align: center;">
                        <div style="font-size: 1rem; color: #8892b0; font-weight: 600;">Detection Confidence</div>
                        <div style="font-size: 1.5rem; font-weight: 800; color: {header_color};">{analysis.dunk_probability * 100:.0f}%</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 1rem; color: #8892b0; font-weight: 600;">Classification Confidence</div>
                        <div style="font-size: 1.5rem; font-weight: 800; color: {header_color};">{analysis.dunk_type_confidence * 100:.0f}%</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 1rem; color: #8892b0; font-weight: 600;">Special Features</div>
                        <div style="font-size: 1.5rem; font-weight: 800; color: {header_color};">
                            {"Lob" if (analysis.alley_oop or analysis.self_lob) else ""}
                            {" + Object" if analysis.over_object else " + Standard" if not (analysis.alley_oop or analysis.self_lob) else ""}
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
        # Stacked layout: video on top, full-width report below
        with results_placeholder.container():
            st.markdown('<div class="analysis-video-title">Motion Analysis Video</div>', unsafe_allow_html=True)
            if overlay_bytes:
                st.video(
                    overlay_bytes,
                    format="video/mp4",
                    start_time=0,
                    autoplay=True,
                    muted=True,
                    loop=True,
                )
            else:
                st.warning("Could not render tracked video output.")

            st.markdown(
                '<div style="height:1px; width:100%; margin:1rem 0 1rem 0; background:rgba(0,255,255,0.22);"></div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="analysis-report-title">Analysis Report</div>', unsafe_allow_html=True)
            components.html(
                f'<div style="font-family: Inter, sans-serif;">{results_html}</div>',
                height=1700,
                scrolling=False,
            )

    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        st.session_state.uploaded_file_data = None
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except (PermissionError, OSError):
            pass


if __name__ == "__main__":
    main()
