"""
Slam Dunk Score Predictor
Upload MP4 dunk video, view MediaPipe skeleton overlay, and get Judges' Scorecard.
"""
import tempfile
from pathlib import Path

import streamlit as st
import cv2

from pose_processor import PoseProcessor, process_video
from physics_engine import PhysicsEngine, PhysicsResult
from dunk_analyzer import DunkAnalyzer


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


def main():
    st.set_page_config(
        page_title="DunkR8",
        page_icon="🏀",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Smooth dark UI — contest-style
    st.markdown("""
    <style>
    /* Base */
    .stApp { background: linear-gradient(165deg, #0c0c0c 0%, #141414 40%, #0a0a0a 100%); min-height: 100vh; }
    .stApp header { background: transparent !important; }
    [data-testid="stHeader"] { background: rgba(0,0,0,0.4) !important; backdrop-filter: blur(8px); }
    /* Main — smoother spacing */
    .main .block-container { padding: 2.25rem 2.25rem 3.5rem; max-width: 1200px; }
    .main .block-container > div { margin-bottom: 1.25rem; }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e0e0e 0%, #121212 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] .stMarkdown { color: #b0b0b0; }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: #fff !important; }
    /* Metrics — card feel */
    [data-testid="stMetric"] {
        background: rgba(22,22,22,0.8); padding: 1rem 1.1rem; border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.06); transition: border-color 0.2s;
    }
    [data-testid="stMetric"]:hover { border-color: rgba(255,255,255,0.12); }
    [data-testid="stMetric"] label { color: #888 !important; font-size: 0.85rem; }
    [data-testid="stMetric"] [data-testid="stMetricValue"] { color: #fff !important; }
    /* Typography */
    h1, h2, h3 { color: #fff !important; font-weight: 600; letter-spacing: -0.02em; }
    h2 { font-size: 1.25rem; margin-top: 1.5rem; }
    p, .stMarkdown { color: #b0b0b0 !important; }
    label { color: #999 !important; }
    /* File uploader */
    [data-testid="stFileUploader"] { background: rgba(17,17,17,0.9); border: 1px dashed rgba(255,255,255,0.12); border-radius: 14px; padding: 1.25rem; }
    [data-testid="stFileUploader"] section { background: transparent !important; }
    /* Alerts */
    .stSuccess { background: rgba(34,197,94,0.12) !important; color: #86efac !important; border: 1px solid rgba(34,197,94,0.3); border-radius: 10px; }
    .stInfo { background: rgba(59,130,246,0.1) !important; color: #93c5fd !important; border: 1px solid rgba(59,130,246,0.25); border-radius: 10px; }
    .stError { background: rgba(239,68,68,0.12) !important; color: #fca5a5 !important; border: 1px solid rgba(239,68,68,0.3); border-radius: 10px; }
    .stWarning { background: rgba(234,179,8,0.12) !important; color: #fde047 !important; border: 1px solid rgba(234,179,8,0.3); border-radius: 10px; }
    /* Slider */
    .stSlider label { color: #999 !important; }
    [data-testid="stSlider"] div div { background: rgba(37,37,37,0.9) !important; border-radius: 6px; }
    /* Expander */
    .streamlit-expanderHeader { background: rgba(21,21,21,0.9) !important; color: #fff !important; border-radius: 10px; border: 1px solid rgba(255,255,255,0.06); }
    /* Video */
    [data-testid="stVideo"] {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(0,0,0,0.35);
    }
    /* Extra metric cards at bottom */
    .extra-metric-card {
        background: rgba(18,18,18,0.9); border: 1px solid rgba(255,255,255,0.06); border-radius: 12px;
        padding: 1rem 1.25rem; text-align: center; margin: 0.25rem 0;
    }
    .extra-metric-card .value { font-size: 1.35rem; font-weight: 700; color: #fff; }
    .extra-metric-card .label { font-size: 0.8rem; color: #888; margin-top: 0.25rem; }
    </style>
    """, unsafe_allow_html=True)

    st.title("🏀 DunkR8")
    st.caption("Reliable dunk detection, taxonomy, and contest scoring")

    st.sidebar.header("Judges' Scorecard")
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "DunkR8 engine (pose + ball tracking)\n"
        "with trainable dunk prototypes."
    )
    with st.sidebar.expander("Train model from reference clips"):
        st.caption(
            "Place clips in `reference_dunks/clips/` with dunk type names in filenames "
            "(example: `windmill.mp4`, `360_dunk.mp4`, `reverse_eastbay.mp4`)."
        )
        if st.button("Train DunkR8 model", use_container_width=True):
            try:
                from ontology_trainer import train_from_reference_clips
                with st.spinner("Training model from reference clips..."):
                    summary = train_from_reference_clips()
                st.success(
                    f"Trained {summary.classes_trained} class(es) from {summary.clips_used}/{summary.clips_found} clip(s)."
                )
                if summary.skipped:
                    st.warning("Skipped clips:\n- " + "\n- ".join(summary.skipped[:6]))
            except Exception as e:
                st.error(str(e))
    uploaded_file = st.sidebar.file_uploader(
        "Upload Dunk Video",
        type=["mp4", "mov", "avi", "webm"],
        help="Upload a dunk video (MP4, MOV, AVI, WebM)",
    )

    if uploaded_file is None:
        st.markdown(
            '<p style="color:#b0b0b0; font-size:1.1rem; margin-bottom:1.5rem;">'
            'Upload a dunk clip to run DunkR8 tracking and score it on a contest 40–50 scale.'
            '</p>',
            unsafe_allow_html=True,
        )
        with st.expander("How it works", expanded=False):
            st.markdown("""
            - **Player** — MediaPipe Pose tracks heels, hips, shoulders
            - **Ball** — Orange/red detection tracks the basketball (air time for lobs/alley-oops)
            - **Physics** — Hang time, max vertical, rotation, arm path cues
            - **Dunk ontology** — Canonical dunk taxonomy + strict rejection logic
            - **Score 40–50** — Contest scale from difficulty, style, rotation, height, and lob context
            """)
        return

    upload_bytes = uploaded_file.getvalue()
    ext = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(upload_bytes)
        tmp_path = tmp.name

    try:
        with st.spinner("Tracking player + ball and computing metrics..."):
            processor = PoseProcessor()
            pose_frames, skeleton_frames, fps, ball_detections, ball_air_time_s, lob_type = process_video(tmp_path, processor)
            processor.close()

        if not pose_frames:
            st.error("No person detected in the video. Please upload a clip with a visible player.")
            return

        if not skeleton_frames:
            st.error("Could not read video frames.")
            return

        # Physics (use body height from first valid frame for calibration)
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
            st.warning(
                "Insufficient pose data for physics analysis. "
                "Use a clip with the player fully visible (full body, good lighting)."
            )
            result = PhysicsResult(
                hang_time_s=0.0,
                max_vertical_inches=0.0,
                rotation_degrees=0.0,
                frames_airborne=0,
                start_hip_y=0.0,
                min_hip_y=0.0,
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
        )

        # Sidebar: judges scorecard + rejection model
        if analysis.is_dunk:
            st.sidebar.subheader("Dunk classification")
            st.sidebar.markdown(f"**{analysis.dunk_type}**")
            st.sidebar.caption(analysis.primary_category)
        else:
            st.sidebar.subheader("Rejection model")
            st.sidebar.error(f"NOT A DUNK — {analysis.non_dunk_type}")
            st.sidebar.caption(analysis.rejection_reason)

        st.sidebar.markdown("---")
        st.sidebar.subheader("Core metrics")
        st.sidebar.metric("Hang Time", f"{analysis.hang_time_s:.2f}s")
        st.sidebar.metric("Max Vertical", f"{analysis.max_vertical_inches:.1f} in")
        st.sidebar.metric("Rotation", f"{analysis.rotation_degrees:.0f}°")
        st.sidebar.metric("Ball Air Time", f"{analysis.ball_air_time_s:.2f}s" if analysis.ball_air_time_s > 0 else "—")
        st.sidebar.metric("Model confidence", f"{analysis.model_confidence:.2f}" if analysis.model_confidence > 0 else "—")

        comps = analysis.score_components
        st.sidebar.markdown("---")
        st.sidebar.subheader("Score Breakdown (40–50)")
        st.sidebar.metric("Base", f"{comps.base_score:.1f}")
        st.sidebar.metric("Hang Bonus", f"+{comps.hang_time_bonus:.1f}")
        st.sidebar.metric("Vertical Bonus", f"+{comps.vertical_bonus:.1f}")
        st.sidebar.metric("Rotation Bonus", f"+{comps.rotation_bonus:.1f}")
        st.sidebar.metric("Trick Bonus", f"+{comps.trick_bonus:.1f}")
        st.sidebar.metric("Lob Bonus", f"+{comps.lob_bonus:.1f}" if comps.lob_bonus > 0 else "—")
        st.sidebar.metric("Distance Bonus", f"+{comps.distance_bonus:.1f}" if comps.distance_bonus > 0 else "—")
        st.sidebar.metric("Reliability Adj.", f"{comps.reliability_adjustment:+.1f}" if comps.reliability_adjustment else "—")
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"### Predicted Score: **{analysis.final_contest_score:.1f}**")
        st.sidebar.caption("NBA contest scale: 40–50")

        if analysis.is_dunk:
            st.success("Dunk detected. Review DunkR8 output below.")
        else:
            st.warning("Clip rejected as a dunk. Review the rejection reason and diagnostics below.")

        # Single tracked video output (autoplay + controls)
        st.subheader("DunkR8 Tracking View")
        st.caption("Single playback output. Green = pose. Orange = ball.")
        overlay_bytes = write_overlay_video_to_bytes(skeleton_frames, fps)
        if overlay_bytes:
            st.video(
                overlay_bytes,
                format="video/mp4",
                start_time=0,
                autoplay=True,
                muted=True,
                loop=True,
                width="stretch",
            )
        else:
            st.warning("Could not render tracking video output.")

        label_text = analysis.dunk_type if analysis.is_dunk else f"NOT A DUNK ({analysis.non_dunk_type})"
        aux_text = analysis.primary_category if analysis.is_dunk else analysis.rejection_reason
        st.markdown(
            f'<div style="'
            f'text-align: center; margin: 1.5rem 0; padding: 1.5rem 1.75rem; '
            f'background: rgba(18,18,18,0.95); border: 1px solid rgba(255,255,255,0.08); '
            f'border-radius: 14px;'
            f'">'
            f'<div style="font-size: 0.75rem; color: #666; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.1em;">Classification Output</div>'
            f'<div style="font-size: clamp(1.35rem, 3.4vw, 1.95rem); font-weight: 700; color: #fff; line-height: 1.35;">{label_text}</div>'
            f'<div style="font-size: 0.9rem; color: #aaa; margin-top: 0.5rem;">{aux_text}</div>'
            f'<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 1rem; margin-top: 1rem; font-size: 0.85rem;">'
            f'<span style="color: #888;">Primary category: <strong style="color: #ccc;">{analysis.primary_category}</strong></span>'
            f'<span style="color: #888;">Alley-oop: <strong style="color: #ccc;">{"Yes" if analysis.alley_oop else "No"}</strong></span>'
            f'<span style="color: #888;">Self-lob: <strong style="color: #ccc;">{"Yes" if analysis.self_lob else "No"}</strong></span>'
            f'<span style="color: #888;">Rotation band: <strong style="color: #ccc;">{analysis.rotation_band}</strong></span>'
            f'<span style="color: #888;">Over object: <strong style="color: #ccc;">{"Yes" if analysis.over_object else "No"}</strong></span>'
            f'<span style="color: #888;">Model: <strong style="color: #ccc;">{analysis.model_prediction or "rule-only"} ({analysis.model_confidence:.2f})</strong></span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        st.subheader("Core output (requested fields)")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Hang time", f"{analysis.hang_time_s:.2f} s")
            st.metric("Frames airborne", analysis.frames_airborne)
            st.metric("Ball air time", f"{analysis.ball_air_time_s:.2f} s" if analysis.ball_air_time_s > 0 else "—")
        with col2:
            st.metric("Max vertical", f"{analysis.max_vertical_inches:.1f} in")
            st.metric("Apex height", f"{analysis.apex_height_ft:.2f} ft")
            st.metric("Rotation", f"{analysis.rotation_degrees:.0f}°")
        with col3:
            st.metric("Dunk type", analysis.dunk_type if analysis.is_dunk else analysis.non_dunk_type)
            st.metric("Difficulty tier", analysis.difficulty_tier)
            st.metric("Style grade", analysis.style_grade)
        with col4:
            st.metric("Comparable tier", analysis.comparable_tier)
            st.metric("Score", f"{analysis.final_contest_score:.1f} / 50")
            st.metric("Alley-oop", "Yes" if analysis.alley_oop else "No")
            st.metric("Model prediction", analysis.model_prediction or "rule-only")

        with st.expander("Detection diagnostics"):
            st.caption("Pass/fail checks used by dunk validation.")
            st.json(analysis.validation_checks)

        st.subheader("Biomechanics + motion cues")
        b1, b2, b3, b4, b5 = st.columns(5)
        with b1:
            st.metric("Takeoff feet", analysis.takeoff_foot_count)
            st.metric("Takeoff distance", f"{analysis.takeoff_distance_ft:.1f} ft")
        with b2:
            st.metric("Approach speed", f"{analysis.approach_speed_ft_s:.1f} ft/s")
            st.metric("Gather time", f"{analysis.gather_time_s:.2f} s")
        with b3:
            st.metric("Leg tuck angle", f"{analysis.leg_tuck_angle_deg:.0f}°")
            st.metric("Shoulder flexion", f"{analysis.shoulder_flexion_angle_deg:.0f}°")
        with b4:
            st.metric("Elbow ext velocity", f"{analysis.elbow_extension_velocity_deg_s:.0f}°/s")
            st.metric("Arm path curvature", f"{analysis.arm_path_curvature_deg:.0f}°")
        with b5:
            st.metric("Ball path arc", f"{analysis.ball_path_arc_ft:.2f} ft")
            st.metric("Lob mode", analysis.lob_type)

        st.subheader("Final contest score")
        score_color = "#22c55e" if analysis.final_contest_score >= 47 else "#eab308" if analysis.final_contest_score >= 43 else "#71717a"
        st.markdown(
            f"""
            <div style="
                text-align: center;
                font-size: 52px;
                font-weight: 700;
                color: {score_color};
                padding: 24px;
                border: 2px solid {score_color};
                border-radius: 14px;
                background: rgba(0,0,0,0.3);
            ">
                {analysis.final_contest_score:.1f}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("40–50 contest scale · ontology + biomechanics + rejection model")

    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except (PermissionError, OSError):
            pass  # Windows may still have file open; ignore


if __name__ == "__main__":
    main()
