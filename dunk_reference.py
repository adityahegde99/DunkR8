"""
Reference dunk signatures: classify by finding the closest reference.
Put one clip per dunk type in reference_dunks/clips/ (e.g. one_handed.mp4, windmill.mp4),
then run build_references_from_clips() to generate features. New videos are compared to these.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Feature vector keys (must match PhysicsResult + arm metrics)
FEATURE_KEYS = [
    "rotation_degrees",
    "max_vertical_inches",
    "hang_time_s",
    "left_wrist_sweep_deg",
    "right_wrist_sweep_deg",
    "wrist_below_hip",  # 0 or 1
]

# Weights for distance (higher = feature matters more for matching)
WEIGHTS = {
    "rotation_degrees": 1.5,
    "max_vertical_inches": 0.8,
    "hang_time_s": 0.6,
    "left_wrist_sweep_deg": 1.2,
    "right_wrist_sweep_deg": 1.2,
    "wrist_below_hip": 2.0,
}

# Normalize so different scales don't dominate (typical ranges)
SCALES = {
    "rotation_degrees": 360.0,
    "max_vertical_inches": 40.0,
    "hang_time_s": 1.0,
    "left_wrist_sweep_deg": 360.0,
    "right_wrist_sweep_deg": 360.0,
    "wrist_below_hip": 1.0,
}

REFERENCE_DIR = Path(__file__).parent / "reference_dunks"
FEATURES_DIR = REFERENCE_DIR / "features"
CLIPS_DIR = REFERENCE_DIR / "clips"

# Default canonical signatures (used if no clips/features exist)
DEFAULT_SIGNATURES = {
    "One-Handed": {
        "rotation_degrees": 5,
        "max_vertical_inches": 24,
        "hang_time_s": 0.4,
        "left_wrist_sweep_deg": 0,
        "right_wrist_sweep_deg": 0,
        "wrist_below_hip": 0,
        "description": "Standard dunk — just goes in.",
    },
    "Two-Handed Power": {
        "rotation_degrees": 10,
        "max_vertical_inches": 28,
        "hang_time_s": 0.45,
        "left_wrist_sweep_deg": 0,
        "right_wrist_sweep_deg": 0,
        "wrist_below_hip": 0,
        "description": "Two-handed power finish.",
    },
    "Windmill": {
        "rotation_degrees": 120,
        "max_vertical_inches": 28,
        "hang_time_s": 0.5,
        "left_wrist_sweep_deg": 320,
        "right_wrist_sweep_deg": 0,
        "wrist_below_hip": 0,
        "description": "Arm goes full around (windmill).",
    },
    "360 Spin": {
        "rotation_degrees": 340,
        "max_vertical_inches": 26,
        "hang_time_s": 0.55,
        "left_wrist_sweep_deg": 40,
        "right_wrist_sweep_deg": 40,
        "wrist_below_hip": 0,
        "description": "Full 360° (body).",
    },
    "540 Spin": {
        "rotation_degrees": 480,
        "max_vertical_inches": 27,
        "hang_time_s": 0.6,
        "left_wrist_sweep_deg": 60,
        "right_wrist_sweep_deg": 60,
        "wrist_below_hip": 0,
        "description": "One and a half rotations (body).",
    },
    "720 Spin": {
        "rotation_degrees": 660,
        "max_vertical_inches": 28,
        "hang_time_s": 0.65,
        "left_wrist_sweep_deg": 80,
        "right_wrist_sweep_deg": 80,
        "wrist_below_hip": 0,
        "description": "Two full rotations (body).",
    },
    "Eastbay (Between-the-Legs)": {
        "rotation_degrees": 140,
        "max_vertical_inches": 28,
        "hang_time_s": 0.5,
        "left_wrist_sweep_deg": 180,
        "right_wrist_sweep_deg": 180,
        "wrist_below_hip": 1,
        "description": "Ball between the legs then dunk.",
    },
    "Reverse Dunk": {
        "rotation_degrees": 180,
        "max_vertical_inches": 26,
        "hang_time_s": 0.45,
        "left_wrist_sweep_deg": 60,
        "right_wrist_sweep_deg": 60,
        "wrist_below_hip": 0,
        "description": "Body reverses to the rim.",
    },
    "Tomahawk": {
        "rotation_degrees": 60,
        "max_vertical_inches": 28,
        "hang_time_s": 0.45,
        "left_wrist_sweep_deg": 80,
        "right_wrist_sweep_deg": 0,
        "wrist_below_hip": 0,
        "description": "Power tomahawk.",
    },
}


def _result_to_feature_vector(result) -> Dict[str, float]:
    """Build feature dict from PhysicsResult."""
    return {
        "rotation_degrees": result.rotation_degrees,
        "max_vertical_inches": result.max_vertical_inches,
        "hang_time_s": result.hang_time_s,
        "left_wrist_sweep_deg": getattr(result, "left_wrist_angle_sweep_deg", 0.0),
        "right_wrist_sweep_deg": getattr(result, "right_wrist_angle_sweep_deg", 0.0),
        "wrist_below_hip": 1.0 if getattr(result, "wrist_went_below_hip", False) else 0.0,
    }


def _weighted_distance(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Normalized weighted L2 distance between two feature dicts."""
    total = 0.0
    for key in FEATURE_KEYS:
        scale = SCALES.get(key, 1.0)
        if scale <= 0:
            scale = 1.0
        w = WEIGHTS.get(key, 1.0)
        va = a.get(key, 0.0) / scale
        vb = b.get(key, 0.0) / scale
        total += w * (va - vb) ** 2
    return total ** 0.5


def load_references() -> Dict[str, Dict[str, float]]:
    """Load reference signatures from reference_dunks/features/*.json, or fall back to defaults."""
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = list(FEATURES_DIR.glob("*.json"))
    if not paths:
        for name, sig in DEFAULT_SIGNATURES.items():
            path = FEATURES_DIR / f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.json"
            with open(path, "w") as f:
                json.dump({k: v for k, v in sig.items() if k in FEATURE_KEYS + ["description"]}, f, indent=2)
        return dict(DEFAULT_SIGNATURES)
    stem_to_name = {
        "one_handed": "One-Handed", "two_handed": "Two-Handed Power",
        "windmill": "Windmill", "360_spin": "360 Spin", "540_spin": "540 Spin", "720_spin": "720 Spin",
        "eastbay": "Eastbay (Between-the-Legs)", "reverse": "Reverse Dunk", "tomahawk": "Tomahawk",
    }
    refs = dict(DEFAULT_SIGNATURES)
    for path in paths:
        try:
            with open(path) as f:
                data = json.load(f)
            stem = path.stem.lower().replace(" ", "_").replace("(", "").replace(")", "")
            name = stem_to_name.get(stem, stem.replace("_", " ").title())
            if "description" not in data and name in DEFAULT_SIGNATURES:
                data["description"] = DEFAULT_SIGNATURES[name].get("description", "")
            refs[name] = data
        except Exception:
            continue
    return refs


def find_closest_reference(result, references: Optional[Dict[str, Dict[str, float]]] = None) -> Tuple[str, str, float]:
    """
    Find the reference dunk type closest to the given PhysicsResult.
    Returns (dunk_type, description, distance). Lower distance = better match.
    """
    if references is None:
        references = load_references()
    vec = _result_to_feature_vector(result)
    best_name = "One-Handed"
    best_desc = DEFAULT_SIGNATURES["One-Handed"]["description"]
    best_dist = float("inf")
    for name, ref in references.items():
        ref_vec = {k: ref.get(k, 0.0) for k in FEATURE_KEYS}
        d = _weighted_distance(vec, ref_vec)
        if d < best_dist:
            best_dist = d
            best_name = name
            best_desc = ref.get("description", best_desc)
    return best_name, best_desc, best_dist


def build_references_from_clips(
    clips_dir: Optional[Path] = None,
    processor_factory=None,
) -> Dict[str, Dict[str, float]]:
    """
    Process each video in reference_dunks/clips/ and save a feature JSON per dunk type.
    Name clips like: one_handed.mp4, windmill.mp4, 360_spin.mp4, eastbay.mp4, reverse.mp4, tomahawk.mp4, two_handed.mp4.
    Returns the new references dict.
    """
    clips_dir = clips_dir or CLIPS_DIR
    if not clips_dir.exists():
        CLIPS_DIR.mkdir(parents=True, exist_ok=True)
        return load_references()
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    from pose_processor import process_video
    from physics_engine import PhysicsEngine
    if processor_factory is None:
        from pose_processor import PoseProcessor
        def default_factory():
            return PoseProcessor()
        processor_factory = default_factory
    refs = {}
    # Map file stem to reference name (e.g. 360_spin -> 360 Spin)
    stem_to_name = {
        "one_handed": "One-Handed", "two_handed": "Two-Handed Power",
        "windmill": "Windmill", "360_spin": "360 Spin", "540_spin": "540 Spin", "720_spin": "720 Spin",
        "eastbay": "Eastbay (Between-the-Legs)", "reverse": "Reverse Dunk", "tomahawk": "Tomahawk",
    }
    for ext in ("*.mp4", "*.mov", "*.avi", "*.webm"):
        for path in sorted(clips_dir.glob(ext)):
            name_key = path.stem.lower().replace(" ", "_").replace("(", "").replace(")", "")
            display_name = stem_to_name.get(name_key, name_key.replace("_", " ").title())
            try:
                processor = processor_factory()
                pose_frames, skeleton_frames, fps, _, _, _ = process_video(str(path), processor)
                processor.close()
                if not pose_frames:
                    continue
                frame_h = skeleton_frames[0].shape[0] if skeleton_frames else 720
                body_norm = next((p.body_height_norm for p in pose_frames if getattr(p, "body_height_norm", None)), None)
                physics = PhysicsEngine(fps=fps, frame_height=frame_h, body_height_norm=body_norm)
                result = physics.compute(pose_frames)
                if result is None:
                    continue
                vec = _result_to_feature_vector(result)
                vec["description"] = DEFAULT_SIGNATURES.get(display_name, {}).get("description", f"{display_name} reference.")
                out_path = FEATURES_DIR / f"{name_key}.json"
                with open(out_path, "w") as f:
                    json.dump(vec, f, indent=2)
                refs[display_name] = vec
            except Exception:
                continue
    return refs if refs else load_references()


def get_clips_folder() -> Path:
    """Return the path where user should place reference clips."""
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    return CLIPS_DIR
