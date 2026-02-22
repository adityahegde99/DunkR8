"""
Lightweight trainable prototype model for dunk taxonomy support.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple
import json
import math

from physics_engine import PhysicsResult
from dunk_ontology import DUNK_ONTOLOGY


MODEL_PATH = Path(__file__).resolve().parent / "models" / "ontology_prototypes.json"

FEATURE_KEYS = [
    "rotation_degrees",
    "hang_time_s",
    "max_vertical_inches",
    "left_wrist_sweep_deg",
    "right_wrist_sweep_deg",
    "wrist_below_hip",
    "wrist_below_hip_midline",
    "two_hands_cue",
    "ball_air_time_s",
]

FEATURE_SCALES = {
    "rotation_degrees": 360.0,
    "hang_time_s": 1.0,
    "max_vertical_inches": 42.0,
    "left_wrist_sweep_deg": 360.0,
    "right_wrist_sweep_deg": 360.0,
    "wrist_below_hip": 1.0,
    "wrist_below_hip_midline": 1.0,
    "two_hands_cue": 1.0,
    "ball_air_time_s": 2.0,
}

FEATURE_WEIGHTS = {
    "rotation_degrees": 1.4,
    "hang_time_s": 0.9,
    "max_vertical_inches": 1.1,
    "left_wrist_sweep_deg": 1.0,
    "right_wrist_sweep_deg": 1.0,
    "wrist_below_hip": 1.2,
    "wrist_below_hip_midline": 1.4,
    "two_hands_cue": 1.0,
    "ball_air_time_s": 0.7,
}


def build_feature_dict(
    result: PhysicsResult,
    ball_air_time_s: float,
) -> Dict[str, float]:
    return {
        "rotation_degrees": float(result.rotation_degrees),
        "hang_time_s": float(result.hang_time_s),
        "max_vertical_inches": float(result.max_vertical_inches),
        "left_wrist_sweep_deg": float(getattr(result, "left_wrist_angle_sweep_deg", 0.0)),
        "right_wrist_sweep_deg": float(getattr(result, "right_wrist_angle_sweep_deg", 0.0)),
        "wrist_below_hip": 1.0 if getattr(result, "wrist_went_below_hip", False) else 0.0,
        "wrist_below_hip_midline": 1.0 if getattr(result, "wrist_below_hip_near_midline", False) else 0.0,
        "two_hands_cue": 1.0 if getattr(result, "two_hands_cue", False) else 0.0,
        "ball_air_time_s": float(ball_air_time_s),
    }


def _stem_to_tokens(stem: str) -> str:
    cleaned = stem.strip().lower().replace("-", " ").replace("_", " ")
    cleaned = " ".join(cleaned.split())
    return cleaned


def normalize_dunk_label(raw_label: str) -> Optional[str]:
    tokens = _stem_to_tokens(raw_label)
    if not tokens:
        return None

    # Exact match against canonical names first.
    for canonical in DUNK_ONTOLOGY.keys():
        if _stem_to_tokens(canonical) == tokens:
            return canonical

    # Common aliases used in filenames.
    alias_map = {
        "reversedunk": "180 Dunk",
        "reverse dunk": "180 Dunk",
        "alley 360": "Alley-Oop 360",
        "alley 360 dunk": "Alley-Oop 360",
        "alley oop 360": "Alley-Oop 360",
        "alley power": "Alley-Oop Power",
        "alley power dunk": "Alley-Oop Power",
        "alley oop power": "Alley-Oop Power",
        "windmill": "Standard Windmill",
        "reverse windmill": "Reverse Windmill",
        "360 windmill": "360 Windmill",
        "one hand": "One-Hand Power Dunk",
        "one handed": "One-Hand Power Dunk",
        "two hand": "Two-Hand Power Dunk",
        "two handed": "Two-Hand Power Dunk",
        "twohand": "Two-Hand Power Dunk",
        "twohands": "Two-Hand Power Dunk",
        "2 hand": "Two-Hand Power Dunk",
        "2hand": "Two-Hand Power Dunk",
        "reverse power": "Two-Hand Reverse Power",
        "tomahawk": "Tomahawk (Single Arm)",
        "double tomahawk": "Double Tomahawk",
        "statue of liberty": "Statue of Liberty",
        "180": "180 Dunk",
        "180 dunk": "180 Dunk",
        "360": "360 Dunk",
        "360 dunk": "360 Dunk",
        "540": "540 Dunk",
        "540 dunk": "540 Dunk",
        "eastbay": "Eastbay (Between-the-Legs)",
        "eastybay": "Eastbay (Between-the-Legs)",
        "between the legs": "Eastbay (Between-the-Legs)",
        "reverse eastbay": "Reverse Eastbay",
        "lob eastbay": "Lob Eastbay",
        "double eastbay": "Double Eastbay",
        "behind back": "Behind-Back Dunk",
        "360 behind back": "360 Behind-Back",
        "double pump": "Double Pump",
        "reverse double pump": "Reverse Double Pump",
        "alley oop": "Alley-Oop Power",
        "alley oop reverse": "Alley-Oop Reverse",
        "alley oop 360": "Alley-Oop 360",
        "lob windmill": "Lob Windmill",
        "free throw line": "Free Throw Line Dunk",
        "baseline glide": "Baseline Glide",
        "putback": "Putback Dunk",
        "tip dunk": "Tip Dunk",
        "behindback": "Behind-Back Dunk",
        "doubleeastbay": "Double Eastbay",
        "eastbay3": "Eastbay (Between-the-Legs)",
        "eastybay4": "Eastbay (Between-the-Legs)",
        "reverse": "180 Dunk",
        "long dunk": "Free Throw Line Dunk",
        "long": "Free Throw Line Dunk",
    }
    if tokens in alias_map:
        return alias_map[tokens]

    # Contains-based fallback.
    contains_map = [
        ("windmill", "Standard Windmill"),
        ("eastbay", "Eastbay (Between-the-Legs)"),
        ("between", "Eastbay (Between-the-Legs)"),
        ("tomahawk", "Tomahawk (Single Arm)"),
        ("behind", "Behind-Back Dunk"),
        ("double pump", "Double Pump"),
        ("putback", "Putback Dunk"),
        ("free throw", "Free Throw Line Dunk"),
        ("baseline", "Baseline Glide"),
    ]
    for needle, canonical in contains_map:
        if needle in tokens:
            return canonical
    return None


def _weighted_distance(a: Dict[str, float], b: Dict[str, float]) -> float:
    total = 0.0
    for key in FEATURE_KEYS:
        scale = FEATURE_SCALES.get(key, 1.0) or 1.0
        w = FEATURE_WEIGHTS.get(key, 1.0)
        da = (a.get(key, 0.0) - b.get(key, 0.0)) / scale
        total += w * da * da
    return math.sqrt(total)


def load_prototype_model(path: Path = MODEL_PATH) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    protos = data.get("prototypes")
    if not isinstance(protos, dict) or not protos:
        return None
    return data


def predict_from_model(
    features: Dict[str, float],
    model_data: Optional[Dict],
) -> Tuple[str, float, float]:
    """
    Returns (label, confidence, distance). Empty label if no model/prediction.
    """
    if not model_data:
        return "", 0.0, float("inf")
    prototypes = model_data.get("prototypes", {})
    counts = model_data.get("counts", {})
    class_radii = model_data.get("class_radii", {})
    best_label = ""
    best_dist = float("inf")
    best_norm = float("inf")
    for label, proto in prototypes.items():
        dist = _weighted_distance(features, proto)
        radius = max(0.12, float(class_radii.get(label, 0.22)))
        norm_dist = dist / radius
        # Small boost for classes with more support clips.
        support = max(1, int(counts.get(label, 1)))
        norm_adj = norm_dist / (1.0 + min(1.2, 0.08 * (support - 1)))
        if norm_adj < best_norm:
            best_norm = norm_adj
            best_dist = dist
            best_label = label
    if not best_label:
        return "", 0.0, float("inf")

    # Convert normalized distance to confidence in [0,1].
    confidence = math.exp(-0.5 * (best_norm ** 2))
    confidence = max(0.0, min(1.0, confidence))
    # Out-of-distribution guard.
    if best_norm > 3.0:
        return "", 0.0, best_dist
    return best_label, confidence, best_dist

