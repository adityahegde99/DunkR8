"""
Slam Dunk Score Predictor - Dunk Classifier
Rule-based by default; can use a trained model when you provide labeled clips in training_dunks/
and run the training script. Same pose/physics features drive both.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from physics_engine import PhysicsResult, PoseFrame

LOB_AIR_TIME_THRESHOLD_S = 0.35

# All dunk types the model can predict (must match rule-based labels)
DUNK_TYPES = [
    "720 Spin", "540 Spin", "360 Spin",
    "Eastbay (Between-the-Legs)", "Windmill", "Reverse Dunk", "Tomahawk",
    "Two-Handed Power", "One-Handed", "Standard",
]
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = DEFAULT_MODEL_DIR / "dunk_classifier.joblib"
MODEL_META_PATH = DEFAULT_MODEL_DIR / "dunk_classifier_meta.json"
MIN_MODEL_CONFIDENCE = 0.35  # Below this, fall back to rules

# Windmill: only when one arm does a full circle and the other barely moves (strict)
WINDMILL_FULL_CIRCLE_MIN = 330.0   # One arm must go essentially full 360°
WINDMILL_OTHER_ARM_MAX = 55.0      # Other arm must be nearly still (no circle)
WINDMILL_BODY_ROTATION_MAX = 180.0 # Body rotation low (windmill is arm, not body spin)
MIN_VERTICAL_FOR_STYLE = 20.0
MIN_HANG_FOR_COMPLEX = 0.32
# One-handed: arm sweep must be low (no windmill possible)
ONE_HANDED_MAX_SWEEP = 100.0


def physics_result_to_feature_vector(
    result: PhysicsResult,
    ball_air_time_s: float = 0.0,
    lob_type: str = "unknown",
) -> List[float]:
    """
    Build a fixed-size numeric feature vector for training or inference.
    Order must stay fixed so saved models stay valid.
    """
    ls = getattr(result, "left_wrist_angle_sweep_deg", 0.0)
    rs = getattr(result, "right_wrist_angle_sweep_deg", 0.0)
    wbh = 1.0 if getattr(result, "wrist_went_below_hip", False) else 0.0
    wbm = 1.0 if getattr(result, "wrist_below_hip_near_midline", False) else 0.0
    two = 1.0 if getattr(result, "two_hands_cue", False) else 0.0
    lob_backboard = 1.0 if lob_type == "backboard" else 0.0
    lob_bounce = 1.0 if lob_type == "bounce" else 0.0
    return [
        result.rotation_degrees,
        result.hang_time_s,
        result.max_vertical_inches,
        float(result.frames_airborne),
        result.start_hip_y,
        result.min_hip_y,
        ls,
        rs,
        getattr(result, "max_wrist_radius", 0.0),
        wbh,
        wbm,
        two,
        ball_air_time_s,
        lob_backboard,
        lob_bounce,
    ]


@dataclass
class DunkClassification:
    label: str
    description: str
    base_score: float
    rotation_degrees: float
    hang_time_s: float
    max_vertical_inches: float
    rotation_category: str = ""
    dunk_type: str = ""
    is_alley_oop: bool = False
    is_over_object: bool = False


def _base_score(dunk_type: str) -> float:
    m = {
        "720 Spin": 38.0, "540 Spin": 36.0, "360 Spin": 34.0,
        "Eastbay (Between-the-Legs)": 33.0, "Windmill": 32.0,
        "Reverse Dunk": 30.0, "Tomahawk": 27.0, "Two-Handed Power": 25.0,
        "One-Handed": 22.0, "Standard": 22.0,
    }
    return m.get(dunk_type, 24.0)


def _description_for_type(dunk_type: str) -> str:
    """Short description for a dunk type (used when model predicts)."""
    d = {
        "720 Spin": "Two full rotations (body).",
        "540 Spin": "One and a half rotations (body).",
        "360 Spin": "Full 360° (body).",
        "Eastbay (Between-the-Legs)": "Ball goes between the legs then dunk.",
        "Windmill": "One arm goes full circle then dunk.",
        "Reverse Dunk": "Body reverses to the rim.",
        "Tomahawk": "Power tomahawk — ball brought back then down.",
        "Two-Handed Power": "Two-handed finish (wrists close).",
        "One-Handed": "One-handed finish.",
        "Standard": "Standard dunk.",
    }
    return d.get(dunk_type, "Dunk.")


def _rotation_category(rotation_degrees: float) -> str:
    r = rotation_degrees
    if r >= 630:
        return "720°"
    if r >= 450:
        return "540°"
    if r >= 300:
        return "360°"
    if r >= 150:
        return "~180°"
    if r >= 60:
        return "~90°"
    return "0°"


def _score_dunk_types(result: PhysicsResult) -> List[Tuple[str, str, float]]:
    """
    Score each dunk type 0..1 from strict multi-criteria. Returns list of (dunk_type, description, score).
    Pick argmax; if max score < threshold, return Standard. No bias toward any one type.
    """
    r = result.rotation_degrees
    h = result.hang_time_s
    v = result.max_vertical_inches
    ls = getattr(result, "left_wrist_angle_sweep_deg", 0.0)
    rs = getattr(result, "right_wrist_angle_sweep_deg", 0.0)
    wrist_below = getattr(result, "wrist_went_below_hip", False)
    wrist_midline = getattr(result, "wrist_below_hip_near_midline", False)
    two_hands = getattr(result, "two_hands_cue", False)
    max_sweep = max(ls, rs)
    min_sweep = min(ls, rs)

    out: List[Tuple[str, str, float]] = []

    # ---- 720 Spin: body rotation only ----
    if r >= 600:
        score = min(1.0, (r - 600) / 100.0 + 0.7)
        out.append(("720 Spin", "Two full rotations (body).", score))

    # ---- 540 Spin ----
    if r >= 420:
        score = min(1.0, (r - 420) / 80.0 + 0.6)
        out.append(("540 Spin", "One and a half rotations (body).", score))

    # ---- 360 Spin: body rotation 300+ ----
    if r >= 280:
        score = min(1.0, (r - 280) / 80.0 + 0.5)
        out.append(("360 Spin", "Full 360° (body).", score))

    # ---- Windmill: one arm FULL circle (330°+), other arm nearly still (55° or less), body not spinning ----
    if (
        max_sweep >= WINDMILL_FULL_CIRCLE_MIN
        and min_sweep <= WINDMILL_OTHER_ARM_MAX
        and r < WINDMILL_BODY_ROTATION_MAX
        and h >= 0.38
        and v >= 22
    ):
        score = min(1.0, 0.5 + (max_sweep - 330) / 50.0 * 0.2 + (1.0 - min_sweep / 55.0) * 0.2)
        out.append(("Windmill", "One arm goes full circle then dunk.", score))

    # ---- Eastbay: ball between legs — wrist below hip AND near midline (between the legs) ----
    if wrist_below and wrist_midline and 50 <= r < 250 and v >= MIN_VERTICAL_FOR_STYLE and h >= MIN_HANG_FOR_COMPLEX:
        score = 0.5 + (0.2 if wrist_below else 0) + (0.3 if wrist_midline else 0)
        out.append(("Eastbay (Between-the-Legs)", "Ball goes between the legs (wrist low and between hips) then dunk.", min(1.0, score)))

    # ---- Reverse Dunk: body ~180°, not windmill, not Eastbay ----
    if 130 <= r < 280 and v >= MIN_VERTICAL_FOR_STYLE and max_sweep < WINDMILL_FULL_CIRCLE_MIN and not (wrist_below and wrist_midline):
        score = min(1.0, 0.3 + (r - 130) / 200.0)
        out.append(("Reverse Dunk", "Body reverses to the rim.", score))

    # ---- Tomahawk: power, arm back then down — moderate sweep, not full circle ----
    if 40 <= r < 120 and 80 <= max_sweep < 280 and v >= 24 and max_sweep < WINDMILL_FULL_CIRCLE_MIN:
        score = min(1.0, 0.35 + max_sweep / 400.0)
        out.append(("Tomahawk", "Power tomahawk — ball brought back then down.", score))

    # ---- Two-Handed Power: both wrists close, low rotation, low arm sweep ----
    if two_hands and r < 40 and max_sweep < 150 and v >= MIN_VERTICAL_FOR_STYLE:
        score = 0.5 + 0.2 * min(v / 35.0, 1.0)
        out.append(("Two-Handed Power", "Two-handed finish (wrists close).", min(1.0, score)))

    # ---- One-Handed: low rotation, low arm sweep (no windmill), no two-hands, no Eastbay ----
    if r < 55 and not two_hands and max_sweep < ONE_HANDED_MAX_SWEEP and not (wrist_below and wrist_midline) and v >= 18:
        score = 0.5 + 0.35 * (1.0 - max_sweep / ONE_HANDED_MAX_SWEEP)
        out.append(("One-Handed", "One-handed finish — just goes in.", min(1.0, score)))

    # ---- Standard: baseline when nothing else fits well ----
    out.append(("Standard", "Standard dunk.", 0.25))

    return out


def _classify_by_score(result: PhysicsResult) -> Tuple[str, str]:
    """Pick dunk type with highest score. Standard wins only if no type scores above 0.4."""
    scored = _score_dunk_types(result)
    best = max(scored, key=lambda x: x[2])
    dunk_type, desc, score = best
    if dunk_type == "Standard":
        others = [s for s in scored if s[0] != "Standard"]
        if others:
            top = max(others, key=lambda x: x[2])
            if top[2] >= 0.4:
                return top[0], top[1]
    return dunk_type, desc


class DunkClassifier:
    """
    Classification by trained model (if present) or rule-based. Put labeled clips in
    training_dunks/<dunk_type>/video.mp4 and run train_dunk_classifier.py to train.
    """

    def __init__(self, model_path: Optional[Path] = None):
        self._model = None
        self._classes: List[str] = []
        path = model_path or MODEL_PATH
        if path.exists():
            try:
                import joblib
                obj = joblib.load(path)
                if isinstance(obj, dict):
                    self._model = obj.get("model")
                    self._classes = obj.get("classes", DUNK_TYPES)
                else:
                    self._model = obj
                    self._classes = list(getattr(obj, "classes_", DUNK_TYPES))
                if not self._classes:
                    meta_path = path.parent / "dunk_classifier_meta.json"
                    if meta_path.exists():
                        import json
                        with open(meta_path, "r") as f:
                            self._classes = json.load(f).get("classes", DUNK_TYPES)
            except Exception:
                self._model = None
                self._classes = []

    def classify(
        self,
        result: PhysicsResult,
        pose_frames: Optional[List[PoseFrame]] = None,
        ball_air_time_s: float = 0.0,
        lob_type: str = "unknown",
    ) -> DunkClassification:
        r = result.rotation_degrees
        h = result.hang_time_s
        v = result.max_vertical_inches

        rotation_cat = _rotation_category(r)
        dunk_type, desc = _classify_by_score(result)
        if self._model is not None and self._classes:
            try:
                import numpy as np
                vec = physics_result_to_feature_vector(result, ball_air_time_s, lob_type)
                X = np.asarray([vec], dtype=np.float64)
                pred_label = self._model.predict(X)[0]
                proba = self._model.predict_proba(X)[0]
                pred_idx = list(self._model.classes_).index(pred_label) if pred_label in self._model.classes_ else -1
                conf = proba[pred_idx] if 0 <= pred_idx < len(proba) else 0.0
                if conf >= MIN_MODEL_CONFIDENCE and pred_label in self._classes:
                    dunk_type = pred_label
                    desc = _description_for_type(dunk_type)
            except Exception:
                pass
        base = _base_score(dunk_type)

        is_alley_oop = ball_air_time_s >= LOB_AIR_TIME_THRESHOLD_S
        is_over_object = False

        parts = []
        if is_alley_oop:
            if lob_type == "backboard":
                parts.append("Off the Backboard")
            elif lob_type == "bounce":
                parts.append("Off the Bounce")
            else:
                parts.append("Lob / Alley-oop")
        parts.append(dunk_type)
        if is_over_object:
            parts.append("over object")
        label = " ".join(parts)

        if is_alley_oop:
            if lob_type == "backboard":
                desc = f"Ball off the backboard into {dunk_type}. {desc}"
            elif lob_type == "bounce":
                desc = f"Ball off the bounce into {dunk_type}. {desc}"
            else:
                desc = f"Lob or alley-oop into {dunk_type}. {desc}"

        return DunkClassification(
            label=label,
            description=desc,
            base_score=base,
            rotation_degrees=r,
            hang_time_s=h,
            max_vertical_inches=v,
            rotation_category=rotation_cat,
            dunk_type=dunk_type,
            is_alley_oop=is_alley_oop,
            is_over_object=is_over_object,
        )
