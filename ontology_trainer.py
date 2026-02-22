"""
Train prototype taxonomy model from clips in reference_dunks/clips.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime, timezone
import json

from pose_processor import PoseProcessor, process_video
from physics_engine import PhysicsEngine
from ontology_model import (
    MODEL_PATH,
    FEATURE_KEYS,
    FEATURE_SCALES,
    FEATURE_WEIGHTS,
    build_feature_dict,
    normalize_dunk_label,
)


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
DEFAULT_CLIPS_DIR = Path(__file__).resolve().parent / "reference_dunks" / "clips"


@dataclass
class TrainingSummary:
    clips_found: int
    clips_used: int
    classes_trained: int
    output_path: Path
    skipped: List[str]


def _collect_clips(clips_dir: Path) -> List[Path]:
    if not clips_dir.exists():
        return []
    return sorted(
        p for p in clips_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def _avg_features(rows: List[Dict[str, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in FEATURE_KEYS:
        vals = [r.get(key, 0.0) for r in rows]
        out[key] = (sum(vals) / len(vals)) if vals else 0.0
    return out


def _weighted_distance(a: Dict[str, float], b: Dict[str, float]) -> float:
    total = 0.0
    for key in FEATURE_KEYS:
        scale = FEATURE_SCALES.get(key, 1.0) or 1.0
        w = FEATURE_WEIGHTS.get(key, 1.0)
        d = (a.get(key, 0.0) - b.get(key, 0.0)) / scale
        total += w * d * d
    return total ** 0.5


def train_from_reference_clips(clips_dir: Path = DEFAULT_CLIPS_DIR) -> TrainingSummary:
    clips = _collect_clips(clips_dir)
    skipped: List[str] = []
    grouped: Dict[str, List[Dict[str, float]]] = {}

    if not clips:
        raise RuntimeError(f"No training clips found in {clips_dir}")

    for clip in clips:
        label = normalize_dunk_label(clip.stem)
        if not label:
            skipped.append(f"{clip.name}: unrecognized dunk label in filename")
            continue
        processor = None
        try:
            # Fresh processor per clip avoids Tasks API timestamp carryover.
            processor = PoseProcessor()
            pose_frames, skeleton_frames, fps, _ball_detections, ball_air_time_s, _lob_type = process_video(str(clip), processor)
            if not pose_frames or not skeleton_frames:
                skipped.append(f"{clip.name}: no usable pose/frames")
                continue
            frame_h = skeleton_frames[0].shape[0]
            body_norm = next(
                (p.body_height_norm for p in pose_frames if p.body_height_norm and p.body_height_norm > 0.1),
                None,
            )
            result = PhysicsEngine(fps=fps, frame_height=frame_h, body_height_norm=body_norm).compute(pose_frames)
            if result is None:
                skipped.append(f"{clip.name}: no physics result")
                continue
            feat = build_feature_dict(result, ball_air_time_s)
            grouped.setdefault(label, []).append(feat)
        except Exception as exc:
            skipped.append(f"{clip.name}: {exc}")
            continue
        finally:
            if processor is not None:
                processor.close()

    if not grouped:
        raise RuntimeError("No clips could be processed into training features")

    prototypes = {label: _avg_features(rows) for label, rows in grouped.items()}
    counts = {label: len(rows) for label, rows in grouped.items()}
    class_radii: Dict[str, float] = {}
    for label, rows in grouped.items():
        proto = prototypes[label]
        dists = sorted(_weighted_distance(r, proto) for r in rows)
        if not dists:
            class_radii[label] = 0.22
            continue
        p90_idx = int(0.9 * (len(dists) - 1))
        p90 = dists[p90_idx]
        mean = sum(dists) / len(dists)
        # Radius floor keeps single-clip classes usable but still bounded.
        class_radii[label] = max(0.18, (0.65 * p90) + (0.35 * mean) + 0.06)

    payload = {
        "version": 1,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_keys": FEATURE_KEYS,
        "feature_scales": FEATURE_SCALES,
        "feature_weights": FEATURE_WEIGHTS,
        "counts": counts,
        "class_radii": class_radii,
        "prototypes": prototypes,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    clips_used = sum(counts.values())
    return TrainingSummary(
        clips_found=len(clips),
        clips_used=clips_used,
        classes_trained=len(counts),
        output_path=MODEL_PATH,
        skipped=skipped,
    )


def main() -> int:
    try:
        summary = train_from_reference_clips()
        print(f"Found clips: {summary.clips_found}")
        print(f"Used clips: {summary.clips_used}")
        print(f"Classes trained: {summary.classes_trained}")
        print(f"Saved: {summary.output_path}")
        if summary.skipped:
            print("Skipped:")
            for s in summary.skipped:
                print(f"  - {s}")
        return 0
    except Exception as exc:
        print(f"Training failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

