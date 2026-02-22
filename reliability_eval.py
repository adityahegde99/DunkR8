"""
Evaluate detection/classification reliability on reference clips.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from pose_processor import PoseProcessor, process_video
from physics_engine import PhysicsEngine
from dunk_analyzer import DunkAnalyzer
from ontology_model import normalize_dunk_label


VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
CLIPS_DIR = Path(__file__).resolve().parent / "reference_dunks" / "clips"


def iter_clips() -> List[Path]:
    if not CLIPS_DIR.exists():
        return []
    return sorted(p for p in CLIPS_DIR.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXT)


def main() -> int:
    clips = iter_clips()
    if not clips:
        print(f"No clips found in {CLIPS_DIR}")
        return 1

    analyzer = DunkAnalyzer()
    total = 0
    dunk_detected = 0
    label_known = 0
    exact_match = 0
    print("Reliability report")
    print("==================")
    for clip in clips:
        expected = normalize_dunk_label(clip.stem)
        total += 1
        processor = None
        try:
            # Fresh processor per clip avoids Tasks API timestamp carryover.
            processor = PoseProcessor()
            pose_frames, skeleton_frames, fps, ball_detections, ball_air_time_s, lob_type = process_video(str(clip), processor)
            if not pose_frames or not skeleton_frames:
                print(f"- {clip.name}: FAILED (no pose/frames)")
                continue
            h, w = skeleton_frames[0].shape[:2]
            body = next((p.body_height_norm for p in pose_frames if p.body_height_norm), None)
            physics = PhysicsEngine(fps=fps, frame_height=h, body_height_norm=body).compute(pose_frames)
            if physics is None:
                print(f"- {clip.name}: FAILED (no physics)")
                continue
            out = analyzer.analyze(
                physics=physics,
                pose_frames=pose_frames,
                ball_detections=ball_detections,
                ball_air_time_s=ball_air_time_s,
                lob_type=lob_type,
                frame_width=w,
                frame_height=h,
                clip_name=clip.stem,
            )
            if out.is_dunk:
                dunk_detected += 1
            status = "DUNK" if out.is_dunk else f"REJECT({out.non_dunk_type})"
            msg = f"- {clip.name}: {status} -> {out.dunk_type if out.is_dunk else out.non_dunk_type}"
            if expected:
                label_known += 1
                match = out.is_dunk and out.dunk_type == expected
                if match:
                    exact_match += 1
                msg += f" | expected={expected} | match={'yes' if match else 'no'}"
            msg += f" | model={out.model_prediction or 'rule-only'}:{out.model_confidence:.2f}"
            print(msg)
        except Exception as exc:
            print(f"- {clip.name}: ERROR ({exc})")
        finally:
            if processor is not None:
                processor.close()

    print("\nSummary")
    print("-------")
    print(f"clips: {total}")
    print(f"dunk detection rate: {dunk_detected}/{total} = {(100.0*dunk_detected/max(1,total)):.1f}%")
    if label_known:
        print(f"exact label accuracy: {exact_match}/{label_known} = {(100.0*exact_match/label_known):.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

