"""
Train the dunk classifier from labeled videos.

Put your clips in a folder per dunk type:

  training_dunks/
    One-Handed/
      clip1.mp4
      clip2.mp4
    Windmill/
      w1.mp4
    Eastbay (Between-the-Legs)/
      e1.mp4
    ...

Then run:

  python train_dunk_classifier.py

The script will:
  1. Process each video (pose + ball + physics)
  2. Extract the same feature vector the rule-based classifier uses
  3. Train a RandomForest classifier
  4. Save models/dunk_classifier.joblib and models/dunk_classifier_meta.json

The app will automatically use the trained model when present (and fall back to
rules when the model is unsure or missing).
"""
from pathlib import Path
import json
import sys

# Project root
ROOT = Path(__file__).resolve().parent
TRAINING_DIR = ROOT / "training_dunks"
MODEL_DIR = ROOT / "models"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def discover_labeled_clips() -> list[tuple[Path, str]]:
    """Find all (video_path, label) under training_dunks/<label>/."""
    out = []
    if not TRAINING_DIR.exists():
        return out
    for label_dir in TRAINING_DIR.iterdir():
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for f in label_dir.iterdir():
            if f.suffix.lower() in VIDEO_EXTENSIONS:
                out.append((f, label))
    return out


def main() -> int:
    from pose_processor import PoseProcessor, process_video
    from physics_engine import PhysicsEngine
    from dunk_classifier import (
        physics_result_to_feature_vector,
        DUNK_TYPES,
        MODEL_PATH,
        MODEL_META_PATH,
    )
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    import numpy as np

    clips = discover_labeled_clips()
    if not clips:
        print("No labeled clips found.")
        print(f"Put videos in: {TRAINING_DIR}/<dunk_type>/video.mp4")
        print("Example: training_dunks/One-Handed/my_dunk.mp4")
        return 1

    print(f"Found {len(clips)} labeled clip(s).")
    processor = PoseProcessor()
    X_list = []
    y_list = []

    for i, (video_path, label) in enumerate(clips):
        print(f"  [{i+1}/{len(clips)}] {video_path.parent.name}/{video_path.name} ... ", end="", flush=True)
        try:
            pose_frames, _, fps, _, ball_air_time_s, lob_type = process_video(str(video_path), processor)
            if not pose_frames:
                print("no pose frames")
                continue
            engine = PhysicsEngine(fps=fps, frame_height=720, body_height_norm=None)
            result = engine.compute(pose_frames)
            if result is None:
                print("no physics result")
                continue
            vec = physics_result_to_feature_vector(result, ball_air_time_s, lob_type)
            X_list.append(vec)
            y_list.append(label)
            print("ok")
        except Exception as e:
            print(f"error: {e}")

    if len(X_list) < 2:
        print("Need at least 2 successfully processed clips to train.")
        return 1

    X = np.asarray(X_list, dtype=np.float64)
    y = np.asarray(y_list)

    print(f"Training on {len(X)} samples.")
    clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, min_samples_leaf=2)
    clf.fit(X, y)
    clf_classes = list(clf.classes_)
    print(f"Classes: {clf_classes}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "classes": clf_classes}, MODEL_PATH)
    with open(MODEL_META_PATH, "w") as f:
        json.dump({"classes": clf_classes}, f, indent=2)
    print(f"Saved model to {MODEL_PATH}")
    print("Restart the app to use the trained classifier.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
