"""
Slam Dunk Score Predictor - Pose Processing
MediaPipe Pose for tracking heels, mid_hip, and shoulders.
Supports both legacy (0.10.x) and Tasks API (0.11+).
"""
import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Any
from dataclasses import dataclass

from config import (
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
    MEDIAPIPE_MODEL_COMPLEXITY,
)
from physics_engine import PoseFrame
from ball_tracker import detect_ball, draw_ball, compute_ball_air_time, infer_lob_type


# MediaPipe Pose landmark indices (BlazePose 33-point)
class Landmark:
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30


def _get_landmark(landmarks, idx: int) -> Optional[Tuple[float, float, float]]:
    """Get landmark (x, y, z) or None if invalid."""
    if landmarks is None or idx >= len(landmarks):
        return None
    lm = landmarks[idx]
    vis = getattr(lm, "visibility", 1.0)
    if vis is not None and vis < 0.5:
        return None
    return (lm.x, lm.y, lm.z)


def _get_landmark_xy(landmarks, idx: int) -> Optional[Tuple[float, float]]:
    """Get landmark (x, y) or None."""
    r = _get_landmark(landmarks, idx)
    return (r[0], r[1]) if r else None


def _landmarks_from_result(results: Any) -> Optional[Any]:
    """Extract landmarks from legacy or Tasks API result."""
    if results is None:
        return None
    if hasattr(results, "pose_landmarks"):
        pl = results.pose_landmarks
        if pl and len(pl) > 0:
            return pl[0] if hasattr(pl[0], "__iter__") and not hasattr(pl[0], "x") else pl
        if hasattr(pl, "landmark"):
            return pl.landmark
    return None


def _create_processor_legacy():
    """Create processor using legacy mp.solutions.pose."""
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=MEDIAPIPE_MODEL_COMPLEXITY,
        min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
    )
    return pose, mp_pose, "legacy"


def _create_processor_tasks():
    """Create processor using MediaPipe Tasks API (0.11+)."""
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions

    model_dir = Path(__file__).parent / ".models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "pose_landmarker_lite.task"

    if not model_path.exists():
        try:
            import urllib.request
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
            urllib.request.urlretrieve(url, model_path)
        except Exception:
            alt_url = "https://huggingface.co/AndorML/Public/resolve/main/pose_landmarker_lite.task"
            try:
                import urllib.request
                urllib.request.urlretrieve(alt_url, model_path)
            except Exception as e:
                raise RuntimeError(
                    f"Could not download pose model. Save pose_landmarker_lite.task to {model_dir}"
                ) from e

    from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
    options = PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        min_pose_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
        min_pose_presence_confidence=MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
        running_mode=VisionTaskRunningMode.VIDEO,
    )
    landmarker = PoseLandmarker.create_from_options(options)
    return landmarker, None, "tasks"


def _create_processor():
    """Create processor using legacy or Tasks API."""
    try:
        import mediapipe as mp
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            return _create_processor_legacy()
    except Exception:
        pass
    return _create_processor_tasks()


class PoseProcessor:
    """
    Processes video frames with MediaPipe Pose.
    Works with both legacy (0.10.x) and Tasks API (0.11+).
    """

    def __init__(self):
        self._pose, self._mp_pose, self._mode = _create_processor()
        # Tasks API VIDEO mode requires strictly increasing timestamps (ms)
        self._last_ts_ms: int = -1

    def _process_legacy(self, rgb: np.ndarray) -> Any:
        return self._pose.process(rgb)

    def _process_tasks(self, rgb: np.ndarray, timestamp_ms: int) -> Any:
        from mediapipe.tasks.python.vision.core.image import Image
        from mediapipe.tasks.python.vision.core.image import ImageFormat
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        return self._pose.detect_for_video(mp_image, timestamp_ms)

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp_s: float,
        results: Any = None,
    ) -> Optional[PoseFrame]:
        """Process single frame, return PoseFrame or None if no person detected.
        When using Tasks API (VIDEO mode), pass pre-computed results from detect_for_video
        so timestamps stay monotonically increasing (do not call detection again here)."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self._mode == "legacy":
            results = self._process_legacy(rgb) if results is None else results
            lm_list = results.pose_landmarks.landmark if results.pose_landmarks else None
        else:
            # Use pre-computed results when provided (monotonic timestamp already enforced by caller)
            if results is None:
                timestamp_ms = int(timestamp_s * 1000)
                if timestamp_ms <= self._last_ts_ms:
                    timestamp_ms = self._last_ts_ms + 1
                self._last_ts_ms = timestamp_ms
                results = self._process_tasks(rgb, timestamp_ms)
            lm_list = results.pose_landmarks[0] if results.pose_landmarks else None

        if lm_list is None or len(lm_list) == 0:
            return None

        lm = lm_list
        left_ankle = _get_landmark(lm, Landmark.LEFT_ANKLE)
        right_ankle = _get_landmark(lm, Landmark.RIGHT_ANKLE)
        left_heel = _get_landmark(lm, Landmark.LEFT_HEEL) if len(lm) > Landmark.RIGHT_HEEL else None
        right_heel = _get_landmark(lm, Landmark.RIGHT_HEEL) if len(lm) > Landmark.RIGHT_HEEL else None
        left_hip = _get_landmark(lm, Landmark.LEFT_HIP)
        right_hip = _get_landmark(lm, Landmark.RIGHT_HIP)
        left_shoulder = _get_landmark_xy(lm, Landmark.LEFT_SHOULDER)
        right_shoulder = _get_landmark_xy(lm, Landmark.RIGHT_SHOULDER)
        left_elbow = _get_landmark_xy(lm, Landmark.LEFT_ELBOW) if len(lm) > Landmark.LEFT_ELBOW else None
        right_elbow = _get_landmark_xy(lm, Landmark.RIGHT_ELBOW) if len(lm) > Landmark.RIGHT_ELBOW else None
        left_wrist = _get_landmark_xy(lm, Landmark.LEFT_WRIST) if len(lm) > Landmark.LEFT_WRIST else None
        right_wrist = _get_landmark_xy(lm, Landmark.RIGHT_WRIST) if len(lm) > Landmark.RIGHT_WRIST else None
        nose = _get_landmark(lm, Landmark.NOSE)

        left_foot_y = None
        if left_ankle and left_heel:
            left_foot_y = min(left_ankle[1], left_heel[1])
        elif left_ankle or left_heel:
            left_foot_y = (left_ankle or left_heel)[1]
        right_foot_y = None
        if right_ankle and right_heel:
            right_foot_y = min(right_ankle[1], right_heel[1])
        elif right_ankle or right_heel:
            right_foot_y = (right_ankle or right_heel)[1]

        mid_hip_y = (left_hip[1] + right_hip[1]) / 2.0 if left_hip and right_hip else None
        left_hip_xy = (left_hip[0], left_hip[1]) if left_hip else None
        right_hip_xy = (right_hip[0], right_hip[1]) if right_hip else None
        body_height_norm = None
        if nose and mid_hip_y is not None and (left_foot_y or right_foot_y):
            foot_y = left_foot_y if left_foot_y is not None else right_foot_y
            body_height_norm = max(0.01, foot_y - nose[1])

        return PoseFrame(
            frame_idx=frame_idx,
            left_heel_y=left_foot_y,
            right_heel_y=right_foot_y,
            mid_hip_y=mid_hip_y,
            body_height_norm=body_height_norm,
            left_shoulder=left_shoulder,
            right_shoulder=right_shoulder,
            left_hip=left_hip_xy,
            right_hip=right_hip_xy,
            left_elbow=left_elbow,
            right_elbow=right_elbow,
            left_wrist=left_wrist,
            right_wrist=right_wrist,
            timestamp_s=timestamp_s,
        )

    def _get_pose_landmarks_for_draw(self, results: Any) -> Optional[Any]:
        """Get landmarks in format for drawing."""
        if results is None:
            return None
        if self._mode == "legacy":
            return results.pose_landmarks
        if self._mode == "tasks" and results.pose_landmarks and len(results.pose_landmarks) > 0:
            return self._wrap_tasks_landmarks(results.pose_landmarks[0])
        return None

    def _wrap_tasks_landmarks(self, landmarks: List) -> Any:
        """Wrap Tasks API landmarks for drawing (already have x, y, z)."""
        return type("Landmarks", (), {"landmark": landmarks})()

    def _draw_person_bbox(self, frame: np.ndarray, pose_landmarks: Any) -> np.ndarray:
        """Draw bounding box around tracked person and label 'Player'."""
        if pose_landmarks is None:
            return frame
        h, w = frame.shape[:2]
        lm = getattr(pose_landmarks, "landmark", None) or (pose_landmarks if hasattr(pose_landmarks, "__iter__") else [])
        if not lm:
            return frame
        xs, ys = [], []
        for l in lm:
            x = getattr(l, "x", None)
            y = getattr(l, "y", None)
            if x is not None and y is not None:
                xs.append(int(x * w))
                ys.append(int(y * h))
        if not xs or not ys:
            return frame
        x1 = max(0, min(xs) - 20)
        y1 = max(0, min(ys) - 20)
        x2 = min(w, max(xs) + 20)
        y2 = min(h, max(ys) + 20)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # BGR green
        cv2.putText(frame, "Player", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        return frame

    def draw_skeleton(self, frame: np.ndarray, results: Any) -> np.ndarray:
        """Draw MediaPipe skeleton overlay on frame."""
        pose_landmarks = self._get_pose_landmarks_for_draw(results)
        if pose_landmarks is None:
            return frame
        if self._mode == "legacy":
            import mediapipe as mp
            overlay = frame.copy()
            mp.solutions.drawing_utils.draw_landmarks(
                overlay,
                pose_landmarks,
                self._mp_pose.POSE_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
            )
            self._draw_person_bbox(overlay, pose_landmarks)
            return overlay
        overlay = self._draw_skeleton_tasks(frame, pose_landmarks)
        self._draw_person_bbox(overlay, pose_landmarks)
        return overlay

    def _draw_skeleton_tasks(self, frame: np.ndarray, pose_landmarks: Any) -> np.ndarray:
        """Draw skeleton for Tasks API (simple line drawing)."""
        from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarksConnections
        overlay = frame.copy()
        h, w = overlay.shape[:2]
        lm = pose_landmarks.landmark
        conns = getattr(PoseLandmarksConnections, "POSE_LANDMARKS", None) or getattr(PoseLandmarksConnections, "POSE_CONNECTIONS", [])
        for conn in conns:
            i, j = (conn.start, conn.end) if hasattr(conn, "start") else (conn[0], conn[1])
            if i < len(lm) and j < len(lm):
                pt1 = (int(lm[i].x * w), int(lm[i].y * h))
                pt2 = (int(lm[j].x * w), int(lm[j].y * h))
                cv2.line(overlay, pt1, pt2, (0, 255, 0), 2)
        for l in lm:
            cx, cy = int(l.x * w), int(l.y * h)
            cv2.circle(overlay, (cx, cy), 3, (0, 255, 255), -1)
        return overlay

    def close(self):
        if self._mode == "legacy":
            self._pose.close()
        else:
            self._pose.close()


def process_video(
    video_path: str,
    processor: PoseProcessor,
) -> Tuple[List[PoseFrame], List[np.ndarray], float, List[Tuple[int, Optional[Tuple[int, int, float]], float]], float, str]:
    """Process full video. Returns (pose_frames, skeleton_frames, fps, ball_detections, ball_air_time_s, lob_type)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    pose_frames: List[PoseFrame] = []
    skeleton_frames: List[np.ndarray] = []
    ball_detections: List[Tuple[int, Optional[Tuple[int, int, float]], float]] = []
    frame_idx = 0
    last_timestamp_ms = -1  # MediaPipe VIDEO mode requires strictly increasing timestamps

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        timestamp_s = frame_idx / fps
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if processor._mode == "legacy":
            results = processor._pose.process(rgb)
        else:
            timestamp_ms = int(timestamp_s * 1000)
            # Ensure strictly increasing (required by PoseLandmarker VIDEO running mode)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms
            from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
            results = processor._pose.detect_for_video(mp_image, timestamp_ms)

        # Pass results so we don't call detect_for_video again (avoids non-monotonic timestamp)
        pf = processor.process_frame(frame, frame_idx, timestamp_s, results=results)
        if pf is not None:
            pose_frames.append(pf)

        skeleton_frame = processor.draw_skeleton(frame, results)
        # Ball tracking: detect basketball and draw on overlay
        ball = detect_ball(frame)
        ball_detections.append((frame_idx, ball, timestamp_s))
        if ball is not None:
            draw_ball(skeleton_frame, ball, color=(0, 165, 255))  # BGR orange
        skeleton_rgb = cv2.cvtColor(skeleton_frame, cv2.COLOR_BGR2RGB)
        skeleton_frames.append(skeleton_rgb)
        frame_idx += 1

    cap.release()
    ball_air_time_s = compute_ball_air_time(ball_detections, fps)
    lob_type = infer_lob_type(ball_detections) if ball_detections else "unknown"
    return pose_frames, skeleton_frames, fps, ball_detections, ball_air_time_s, lob_type
