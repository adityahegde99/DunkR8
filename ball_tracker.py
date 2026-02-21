"""
Slam Dunk Score Predictor - Ball Tracker
Detects basketball (orange/red blob) and estimates ball air time for alley-oops.
"""
from typing import List, Optional, Tuple

import cv2
import numpy as np


def detect_ball(frame_bgr: np.ndarray) -> Optional[Tuple[int, int, float]]:
    """
    Detect basketball in frame (orange/red circular blob).
    Returns (center_x, center_y, radius) or None.
    """
    h, w = frame_bgr.shape[:2]
    # HSV ranges for orange basketball (tune if needed for lighting)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    # Orange: H 5-20, S 100-255, V 80-255
    low = np.array([5, 100, 80])
    high = np.array([25, 255, 255])
    mask1 = cv2.inRange(hsv, low, high)
    # Red wrap-around: H 170-180 and 0-10
    low_red1 = np.array([0, 100, 80])
    high_red1 = np.array([12, 255, 255])
    low_red2 = np.array([168, 100, 80])
    high_red2 = np.array([180, 255, 255])
    mask2 = cv2.bitwise_or(cv2.inRange(hsv, low_red1, high_red1), cv2.inRange(hsv, low_red2, high_red2))
    mask = cv2.bitwise_or(mask1, mask2)
    # Morphology to clean
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter by size (ball ~ 15–80 px radius in typical footage) and circularity
    min_area = (np.pi * 8 * 8)
    max_area = (np.pi * 55 * 55)
    best = None
    best_score = 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(c)
        if radius < 8 or radius > 55:
            continue
        circularity = 4 * np.pi * area / (cv2.arcLength(c, True) ** 2) if cv2.arcLength(c, True) > 0 else 0
        if circularity < 0.5:
            continue
        score = circularity * min(1.0, area / (np.pi * 25 * 25))
        if score > best_score:
            best_score = score
            best = (int(cx), int(cy), float(radius))
    return best


def _get_longest_air_segment(
    detections: List[Tuple[int, Optional[Tuple[int, int, float]], float]],
    gap_seconds: float = 0.15,
) -> List[Tuple[int, Tuple[int, int, float], float]]:
    """Return the longest contiguous segment of frames where ball is detected. Each item: (frame_idx, (cx, cy, r), ts)."""
    if not detections:
        return []
    segments: List[List[Tuple[int, Tuple[int, int, float], float]]] = []
    current: List[Tuple[int, Tuple[int, int, float], float]] = []
    last_ts: float = -1.0
    for idx, ball, ts in detections:
        if ball is not None:
            if current and last_ts >= 0 and (ts - last_ts) > gap_seconds:
                segments.append(current)
                current = []
            current.append((idx, ball, ts))
            last_ts = ts
        else:
            if current:
                segments.append(current)
                current = []
            last_ts = ts
    if current:
        segments.append(current)
    if not segments:
        return []
    return max(segments, key=len)


def infer_lob_type(
    detections: List[Tuple[int, Optional[Tuple[int, int, float]], float]],
) -> str:
    """
    Bounce vs backboard from trajectory. Image coords: y increases downward (lower y = ball higher).
    - Bounce: ball goes UP (after ground), then DOWN, then caught back HIGH. We see rise then fall:
      ball starts low in frame (high y), rises (y decreases to peak), then falls. So first quarter
      has higher avg y than second quarter (ball was lower at start).
    - Backboard: thrown relatively FLAT, never touches ground. Trajectory is flatter; small vertical range.
    """
    segment = _get_longest_air_segment(detections)
    if len(segment) < 8:
        return "unknown"
    ys = [ball[1] for (_, ball, _) in segment]
    n = len(ys)
    y_min, y_max = min(ys), max(ys)
    y_range = y_max - y_min
    q = max(1, n // 4)
    avg_q1 = sum(ys[:q]) / q
    avg_q2 = sum(ys[q : 2 * q]) / q if 2 * q <= n else avg_q1
    # Bounce: ball started low (high y) then rose → first quarter avg y > second quarter
    if avg_q1 > avg_q2 + 0.02 and y_range > 0.025:
        return "bounce"
    # Backboard: flat trajectory, never ground; small vertical range
    if y_range < 0.07 and abs(avg_q1 - avg_q2) < 0.025:
        return "backboard"
    # Fallback: strong rise-from-low (bounce) vs flat (backboard)
    if avg_q1 > avg_q2 + 0.015:
        return "bounce"
    if y_range < 0.05:
        return "backboard"
    return "unknown"


def compute_ball_air_time(
    detections: List[Tuple[int, Optional[Tuple[int, int, float]], float]],
    fps: float,
    gap_seconds: float = 0.15,
) -> float:
    """
    From per-frame (frame_idx, ball_center_or_None, timestamp_s), compute total
    time the ball was detected (air time). Splits by gaps > gap_seconds to get
    distinct "in air" segments and returns the longest segment duration.
    """
    if not detections or fps <= 0:
        return 0.0
    in_air_segments: List[Tuple[float, float]] = []
    segment_start: Optional[float] = None
    for _idx, ball, ts in detections:
        if ball is not None:
            if segment_start is None:
                segment_start = ts
        else:
            if segment_start is not None:
                in_air_segments.append((segment_start, ts))
                segment_start = None
    if segment_start is not None:
        in_air_segments.append((segment_start, detections[-1][2]))
    if not in_air_segments:
        return 0.0
    # Merge segments that are close (same "lob")
    merged: List[Tuple[float, float]] = []
    for s, e in sorted(in_air_segments):
        if merged and s - merged[-1][1] <= gap_seconds:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return max(e - s for s, e in merged) if merged else 0.0


def draw_ball(frame: np.ndarray, center_radius: Optional[Tuple[int, int, float]], color=(255, 165, 0)) -> np.ndarray:
    """Draw ball circle on frame (BGR). color in BGR."""
    if center_radius is None:
        return frame
    cx, cy, r = center_radius
    # Bounding box around ball (what we track)
    pad = int(r) + 8
    x1 = max(0, int(cx) - pad)
    y1 = max(0, int(cy) - pad)
    x2 = min(frame.shape[1], int(cx) + pad)
    y2 = min(frame.shape[0], int(cy) + pad)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, "Ball", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.circle(frame, (int(cx), int(cy)), int(r) + 2, color, 2)
    cv2.circle(frame, (int(cx), int(cy)), 2, (255, 255, 255), -1)
    return frame
