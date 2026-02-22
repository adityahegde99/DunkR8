"""
Slam Dunk Score Predictor - Ball Tracker
Detects basketball (orange/red blob) and estimates ball air time for alley-oops.
"""
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import math


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


def _angle_between_vectors(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    n1 = math.hypot(v1[0], v1[1])
    n2 = math.hypot(v2[0], v2[1])
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cosang = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    cosang = max(-1.0, min(1.0, cosang))
    return float(math.degrees(math.acos(cosang)))


def analyze_ball_trajectory(
    detections: List[Tuple[int, Optional[Tuple[int, int, float]], float]],
) -> Dict[str, Any]:
    """
    Analyze ball path for lob-style trajectory cues.
    Returns a dict with:
    - lob_type: "bounce" | "backboard" | "unknown"
    - bounce_detected / backboard_rebound_detected
    - bounce_angle_deg / rebound_angle_deg
    - y_range_px / x_range_px
    """
    segment = _get_longest_air_segment(detections)
    if len(segment) < 8:
        return {
            "lob_type": "unknown",
            "bounce_detected": False,
            "backboard_rebound_detected": False,
            "bounce_angle_deg": 0.0,
            "rebound_angle_deg": 0.0,
            "y_range_px": 0.0,
            "x_range_px": 0.0,
        }

    xs = [float(ball[0]) for (_idx, ball, _ts) in segment]
    ys = [float(ball[1]) for (_idx, ball, _ts) in segment]
    n = len(xs)

    # Light smoothing to suppress jitter from contour noise.
    xs_s: List[float] = []
    ys_s: List[float] = []
    for i in range(n):
        lo = max(0, i - 1)
        hi = min(n, i + 2)
        xs_s.append(sum(xs[lo:hi]) / float(hi - lo))
        ys_s.append(sum(ys[lo:hi]) / float(hi - lo))

    y_min, y_max = min(ys_s), max(ys_s)
    x_min, x_max = min(xs_s), max(xs_s)
    y_range = y_max - y_min
    x_range = x_max - x_min

    # Detect floor-bounce-like event: local low-in-frame peak (high y) with vertical flip.
    bounce_idx = -1
    bounce_score = 0.0
    min_bounce_prom = max(6.0, 0.12 * y_range)
    for i in range(2, n - 2):
        if not (ys_s[i] >= ys_s[i - 1] and ys_s[i] > ys_s[i + 1]):
            continue
        pre_min = min(ys_s[max(0, i - 3):i])
        post_min = min(ys_s[i + 1:min(n, i + 4)])
        rise_in = ys_s[i] - pre_min
        rise_out = ys_s[i] - post_min
        low_in_frame = ys_s[i] >= (y_min + 0.55 * y_range)
        if low_in_frame and rise_in >= min_bounce_prom and rise_out >= min_bounce_prom:
            score = rise_in + rise_out
            if score > bounce_score:
                bounce_score = score
                bounce_idx = i

    bounce_detected = False
    bounce_angle_deg = 0.0
    if 2 <= bounce_idx <= n - 3:
        prev_i = max(0, bounce_idx - 2)
        next_i = min(n - 1, bounce_idx + 2)
        inc = (xs_s[bounce_idx] - xs_s[prev_i], ys_s[bounce_idx] - ys_s[prev_i])
        out = (xs_s[next_i] - xs_s[bounce_idx], ys_s[next_i] - ys_s[bounce_idx])
        bounce_angle_deg = _angle_between_vectors(inc, out)
        # Incoming should move downward (positive y), outgoing upward (negative y).
        bounce_detected = (inc[1] > 1.5) and (out[1] < -1.5)

    # Detect backboard-like rebound: horizontal direction reversal near top of path.
    rebound_idx = -1
    rebound_score = 0.0
    for i in range(2, n - 2):
        before_dx = xs_s[i] - xs_s[i - 2]
        after_dx = xs_s[i + 2] - xs_s[i]
        if abs(before_dx) < 4.0 or abs(after_dx) < 4.0:
            continue
        if before_dx * after_dx >= 0:
            continue
        near_top = ys_s[i] <= (y_min + 0.45 * y_range)
        if not near_top:
            continue
        score = abs(before_dx) + abs(after_dx)
        if score > rebound_score:
            rebound_score = score
            rebound_idx = i

    backboard_rebound_detected = False
    rebound_angle_deg = 0.0
    if 2 <= rebound_idx <= n - 3:
        prev_i = max(0, rebound_idx - 2)
        next_i = min(n - 1, rebound_idx + 2)
        inc = (xs_s[rebound_idx] - xs_s[prev_i], ys_s[rebound_idx] - ys_s[prev_i])
        out = (xs_s[next_i] - xs_s[rebound_idx], ys_s[next_i] - ys_s[rebound_idx])
        rebound_angle_deg = _angle_between_vectors(inc, out)
        backboard_rebound_detected = abs(inc[0]) > 2.0 and abs(out[0]) > 2.0 and (inc[0] * out[0] < 0)

    # Final lob type decision from trajectory cues.
    lob_type = "unknown"
    if bounce_detected and not backboard_rebound_detected:
        lob_type = "bounce"
    elif backboard_rebound_detected and not bounce_detected:
        lob_type = "backboard"
    elif bounce_detected and backboard_rebound_detected:
        lob_type = "bounce" if bounce_score >= (rebound_score * 1.1) else "backboard"
    else:
        q = max(1, n // 4)
        avg_q1 = sum(ys_s[:q]) / q
        avg_q2 = sum(ys_s[q:2 * q]) / q if (2 * q) <= n else avg_q1
        rise_from_low = avg_q1 - avg_q2
        if rise_from_low > max(5.0, 0.08 * y_range) and y_range > 10.0:
            lob_type = "bounce"
        elif y_range <= 18.0 and abs(avg_q1 - avg_q2) <= 6.0:
            lob_type = "backboard"

    return {
        "lob_type": lob_type,
        "bounce_detected": bool(bounce_detected),
        "backboard_rebound_detected": bool(backboard_rebound_detected),
        "bounce_angle_deg": float(bounce_angle_deg),
        "rebound_angle_deg": float(rebound_angle_deg),
        "y_range_px": float(y_range),
        "x_range_px": float(x_range),
    }


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
    return str(analyze_ball_trajectory(detections).get("lob_type", "unknown"))


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

    # Start from the most coherent tracked segment so crowd false-positives don't dominate.
    segment = _get_longest_air_segment(detections, gap_seconds=gap_seconds)
    if len(segment) < 4:
        return 0.0

    times = [ts for (_idx, _ball, ts) in segment]
    ys = [float(ball[1]) for (_idx, ball, _ts) in segment]
    raw_duration = max(0.0, times[-1] - times[0])

    # If raw track is already plausible, use it.
    if raw_duration <= 2.0:
        return raw_duration

    # Long segments are usually tracking noise. Extract the core arc around apex.
    apex_idx = min(range(len(ys)), key=lambda i: ys[i])  # lowest y in image => highest ball point
    y_min, y_max = min(ys), max(ys)
    y_range = max(1.0, y_max - y_min)
    # Core arc band near apex (keeps only the "true lob" part).
    core_band = y_min + (0.38 * y_range)

    start_idx = apex_idx
    while start_idx > 0 and ys[start_idx] <= core_band:
        start_idx -= 1
    end_idx = apex_idx
    while end_idx < (len(ys) - 1) and ys[end_idx] <= core_band:
        end_idx += 1

    # Expand slightly for realism but keep physically plausible limits.
    start_idx = max(0, start_idx - 1)
    end_idx = min(len(times) - 1, end_idx + 1)
    core_duration = max(0.0, times[end_idx] - times[start_idx])
    return min(core_duration, 2.0)


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
