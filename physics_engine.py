"""
Slam Dunk Score Predictor - PhysicsEngine
Calculates hang time, max vertical, and rotation from pose keypoints.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple
import math


@dataclass
class PoseFrame:
    """Single frame pose data."""
    frame_idx: int
    left_heel_y: Optional[float]
    right_heel_y: Optional[float]
    mid_hip_y: Optional[float]
    left_shoulder: Optional[Tuple[float, float]]
    right_shoulder: Optional[Tuple[float, float]]
    timestamp_s: float
    body_height_norm: Optional[float] = None
    mid_hip_x: Optional[float] = None
    nose: Optional[Tuple[float, float]] = None
    left_hip: Optional[Tuple[float, float]] = None
    right_hip: Optional[Tuple[float, float]] = None
    left_knee: Optional[Tuple[float, float]] = None
    right_knee: Optional[Tuple[float, float]] = None
    left_ankle: Optional[Tuple[float, float]] = None
    right_ankle: Optional[Tuple[float, float]] = None
    left_elbow: Optional[Tuple[float, float]] = None
    right_elbow: Optional[Tuple[float, float]] = None
    left_wrist: Optional[Tuple[float, float]] = None
    right_wrist: Optional[Tuple[float, float]] = None


@dataclass
class PhysicsResult:
    """Output of physics analysis."""
    hang_time_s: float
    max_vertical_inches: float
    rotation_degrees: float
    frames_airborne: int
    start_hip_y: float
    min_hip_y: float
    airborne_start_frame_idx: int = -1
    airborne_end_frame_idx: int = -1
    airborne_start_timestamp_s: float = 0.0
    airborne_end_timestamp_s: float = 0.0
    apex_frame_idx: int = -1
    apex_timestamp_s: float = 0.0
    apex_height_ft: float = 0.0
    estimated_vertical_leap_inches: float = 0.0
    ground_threshold_y: float = 0.0
    # Arm movement (for windmill / Eastbay / two-hand)
    left_wrist_angle_sweep_deg: float = 0.0
    right_wrist_angle_sweep_deg: float = 0.0
    wrist_went_below_hip: bool = False
    wrist_below_hip_near_midline: bool = False  # When wrist was below hip, was it between the legs (x between hips)?
    two_hands_cue: bool = False                 # Both wrists close at some point (two-handed finish)
    max_wrist_radius: float = 0.0


class PhysicsEngine:
    """
    Computes dunk physics from pose keypoint sequences.
    FPS-aware: all time calculations use frame timestamps.
    """

    def __init__(
        self,
        fps: float,
        frame_height: int,
        pixels_per_inch: Optional[float] = None,
        body_height_norm: Optional[float] = None,
    ):
        self.fps = fps
        self.frame_height = frame_height
        self.frame_duration_s = 1.0 / fps if fps > 0 else 1.0 / 30.0
        # Calibrate: assume 72" person; body_height_norm = normalized body span in frame
        if body_height_norm and body_height_norm > 0.1:
            self.pixels_per_inch = (body_height_norm * frame_height) / 72.0
        else:
            self.pixels_per_inch = pixels_per_inch or (frame_height / 72.0)

    def _y_to_inches(self, y_pixels: float) -> float:
        """Convert pixel y (image coords, origin top-left) to inches. Lower y = higher."""
        return abs(y_pixels) / self.pixels_per_inch

    def _pixel_delta_to_inches(self, delta_pixels: float) -> float:
        """Convert vertical pixel delta to inches (positive = jumped up)."""
        return delta_pixels / self.pixels_per_inch

    def _shoulder_angle_degrees(
        self,
        left: Tuple[float, float],
        right: Tuple[float, float],
    ) -> float:
        """Angle of shoulder line from horizontal, in degrees [-180, 180]."""
        dx = right[0] - left[0]
        dy = right[1] - left[1]
        return math.degrees(math.atan2(dy, dx))

    def compute(
        self,
        pose_frames: List[PoseFrame],
    ) -> Optional[PhysicsResult]:
        """
        Compute hang time, max vertical, and rotation from pose sequence.
        Returns None if insufficient valid data.
        """
        if not pose_frames:
            return None

        valid = [
            p
            for p in pose_frames
            if p.left_heel_y is not None
            and p.right_heel_y is not None
            and p.mid_hip_y is not None
        ]
        if len(valid) < 5:
            return None

        # Ground level: percentile of foot y (when feet are lowest = on ground)
        all_foot_ys = []
        for p in valid:
            if p.left_heel_y is not None and -0.2 <= p.left_heel_y <= 1.5:
                all_foot_ys.append(p.left_heel_y)
            if p.right_heel_y is not None and -0.2 <= p.right_heel_y <= 1.5:
                all_foot_ys.append(p.right_heel_y)
        all_foot_ys.sort()
        if all_foot_ys:
            idx_ground = int((len(all_foot_ys) - 1) * 0.85)
            baseline_y = all_foot_ys[idx_ground]
        else:
            baseline_y = 0.9
        # Foot-based airborne threshold (lower y = higher in frame).
        # Slightly stricter margin avoids counting running strides as airborne.
        threshold_y = baseline_y - 0.055
        # Hip-based fallback keeps airborne state robust when one foot landmark flickers.
        # Use a robust "ground-ish" hip baseline (high percentile), not raw max.
        hip_vals = [p.mid_hip_y for p in valid if p.mid_hip_y is not None and -0.2 <= p.mid_hip_y <= 1.5]
        if hip_vals:
            hip_vals_sorted = sorted(hip_vals)
            idx_hip_ground = int((len(hip_vals_sorted) - 1) * 0.85)
            hip_baseline_y = hip_vals_sorted[idx_hip_ground]
        else:
            hip_baseline_y = 0.7
        hip_air_threshold = hip_baseline_y - 0.04

        # Find airborne intervals using both foot and hip cues.
        airborne_start: Optional[int] = None
        airborne_end: Optional[int] = None
        max_airborne_duration = 0.0
        best_start, best_end = 0, 0

        i = 0
        while i < len(valid):
            p = valid[i]
            left_above = p.left_heel_y < threshold_y
            right_above = p.right_heel_y < threshold_y
            both_above = left_above and right_above
            avg_foot_y = (p.left_heel_y + p.right_heel_y) * 0.5
            # Allow hip fallback only when average foot height also indicates lift.
            avg_feet_elevated = avg_foot_y < (threshold_y + 0.015)
            hip_high = p.mid_hip_y is not None and p.mid_hip_y < hip_air_threshold
            # Require foot corroboration for hip fallback to avoid long false-airborne runs.
            airborne_like = both_above or (hip_high and avg_feet_elevated)
            if airborne_like and airborne_start is None:
                airborne_start = i
            elif not airborne_like and airborne_start is not None:
                airborne_end = i - 1
                start_ts = valid[airborne_start].timestamp_s
                end_ts = valid[airborne_end].timestamp_s
                duration = end_ts - start_ts
                if duration > max_airborne_duration:
                    max_airborne_duration = duration
                    best_start, best_end = airborne_start, airborne_end
                airborne_start = None
            i += 1

        if airborne_start is not None:
            airborne_end = len(valid) - 1
            start_ts = valid[airborne_start].timestamp_s
            end_ts = valid[airborne_end].timestamp_s
            duration = end_ts - start_ts
            if duration > max_airborne_duration:
                max_airborne_duration = duration
                best_start, best_end = airborne_start, airborne_end

        # If interval is implausibly long, re-anchor to the core hip-elevation window.
        if max_airborne_duration > 2.5:
            hip_ys_valid = [p.mid_hip_y for p in valid if p.mid_hip_y is not None]
            if hip_ys_valid:
                hip_min = min(hip_ys_valid)
                core_idxs = [idx for idx, p in enumerate(valid) if p.mid_hip_y is not None and p.mid_hip_y <= (hip_min + 0.07)]
                if core_idxs:
                    core_start = core_idxs[0]
                    core_end = core_idxs[-1]
                    core_dur = valid[core_end].timestamp_s - valid[core_start].timestamp_s
                    if 0.22 <= core_dur <= 2.5:
                        best_start, best_end = core_start, core_end
                        max_airborne_duration = core_dur

        # Average dunk hang ~0.53s; most in-game 0.35–0.55s. Cap raw to avoid overestimation.
        hang_time_s = min(max_airborne_duration, 0.60)
        airborne_frames_for_arms = valid[best_start : best_end + 1] if max_airborne_duration > 0 else []
        airborne_start_frame_idx = (
            airborne_frames_for_arms[0].frame_idx if airborne_frames_for_arms else -1
        )
        airborne_end_frame_idx = (
            airborne_frames_for_arms[-1].frame_idx if airborne_frames_for_arms else -1
        )
        airborne_start_timestamp_s = (
            airborne_frames_for_arms[0].timestamp_s if airborne_frames_for_arms else 0.0
        )
        airborne_end_timestamp_s = (
            airborne_frames_for_arms[-1].timestamp_s if airborne_frames_for_arms else 0.0
        )

        # Rim-touch frame: peak height in airborne segment (hands touch rim there); rotation counts only jump -> rim touch
        rim_touch_end_idx = len(airborne_frames_for_arms) - 1
        if len(airborne_frames_for_arms) >= 3:
            hip_ys_air = [p.mid_hip_y for p in airborne_frames_for_arms if p.mid_hip_y is not None]
            if hip_ys_air:
                min_hip_in_air = min(hip_ys_air)
                for idx, p in enumerate(airborne_frames_for_arms):
                    if p.mid_hip_y is not None and p.mid_hip_y <= min_hip_in_air + 0.01:
                        rim_touch_end_idx = idx
                        break
        rotation_frames = airborne_frames_for_arms[: rim_touch_end_idx + 1]

        # Max vertical: physics formula h = g*t^2/8 (flight time to jump height). Prefer underball.
        # Hang time (s) -> vertical (m): h_m = g * t^2 / 8. Then inches = h_m / 0.0254.
        flight_vertical_inches = (9.81 * (hang_time_s ** 2) / 8.0) / 0.0254 if hang_time_s > 0 else 0.0
        ground_frames = valid[: max(1, len(valid) // 5)]
        start_hip_y = max(p.mid_hip_y for p in ground_frames if p.mid_hip_y is not None)
        hip_ys = [p.mid_hip_y for p in valid if p.mid_hip_y is not None]
        min_hip_y = min(hip_ys)
        delta_normalized = start_hip_y - min_hip_y
        delta_pixels = delta_normalized * self.frame_height
        pixel_vertical_inches = max(0.0, self._pixel_delta_to_inches(delta_pixels))
        # Use lower of pixel estimate or physics-based; apply 0.92 factor to underball (user preference).
        if flight_vertical_inches > 0.0:
            conservative_flight = flight_vertical_inches * 0.92
            pixel_vertical_inches = min(pixel_vertical_inches, conservative_flight)
        # Conservative cap 36" to avoid overestimation (elite ~40" but we underball).
        max_vertical_inches = max(0.0, min(36.0, pixel_vertical_inches))
        apex_height_ft = (72.0 + max_vertical_inches) / 12.0
        estimated_vertical_leap_inches = max_vertical_inches

        apex_frame_idx = -1
        apex_timestamp_s = 0.0
        if airborne_frames_for_arms:
            apex_frame = min(
                (p for p in airborne_frames_for_arms if p.mid_hip_y is not None),
                key=lambda p: p.mid_hip_y,
                default=None,
            )
            if apex_frame is not None:
                apex_frame_idx = apex_frame.frame_idx
                apex_timestamp_s = apex_frame.timestamp_s

        # Rotation: from jump to rim touch only (not after hands touch rim)
        shoulder_frames = [
            p for p in rotation_frames
            if p.left_shoulder is not None and p.right_shoulder is not None
        ]
        hip_frames = [
            p for p in rotation_frames
            if getattr(p, "left_hip", None) is not None and getattr(p, "right_hip", None) is not None
        ]
        rotation_degrees = 0.0
        if len(shoulder_frames) >= 2:
            angles = [
                self._shoulder_angle_degrees(p.left_shoulder, p.right_shoulder)
                for p in shoulder_frames
            ]
            total_delta = 0.0
            for j in range(1, len(angles)):
                delta = angles[j] - angles[j - 1]
                if delta > 180:
                    delta -= 360
                elif delta < -180:
                    delta += 360
                total_delta += delta
            rotation_degrees = abs(total_delta)
        if len(hip_frames) >= 2:
            hip_angles = [
                self._shoulder_angle_degrees(p.left_hip, p.right_hip)
                for p in hip_frames
            ]
            total_delta = 0.0
            for j in range(1, len(hip_angles)):
                delta = hip_angles[j] - hip_angles[j - 1]
                if delta > 180:
                    delta -= 360
                elif delta < -180:
                    delta += 360
                total_delta += delta
            rotation_degrees = max(rotation_degrees, abs(total_delta))

        frames_airborne = int(hang_time_s * self.fps) if hang_time_s > 0 else 0

        # Arm movement over the main airborne segment
        left_sweep = 0.0
        right_sweep = 0.0
        wrist_below_hip = False
        wrist_below_hip_near_midline = False
        two_hands_cue = False
        max_radius = 0.0
        min_wrist_dist = 1.0
        if len(airborne_frames_for_arms) < 3 and len(valid) >= 5:
            hip_ys = [p.mid_hip_y for p in valid if p.mid_hip_y is not None]
            if hip_ys:
                min_hip = min(hip_ys)
                airborne_frames_for_arms = [p for p in valid if p.mid_hip_y is not None and p.mid_hip_y <= min_hip + 0.05]
        for p in airborne_frames_for_arms:
            if p.mid_hip_y is None:
                continue
            hip_left_x = p.left_hip[0] if p.left_hip else None
            hip_right_x = p.right_hip[0] if p.right_hip else None
            if p.left_wrist and p.left_shoulder:
                dx = p.left_wrist[0] - p.left_shoulder[0]
                dy = p.left_wrist[1] - p.left_shoulder[1]
                max_radius = max(max_radius, math.sqrt(dx * dx + dy * dy))
                if p.left_wrist[1] > p.mid_hip_y:
                    wrist_below_hip = True
                    if hip_left_x is not None and hip_right_x is not None:
                        mid_x_min, mid_x_max = min(hip_left_x, hip_right_x), max(hip_left_x, hip_right_x)
                        if mid_x_min <= p.left_wrist[0] <= mid_x_max:
                            wrist_below_hip_near_midline = True
            if p.right_wrist and p.right_shoulder:
                dx = p.right_wrist[0] - p.right_shoulder[0]
                dy = p.right_wrist[1] - p.right_shoulder[1]
                max_radius = max(max_radius, math.sqrt(dx * dx + dy * dy))
                if p.right_wrist[1] > p.mid_hip_y:
                    wrist_below_hip = True
                    if hip_left_x is not None and hip_right_x is not None:
                        mid_x_min, mid_x_max = min(hip_left_x, hip_right_x), max(hip_left_x, hip_right_x)
                        if mid_x_min <= p.right_wrist[0] <= mid_x_max:
                            wrist_below_hip_near_midline = True
            if p.left_wrist and p.right_wrist:
                d = math.sqrt((p.left_wrist[0] - p.right_wrist[0]) ** 2 + (p.left_wrist[1] - p.right_wrist[1]) ** 2)
                min_wrist_dist = min(min_wrist_dist, d)
        two_hands_cue = min_wrist_dist < 0.12
        if len(airborne_frames_for_arms) >= 2:
            for side, shoulder_attr, wrist_attr in [
                ("left", "left_shoulder", "left_wrist"),
                ("right", "right_shoulder", "right_wrist"),
            ]:
                pts = []
                for p in airborne_frames_for_arms:
                    sh = getattr(p, shoulder_attr, None)
                    wr = getattr(p, wrist_attr, None)
                    if sh and wr:
                        dx = wr[0] - sh[0]
                        dy = wr[1] - sh[1]
                        ang = math.degrees(math.atan2(dy, dx))
                        pts.append(ang)
                if len(pts) >= 2:
                    total = 0.0
                    for j in range(1, len(pts)):
                        d = pts[j] - pts[j - 1]
                        if d > 180:
                            d -= 360
                        elif d < -180:
                            d += 360
                        total += d
                    sweep = abs(total)
                    if side == "left":
                        left_sweep = sweep
                    else:
                        right_sweep = sweep

        return PhysicsResult(
            hang_time_s=hang_time_s,
            max_vertical_inches=max_vertical_inches,
            rotation_degrees=rotation_degrees,
            frames_airborne=frames_airborne,
            start_hip_y=start_hip_y,
            min_hip_y=min_hip_y,
            airborne_start_frame_idx=airborne_start_frame_idx,
            airborne_end_frame_idx=airborne_end_frame_idx,
            airborne_start_timestamp_s=airborne_start_timestamp_s,
            airborne_end_timestamp_s=airborne_end_timestamp_s,
            apex_frame_idx=apex_frame_idx,
            apex_timestamp_s=apex_timestamp_s,
            apex_height_ft=apex_height_ft,
            estimated_vertical_leap_inches=estimated_vertical_leap_inches,
            ground_threshold_y=threshold_y,
            left_wrist_angle_sweep_deg=left_sweep,
            right_wrist_angle_sweep_deg=right_sweep,
            wrist_went_below_hip=wrist_below_hip,
            wrist_below_hip_near_midline=wrist_below_hip_near_midline,
            two_hands_cue=two_hands_cue,
            max_wrist_radius=max_radius,
        )
