"""
Slam Dunk Score Predictor - ScoringAlgorithm
NBA Slam Dunk Contest style: 1-50 score from dunk type (difficulty) + hang time, rotation, height.
"""
from dataclasses import dataclass
from typing import Optional

from config import (
    HANG_TIME_BONUS_MAX,
    HANG_TIME_THRESHOLD_S,
    ROTATION_180_EXTRA,
    ROTATION_360_EXTRA,
    HEIGHT_BONUS_MAX,
    VERTICAL_THRESHOLD_INCHES,
    SCORE_MIN,
    SCORE_MAX,
    SCORE_RAW_MIN,
    SCORE_RAW_MAX,
    LOB_AIR_TIME_THRESHOLD_S,
    LOB_BONUS_MAX,
)


@dataclass
class ScoreBreakdown:
    """Detailed score components (1-50 scale)."""
    dunk_type: str
    base_score: float
    hang_time_bonus: float
    rotation_bonus: float
    height_bonus: float
    lob_bonus: float
    raw_total: float
    final_score: float
    hang_time_s: float
    max_vertical_inches: float
    rotation_degrees: float
    ball_air_time_s: float


class ScoringAlgorithm:
    """
    Dunk contest scoring 1-50:
    - Base: from DunkClassifier (difficulty: One-Hander 22 → 360 34)
    - Hang Time: +0 to +6 (scaled above 0.35s)
    - Rotation extra: +1.5 for 180°, +3 for 360° (on top of type)
    - Height: +0 to +6 (vertical above 24 in)
    """

    def compute(
        self,
        dunk_base_score: float,
        dunk_type: str,
        hang_time_s: float,
        max_vertical_inches: float,
        rotation_degrees: float,
        ball_air_time_s: float = 0.0,
    ) -> ScoreBreakdown:
        base = dunk_base_score

        # Hang Time Bonus: scaled from threshold (0.35s -> 0, 0.7s+ -> max)
        if hang_time_s >= HANG_TIME_THRESHOLD_S:
            excess = hang_time_s - HANG_TIME_THRESHOLD_S
            hang_bonus = min(HANG_TIME_BONUS_MAX, excess / 0.35 * HANG_TIME_BONUS_MAX)
        else:
            hang_bonus = 0.0

        # Rotation extra (style on top of type)
        if rotation_degrees >= 330:
            rot_bonus = ROTATION_360_EXTRA
        elif rotation_degrees >= 150:
            rot_bonus = ROTATION_180_EXTRA
        else:
            rot_bonus = 0.0

        # Height Bonus: vertical above threshold, scaled
        if max_vertical_inches >= VERTICAL_THRESHOLD_INCHES:
            excess_inches = max_vertical_inches - VERTICAL_THRESHOLD_INCHES
            height_bonus = min(HEIGHT_BONUS_MAX, excess_inches / 12.0 * HEIGHT_BONUS_MAX)
        else:
            height_bonus = 0.0

        # Lob / Alley-oop: ball in air (self off backboard/bounce or assist from teammate)
        if ball_air_time_s >= LOB_AIR_TIME_THRESHOLD_S:
            excess = ball_air_time_s - LOB_AIR_TIME_THRESHOLD_S
            lob_bonus = min(LOB_BONUS_MAX, LOB_BONUS_MAX * (0.5 + excess / 0.5))
        else:
            lob_bonus = 0.0

        raw_total = base + hang_bonus + rot_bonus + height_bonus + lob_bonus
        # Map raw (≈22–52) to contest range 40–50 (research: contest uses 40–50, 50 rare)
        span = max(0.01, SCORE_RAW_MAX - SCORE_RAW_MIN)
        contest_score = SCORE_MIN + (raw_total - SCORE_RAW_MIN) / span * (SCORE_MAX - SCORE_MIN)
        final_score = max(SCORE_MIN, min(SCORE_MAX, round(contest_score, 1)))

        return ScoreBreakdown(
            dunk_type=dunk_type,
            base_score=base,
            hang_time_bonus=hang_bonus,
            rotation_bonus=rot_bonus,
            height_bonus=height_bonus,
            lob_bonus=lob_bonus,
            raw_total=raw_total,
            final_score=final_score,
            hang_time_s=hang_time_s,
            max_vertical_inches=max_vertical_inches,
            rotation_degrees=rotation_degrees,
            ball_air_time_s=ball_air_time_s,
        )
