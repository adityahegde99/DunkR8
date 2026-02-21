"""
Slam Dunk Score Predictor - Configuration Constants
"""
# MediaPipe config
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.5
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.5
MEDIAPIPE_MODEL_COMPLEXITY = 1  # 0=lite, 1=full, 2=heavy

# Physics / Scoring thresholds (NBA Slam Dunk Contest inspired)
HANG_TIME_THRESHOLD_S = 0.35  # 0.35s+ gets bonus (typical dunk ~0.5s)
ROTATION_180_EXTRA = 1.5   # Extra points for 180° on top of type
ROTATION_360_EXTRA = 3.0   # Extra points for 360° on top of type
VERTICAL_THRESHOLD_INCHES = 24.0  # Vertical (inches) above this gets height bonus

# Score range (NBA Slam Dunk Contest: 40-50; 50 = perfect, 40 = minimum)
SCORE_MIN = 40.0
SCORE_MAX = 50.0
SCORE_RAW_MIN = 22.0   # Lowest typical raw (One-Hander base)
SCORE_RAW_MAX = 52.0   # Base + all bonuses before cap

# Bonuses on top of dunk-type base (base comes from DunkClassifier)
HANG_TIME_BONUS_MAX = 6.0
ROTATION_EXTRA_BONUS_MAX = 4.0  # Extra for exceptional rotation beyond type
HEIGHT_BONUS_MAX = 6.0
# Lob / alley-oop: ball in air (self off backboard/bounce or assist)
LOB_AIR_TIME_THRESHOLD_S = 0.35  # Ball air time above this counts as lob
LOB_BONUS_MAX = 3.0              # Max bonus for lob (self or alley-oop)

# Pixel-to-inch calibration heuristic (typical 1080p basketball footage)
# Assume ~6ft person = 72 inches, body spans ~400px vertically
PIXELS_PER_INCH_DEFAULT = 400 / 72  # ~5.56 px/inch
