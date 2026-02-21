# Slam Dunk Score Predictor

**Computer Vision Analysis of Dunk Performance — NBA Slam Dunk Contest Style**

Uses MediaPipe Pose, OpenCV, and a custom PhysicsEngine to analyze dunk videos and predict a judges' score (40–50 range).

## Tech Stack

- **CV:** OpenCV, MediaPipe Pose
- **Frontend:** Streamlit

## Project Structure

```
ImpactInsight/
├── app.py            # Streamlit dashboard (main UI)
├── config.py         # Scoring constants and thresholds
├── pose_processor.py # MediaPipe pose detection & skeleton overlay
├── physics_engine.py # Hang time, max vertical, rotation
├── scoring.py       # NBA Slam Dunk Contest scoring algorithm
├── requirements.txt
└── README.md
```

## Installation

```bash
cd ImpactInsight
pip install -r requirements.txt
```

**Note:** Works with both MediaPipe 0.10.x (legacy) and 0.11+ (Tasks API). On first run with 0.11+, the pose model (~6MB) downloads automatically.

## Usage

```bash
streamlit run app.py
```

1. Upload an MP4 dunk video
2. View the **Judges' Scorecard** in the sidebar (metrics + score breakdown)
3. Scrub through the video with the **MediaPipe skeleton overlay**
4. See the **Final Predicted Score** (40.0 – 50.0)

## Scoring Algorithm

| Component      | Rule                                              |
|----------------|---------------------------------------------------|
| Base Score     | 30.0                                              |
| Hang Time      | +10 max if > 0.8s (scaled linearly)               |
| Rotation       | +5 for 180°, +10 for 360°                         |
| Height         | +5 max if vertical > 35 inches (scaled)           |
| **Final Range**| 40.0 – 50.0                                       |

## Physics Metrics

- **Hang Time:** Time (seconds) both heels are above baseline until landing
- **Max Vertical:** Peak hip height delta from start (inches)
- **Rotation:** Cumulative shoulder angle change (degrees)

## Error Handling

- No person detected → Clear error message
- Insufficient pose data → Warning with fallback metrics
- FPS-aware calculations for accurate timing

## License

MIT
