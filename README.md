# DunkR8

Pose + ball tracking system for dunk detection, taxonomy classification, rejection, and 40-50 contest scoring.

## What It Does

- Detects whether a clip contains a valid dunk (strict multi-condition validation)
- Rejects non-dunks (layup, finger roll, floater, tip-in, missed/blocked dunk, etc.)
- Classifies canonical dunk types via a formal ontology (power, rotation, windmill, leg-thread, lob, distance, reaction)
- Distinguishes lob mode (`alley-oop` vs `self-lob` when trajectory indicates backboard/bounce)
- Extracts biomechanics (hang time, airborne frames, apex, vertical, rotation, takeoff cues, arm and ball path behavior)
- Produces contest scoring on a 40-50 scale with difficulty/style tiers
- Supports optional prototype training from your own labeled clips

## Tech Stack

- **CV:** OpenCV, MediaPipe Pose
- **Frontend:** Streamlit
- **Pipeline:** Rule-based ontology + biomechanical heuristics (no reference clip matcher, no trained classifier)

## Project Structure

```
DunkR8/
├── app.py             # Streamlit app (UI + output formatting)
├── dunk_analyzer.py   # Dunk validation, taxonomy classification, rejection, scoring
├── dunk_ontology.py   # Canonical dunk taxonomy and category metadata
├── pose_processor.py  # Pose extraction and skeleton overlay
├── ball_tracker.py    # Ball detection and trajectory helpers
├── physics_engine.py  # Core kinematics from pose sequence
├── config.py
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

MediaPipe model note: on first run with Tasks API, the pose model downloads automatically to `.models/`.

## Run

```bash
streamlit run app.py
```

## Optional Training (Your Clips)

Put labeled clips in `reference_dunks/clips/` and include the dunk type in the filename:

- `windmill.mp4`
- `360_dunk.mp4`
- `reverse_eastbay.mov`

Then train:

```bash
python ontology_trainer.py
```

This writes `models/ontology_prototypes.json`, which is automatically loaded by the analyzer.

To benchmark current reliability on all reference clips:

```bash
python reliability_eval.py
```

## Output Fields

For each upload, the app reports:

- Primary classification category
- Canonical dunk type (or non-dunk type)
- Alley-oop and self-lob flags
- Rotation (degrees + band)
- Hang time, max vertical, apex height, airborne frames
- Ball air time and ball-path arc
- Difficulty tier, style grade, comparable tier
- Final contest score (40-50)

## License

MIT
