# Image Sensor Architecture: CIS vs DVS

**Group project** — completed as part of a graduate imaging course. This folder contains the full team's work.

## Team contributions

| Folder | Contributor | Work |
|---|---|---|
| `Ishs-work/` | Ish Malik | DVS power model, pixel breakdown analysis, temporal variation simulation |
| `Harshithas-work/` | Harshitha | CIS (CMOS Image Sensor) behavioral model |
| `Remaas-work/` | Ramaa | Scene model and visual computing simulations |
| `Sergeys-work/` | Sergey | CIS vs DVS comparison analysis |
| `Documents/` | Team | Project proposal, reference papers, progress reports |

## Project overview

A task-driven hardware comparison of two fundamentally different image sensor architectures:

- **CIS (CMOS Image Sensor)** — conventional frame-based sensor, reads out all pixels at a fixed frame rate regardless of scene activity
- **DVS (Dynamic Vision Sensor)** — event-based sensor, fires asynchronous events only when a pixel detects a brightness change above a contrast threshold

The goal: model and compare power consumption, tracking feasibility, and efficiency across a range of scene conditions (object velocity, object size, background texture, contrast threshold).

## My work (`Ishs-work/`)

**`dvs_model.py`** — physics-based DVS power model parameterized from the Lichtsteiner 2008 128×128 sensor (3.3V):
- Static power: logic (0.3 mA) + bias generators (5.5 mA) = 19.14 mW
- Dynamic power: scales with event rate via energy-per-event (4.95 nJ)
- Sweeps velocity, object size, background texture, and contrast threshold θ
- Computes pixel activity breakdown (active vs silent pixels)
- Models temporal power variation as scene activity changes over time

**`dvs_model_explained.py`** — annotated version of the model with detailed inline explanations of the physics and design choices

**`dvs_results/`** — generated plots and CSVs:
- Power vs velocity, threshold, background
- Static vs dynamic power breakdown
- Active vs silent pixel breakdown (donut charts)
- Temporal variation (DVS power tracks scene activity; CIS locked to worst-case FPS)

> **Note:** `dvs_model.py` is still in progress — updates pending.

## Key finding

DVS static power dominates across the entire velocity range modeled. Dynamic power (event-driven) is small relative to the static floor, meaning DVS efficiency gains over CIS are most significant at high frame rates or in low-activity scenes where CIS would still run at full power.

## Reference papers (`Documents/`)

- Lichtsteiner et al. 2008 — original DVS sensor paper (128×128, 120 dB, 15 µs latency)
- ModuCIS manuscript — CIS modeling methodology
- Task-Driven Hardware Optimization of CIS and DVS Image Sensors

## Dependencies

**DVS model (Python):**
```bash
pip install numpy pandas matplotlib
python Ishs-work/dvs_model.py
```

**CIS model (Python):**
See `Harshithas-work/` — requires the same dependencies plus the scene model from `Remaas-work/`.
