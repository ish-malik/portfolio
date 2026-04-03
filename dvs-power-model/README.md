# DVS Power Model *(in progress)*

Physics-based power model for Dynamic Vision Sensors (DVS). Part of a broader CIS vs DVS sensor comparison study.

**Status:** Work in progress — will be updated when complete.

## Overview

Models the power consumption of a DVS (event-based vision sensor) as a function of:
- Scene velocity (px/s)
- Object size and background texture
- Contrast threshold θ
- Static vs dynamic power breakdown

Parameterized from the Lichtsteiner 2008 128×128 DVS sensor (3.3V supply). Compares against a CIS (conventional image sensor) model to quantify when DVS is more power-efficient.

`Python` · `pandas` · `matplotlib`
