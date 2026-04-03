import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# IMPORT RAMAA'S SCENE MODEL
# =============================================================================
# We need scene parameters (object sizes, velocities, background types) and
# the compute_event_rate() function from Ramaa's scene model (T1).
# os.path.abspath + os.path.join builds the absolute path to Remaas-work/
# regardless of where the script is called from.
# sys.path.insert(0, ...) adds that folder to Python's module search path
# so we can do a normal `from visualcomputing import ...`

SCENE_MODEL_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    '..', 'Remaas-work'
))
sys.path.insert(0, SCENE_MODEL_PATH)

from visualcomputing import (  # type: ignore
    object_sizes,       # list of object sizes in pixels (e.g. [10, 25, 50, 100])
    velocities,         # list of object velocities in px/s (e.g. [10, 50, 100, 200, 500])
    backgrounds,        # dict mapping background name → edge density (e.g. {'low_texture': 2, 'high_texture': 10})
    false_pos_rate,     # fraction of false positive events (noise events fired without real motion)
    scene_width,        # scene resolution width in pixels (640)
    scene_height,       # scene resolution height in pixels (480)
    compute_event_rate, # function(velocity, obj_size, bg_density, fp_rate) → raw event rate (events/sec)
)

# =============================================================================
# DVS HARDWARE PARAMETERS
# Source: Lichtsteiner et al. 2008 — "A 128×128 120dB 15µs Latency DVS"
# Sensor specs: 128×128 pixels, 3.3V supply, ~24mW total
# =============================================================================

# STATIC POWER — always on regardless of scene activity
# The DVS has two always-on current paths:
#   - Logic circuits:      0.3 mA
#   - Bias generators:     5.5 mA
# Total static current = 5.8 mA × 3.3V = 19.14 mW
# This covers ALL pixels on the array, even ones that aren't firing.
P_STATIC_MW = (0.3 + 5.5) * 3.3   # 19.14 mW

# DYNAMIC ENERGY PER EVENT — energy spent each time one pixel fires an event
# The dynamic current path draws ~1.5 mA while processing events.
# The original 128×128 sensor handles up to 1,000,000 events/sec.
# Energy per event = (1.5mA × 3.3V) / 1,000,000 events/sec = 4.95 nJ/event
E_PER_EVENT_NJ = (1.5 * 3.3 * 1e-3) / 1_000_000 * 1e9   # 4.95 nJ

# REFRACTORY PERIOD CAP — maximum achievable event rate for our scene resolution
# After each pixel fires, it goes "silent" for a refractory period (can't fire again).
# The original 128×128 chip maxes out at 1M events/sec.
# We scale that cap proportionally to our larger 640×480 scene:
#   cap = 1M × (640×480) / (128×128) = 18.75M events/sec
# This is the ceiling on event rate — no matter how fast the object moves,
# we can't exceed this rate physically.
REFRACTORY_CAP = 1_000_000 * (scene_width * scene_height) / (128 * 128)   # 18.75M ev/s

# =============================================================================
# TEMPORAL VARIATION SEQUENCE
# =============================================================================
# We simulate a 12-second clip where the object accelerates, peaks, then slows.
# This mimics a real tracking scenario (object enters frame, moves fast, exits).
# Velocity range is extended to 2000 px/s (beyond Ramaa's scene model range of 500)
# because at 500 px/s the dynamic power change is too small (~0.001 mW) relative
# to the 19.14 mW static floor to be visible on a plot.
# At 2000 px/s the variation becomes clearly visible.

TIME_STEPS        = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]             # seconds
VELOCITY_SEQUENCE = [10, 50, 100, 200, 500, 1000, 2000, 1000, 500, 200, 100, 50, 10]  # px/s
TEMPORAL_OBJ_SIZE = 50   # fixed object size (pixels) used for temporal variation

# =============================================================================
# CONTRAST THRESHOLD (THETA) SWEEP
# Source: Lichtsteiner eq. 5 — event_rate ∝ 1/θ
# θ is the minimum log-intensity change a pixel must detect before firing.
# Low θ  → fires on small changes → more events → more noise (false positives)
# High θ → fires only on large changes → fewer events → may miss faint edges (false negatives)
# =============================================================================

BASELINE_THETA = 0.20   # θ value that Ramaa's compute_event_rate() is calibrated to

THRESHOLDS = {
    "low_threshold":  0.10,   # half the baseline → 2× more events (noisier)
    "med_threshold":  0.20,   # baseline (default)
    "high_threshold": 0.40,   # double the baseline → 2× fewer events (cleaner but may miss edges)
}

# =============================================================================
# PIXEL FIRING BREAKDOWN
# =============================================================================

def compute_pixel_breakdown(obj_size: int, bg_density: float, velocity: float = 100) -> dict:
    """
    Estimates how many pixels are 'active' (firing events) vs 'silent' at a given instant.

    DVS pixels only fire when they detect a change in log-intensity (an edge moving past them).
    Most pixels in the array are doing nothing at any given time — static power covers them all.

    Parameters:
        obj_size    — object size in pixels (from Ramaa's scene model)
        bg_density  — background edge density (from Ramaa's backgrounds dict)
        velocity    — object velocity in px/s (affects how many background pixels get swept)

    Returns dict with total, active, silent pixel counts and active fraction.
    """

    # Total pixels in the scene (640 × 480 = 307,200)
    n_total = scene_width * scene_height

    # Object edge pixels: a square object of side `obj_size` has 4 × obj_size edge pixels.
    # These are the pixels right at the boundary of the object — they see a contrast change
    # as the object moves past them. (Interior pixels don't see an edge — no event.)
    n_obj_active = 4 * obj_size

    # Background pixels activated by motion:
    # A cluttered background has more edges that get "swept" by the moving object.
    # The fraction activated scales with bg_density AND velocity:
    #   - Higher bg_density → more texture → more edges to fire
    #   - Higher velocity → object sweeps faster → more edges activate per second
    # 0.01 is a scaling factor to convert bg_density (edge count per 100 px) to a fraction.
    # (velocity / 100) normalizes relative to baseline velocity of 100 px/s.
    n_bg_active = int(bg_density * n_total * 0.01 * (velocity / 100))

    # Total active pixels, capped at the total number of pixels in the scene
    n_active = min(n_obj_active + n_bg_active, n_total)

    # Silent pixels: everything else — always-on static power covers these too
    n_silent = n_total - n_active

    return {
        'n_total_pixels':  n_total,
        'n_active_pixels': n_active,
        'n_silent_pixels': n_silent,
        'active_fraction': round(n_active / n_total, 5),   # e.g. 0.00065 = 0.065%
    }

# =============================================================================
# CORE DVS POWER MODEL
# =============================================================================

def compute_dvs_power(event_rate_raw: float, theta: float = BASELINE_THETA) -> dict:
    """
    Computes DVS total power given a raw event rate and contrast threshold.

    Steps:
      1. Scale raw event rate for the chosen threshold (θ).
         Ramaa's compute_event_rate() assumes θ = BASELINE_THETA (0.20).
         From Lichtsteiner eq. 5: event_rate ∝ 1/θ
         So if we lower θ (more sensitive), we get proportionally more events.
         Scaling: event_rate_scaled = event_rate_raw × (BASELINE_THETA / theta)

      2. Apply refractory period cap.
         The sensor physically cannot process more than REFRACTORY_CAP events/sec.
         Any raw rate above this gets clipped.

      3. Compute dynamic power.
         Each event consumes E_PER_EVENT_NJ nanojoules.
         Power = event_rate_eff × E_per_event (converted to mW)

      4. Add static power.
         Total power = static + dynamic

    Parameters:
        event_rate_raw — raw events/sec from Ramaa's scene model (at baseline θ)
        theta          — contrast threshold being applied (from THRESHOLDS dict)

    Returns dict with breakdown of event rates and power components.
    """

    # Step 1: Scale event rate for chosen threshold
    # e.g. theta=0.10 (half of 0.20) → 2× more events than baseline
    event_rate_scaled = event_rate_raw * (BASELINE_THETA / theta)

    # Step 2: Check if we've hit the refractory period ceiling
    saturated = event_rate_scaled > REFRACTORY_CAP
    event_rate_eff = min(event_rate_scaled, REFRACTORY_CAP)   # effective (capped) event rate

    # Step 3: Dynamic power
    # event_rate_eff (ev/s) × E_PER_EVENT_NJ (nJ/ev) × 1e-9 (J/nJ) × 1e3 (mW/W)
    power_dynamic_mW = (event_rate_eff * E_PER_EVENT_NJ * 1e-9) * 1e3

    # Step 4: Total power
    power_total_mW = P_STATIC_MW + power_dynamic_mW

    return {
        'event_rate_scaled': round(event_rate_scaled, 1),
        'event_rate_eff':    round(event_rate_eff, 1),
        'power_static_mW':   round(P_STATIC_MW, 3),
        'power_dynamic_mW':  round(power_dynamic_mW, 3),
        'power_total_mW':    round(power_total_mW, 3),
        'saturated':         int(saturated),   # 1 if refractory cap was hit, else 0
    }

# =============================================================================
# RUN ALL SCENES
# Sweeps over every combination of: background × object size × velocity × threshold
# This is the main data-generation function — produces the summary CSV.
# =============================================================================

def run_all_scenes() -> pd.DataFrame:
    """
    Runs compute_dvs_power() for every combination of scene parameters.
    Returns a flat DataFrame with one row per (bg, obj_size, vel, threshold) combo.
    """
    rows = []
    for bg_name, bg_density in backgrounds.items():       # e.g. low_texture, high_texture
        for obj_size in object_sizes:                     # e.g. [10, 25, 50, 100] px
            for vel in velocities:                        # e.g. [10, 50, 100, 200, 500] px/s
                # Get raw event rate from Ramaa's scene model for this (vel, obj_size, bg) combo
                raw_rate = compute_event_rate(vel, obj_size, bg_density, false_pos_rate)
                # Get pixel breakdown for this scene configuration
                px_info  = compute_pixel_breakdown(obj_size, bg_density, vel)
                for th_name, theta in THRESHOLDS.items():
                    rows.append({
                        'object_size_px': obj_size,
                        'velocity_px_s':  vel,
                        'background':     bg_name,
                        'threshold_name': th_name,
                        'theta':          theta,
                        'event_rate_raw': raw_rate,
                        # Unpack the power model output dict directly into the row
                        **compute_dvs_power(raw_rate, theta),
                        # Unpack the pixel breakdown dict directly into the row
                        **px_info,
                    })
    return pd.DataFrame(rows)

# =============================================================================
# PLOT: PIXEL FIRING BREAKDOWN (DONUT CHARTS)
# Shows the ratio of active (firing) pixels vs silent pixels at worst-case velocity.
# Active fraction is typically <1% — this plot makes that visually clear.
# =============================================================================

def plot_pixel_breakdown(df, out_dir):
    # Worst-case velocity: most pixels are firing here, largest active fraction
    worst_vel = max(VELOCITY_SEQUENCE)   # 2000 px/s

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, (bg_name, bg_df) in zip(axes, df.groupby('background')):
        # Recompute pixel breakdown at worst-case velocity (may exceed scene model range)
        bg_density = backgrounds[bg_name]
        px = compute_pixel_breakdown(50, bg_density, worst_vel)
        n_active   = px['n_active_pixels']
        n_silent   = px['n_silent_pixels']
        n_total    = px['n_total_pixels']
        pct_active = px['active_fraction'] * 100
        pct_silent = 100 - pct_active

        # Draw donut: [silent, active] — silent is the large slice, active is tiny
        # width=0.45 makes it a donut (ring) rather than a filled pie
        wedges, _ = ax.pie(
            [n_silent, n_active],
            colors=['steelblue', 'tomato'],
            startangle=90,
            wedgeprops=dict(width=0.45),
        )
        # Centre text: show the dominant (silent) percentage
        ax.text(0, 0.12, f'{pct_silent:.2f}%', ha='center', va='center',
                fontsize=16, fontweight='bold', color='steelblue')
        ax.text(0, -0.18, 'silent', ha='center', va='center',
                fontsize=10, color='steelblue')

        # Actual pixel counts below the donut for reference
        ax.text(0, -0.72,
                f'Silent:  {n_silent:,} px\nActive:  {n_active:,} px\nTotal:   {n_total:,} px',
                ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5f5f5', edgecolor='#cccccc'))

        ax.set_title(f'{bg_name}\n(velocity = {worst_vel} px/s)', fontsize=11, fontweight='bold')
        ax.legend(wedges, [f'Silent ({pct_silent:.2f}%)', f'Active ({pct_active:.3f}%)'],
                  loc='upper right', fontsize=8)

    plt.suptitle('DVS: Active vs Silent Pixels at Worst-Case Velocity\n'
                 'Static power covers ALL pixels — active fraction is tiny',
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'dvs_pixel_breakdown.png'), dpi=200)
    plt.close()

# =============================================================================
# TEMPORAL VARIATION — RUN AND PLOT
# Simulates DVS power over a 12-second clip with changing object velocity.
# Key insight: DVS power dynamically tracks scene activity.
# CIS must be configured for worst-case velocity and stays there the whole time.
# =============================================================================

def run_temporal_variation() -> pd.DataFrame:
    """
    Runs the DVS power model at each time step in VELOCITY_SEQUENCE.
    Returns a DataFrame with one row per (time_step, background) combo.
    """
    rows = []
    for t, vel in zip(TIME_STEPS, VELOCITY_SEQUENCE):
        for bg_name, bg_density in backgrounds.items():
            # Get raw event rate from scene model at this velocity
            raw_rate = compute_event_rate(vel, TEMPORAL_OBJ_SIZE, bg_density, false_pos_rate)
            rows.append({
                'time_s':         t,
                'velocity_px_s':  vel,
                'background':     bg_name,
                'event_rate_raw': raw_rate,
                # Use default (baseline) threshold for temporal variation
                **compute_dvs_power(raw_rate),
            })
    return pd.DataFrame(rows)


# CIS must be configured for the fastest object it will ever see — it can't adapt frame-by-frame.
# So it runs at worst-case FPS (and power) even during slow moments.
def get_cis_worst_case_power_mw() -> float | None:
    """
    Reads Harshita's CIS model output CSV and returns the CIS power at worst-case velocity.
    Returns None if the CSV doesn't exist (e.g. running without Harshita's code).

    Why worst-case?
    CIS must be pre-configured for the fastest possible object. It can't lower its FPS
    dynamically the way DVS can — it's always running at max required frame rate.
    So for a fair comparison, we use the CIS power at the highest velocity in the scene.

    Why cap at 500?
    Harshita's CSV only contains data up to 500 px/s. Our temporal variation goes up to
    2000 px/s, so we cap the lookup at 500 to avoid an empty result.
    """
    cis_csv = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'Harshithas-work',
        'ModuCIS.-CIS-modeling-main', 'ModuCIS.-CIS-modeling-main',
        'CIS_Model', 'Use_cases', 'sweeps_results_final_cis_model',
        'cis_all_scenes_summary.csv'
    ))
    if not os.path.exists(cis_csv):
        return None
    import pandas as _pd
    cis_df = _pd.read_csv(cis_csv)
    # Cap at 500 — Harshita's CSV only covers up to 500 px/s regardless of scene model range
    worst_vel = min(max(velocities), 500)
    row = cis_df[(cis_df['velocity_px_s'] == worst_vel) &
                 (cis_df['object_size_px'] == TEMPORAL_OBJ_SIZE)]
    return float(row['power_mW'].values[0]) if not row.empty else None


def plot_temporal_variation(df_temporal, out_dir):
    """
    Plots DVS total power over time (one panel per background type).
    Each point is annotated with the velocity at that time step.
    CIS worst-case power is shown as a horizontal line (or annotation if it's off-chart).
    """
    cis_mw = get_cis_worst_case_power_mw()
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharey=True)   # shared y-axis for easy comparison

    for ax, (bg_name, bg_df) in zip(axes, df_temporal.groupby('background')):
        bg_df = bg_df.sort_values('time_s')

        # DVS power line — blue, filled below for visual weight
        ax.plot(bg_df['time_s'], bg_df['power_total_mW'],
                marker='o', color='#2255AA', linewidth=2.5, label='DVS power')
        ax.fill_between(bg_df['time_s'], bg_df['power_total_mW'],
                        alpha=0.15, color='#2255AA')

        # Label each point with its velocity (px/s)
        for _, r in bg_df.iterrows():
            ax.annotate(f"{int(r['velocity_px_s'])}px/s",
                        (r['time_s'], r['power_total_mW']),
                        textcoords='offset points', xytext=(0, 8),
                        fontsize=7, ha='center')

        # Show CIS line if we have data
        if cis_mw is not None:
            dvs_max = bg_df['power_total_mW'].max()
            if cis_mw <= dvs_max * 1.3:
                # CIS is close enough to DVS range — draw as dashed horizontal line
                ax.axhline(y=cis_mw, color='orange', linewidth=2, linestyle='--',
                           label=f'CIS power (fixed @ {cis_mw:.1f} mW)')
            else:
                # CIS is far above DVS range — don't distort y-axis, annotate instead
                ax.annotate(f'CIS power: {cis_mw:.1f} mW (above chart)',
                            xy=(0.02, 0.95), xycoords='axes fraction',
                            fontsize=8, color='black',
                            bbox=dict(boxstyle='round', facecolor='#fff8e7', edgecolor='orange'))

        # Set y-axis floor just below static power so dynamic variation is visible
        # Without this, matplotlib auto-scales to include 0, making the ~0.001 mW
        # dynamic variation invisible against the 19.14 mW static baseline.
        ax.set_ylim(bottom=P_STATIC_MW * 0.97)
        ax.set(title=f'DVS Power Over Time ({bg_name})',
               xlabel='Time (s)', ylabel='Power (mW)')
        ax.legend(); ax.grid(True)

    plt.suptitle('DVS: Temporal Variation — Power Tracks Scene Activity\n'
                 '(CIS locked to worst-case FPS regardless of velocity)',
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'dvs_temporal_variation.png'), dpi=200)
    plt.close()

# =============================================================================
# PLOT: DVS POWER VS VELOCITY
# Shows how total DVS power changes with object velocity, for each threshold.
# One panel per background type — lets you compare low vs high texture directly.
# =============================================================================

def plot_power_vs_velocity(df, out_dir):
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharey=True)
    for ax, (bg_name, bg_df) in zip(axes, df.groupby('background')):
        for th_name, th_df in bg_df.groupby('threshold_name'):
            # Filter to object size 50 px for a clean single-line comparison per threshold
            sub = th_df[th_df['object_size_px'] == 50].sort_values('velocity_px_s')
            ax.plot(sub['velocity_px_s'], sub['power_total_mW'], marker='o', label=th_name)
        ax.set(title=f'DVS Power vs Velocity ({bg_name})',
               xlabel='Velocity (px/s)', ylabel='DVS Total Power (mW)')
        ax.legend(); ax.grid(True)
    plt.suptitle('DVS: Power vs Velocity', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'dvs_power_vs_velocity.png'), dpi=200)
    plt.close()

# =============================================================================
# PLOT: DVS POWER VS BACKGROUND
# Compares low_texture vs high_texture background at medium threshold.
# High texture = more edges in background = more events = higher power.
# =============================================================================

def plot_power_vs_background(df, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for bg_name, bg_df in df[df['threshold_name'] == 'med_threshold'].groupby('background'):
        sub = bg_df[bg_df['object_size_px'] == 50].sort_values('velocity_px_s')
        ax.plot(sub['velocity_px_s'], sub['power_total_mW'], marker='s', label=bg_name)
    ax.set(title='DVS Power: Low vs High Texture (med_threshold)',
           xlabel='Velocity (px/s)', ylabel='DVS Total Power (mW)')
    ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'dvs_power_vs_background.png'), dpi=200)
    plt.close()

# =============================================================================
# PLOT: DVS POWER VS THRESHOLD (THETA SWEEP)
# Shows the sensitivity tradeoff: lower θ = more events = more power.
# Each line is a different velocity — shows interaction between speed and threshold.
# =============================================================================

def plot_power_vs_threshold(df, out_dir):
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharey=True)
    for ax, (bg_name, bg_df) in zip(axes, df.groupby('background')):
        for vel, vel_df in bg_df[bg_df['object_size_px'] == 50].groupby('velocity_px_s'):
            sub = vel_df.sort_values('theta')
            ax.plot(sub['theta'], sub['power_total_mW'], marker='o', label=f'{vel} px/s')
        ax.set(title=f'DVS Power vs Threshold ({bg_name})',
               xlabel='Threshold θ', ylabel='DVS Total Power (mW)')
        ax.legend(title='Velocity'); ax.grid(True)
    plt.suptitle('DVS: Power vs Threshold', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'dvs_power_vs_threshold.png'), dpi=200)
    plt.close()

# =============================================================================
# PLOT: STATIC VS DYNAMIC POWER BREAKDOWN
# Uses high_threshold + low_texture — cleanest scenario to isolate the components.
# Two panels:
#   Top: full stacked bar — shows static dominance visually
#   Bottom: dynamic power only, zoomed in — shows how it scales with velocity
# =============================================================================

def plot_static_vs_dynamic(df, out_dir):
    # Filter to one specific scenario for clarity
    sub = df[(df['threshold_name'] == 'high_threshold') &
             (df['background'] == 'low_texture') &
             (df['object_size_px'] == 50)].sort_values('velocity_px_s')
    x = sub['velocity_px_s'].astype(str)   # velocity as string for categorical x-axis

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    # Top panel: stacked bar — static (blue) stacked with dynamic (red)
    # The static bar will visually dominate (19.14 mW vs <0.01 mW dynamic)
    ax1.bar(x, sub['power_static_mW'], label='Static (mW)', color='steelblue')
    ax1.bar(x, sub['power_dynamic_mW'], label='Dynamic (mW)', color='tomato',
            bottom=sub['power_static_mW'])
    ax1.set(title='DVS Power Breakdown — Full View',
            xlabel='Velocity (px/s)', ylabel='Power (mW)')
    ax1.legend(); ax1.grid(True, axis='y')

    # Bottom panel: dynamic only — zoom in so the tiny dynamic component is visible
    # Labels show exact mW values on each bar
    ax2.bar(x, sub['power_dynamic_mW'], color='tomato', label='Dynamic (mW)')
    for i, (_, r) in enumerate(sub.iterrows()):
        ax2.text(i, r['power_dynamic_mW'] + 0.002,
                 f"{r['power_dynamic_mW']:.3f}", ha='center', fontsize=8)
    ax2.set(title='Dynamic Power Only — Zoomed',
            xlabel='Velocity (px/s)', ylabel='Dynamic Power (mW)')
    ax2.legend(); ax2.grid(True, axis='y')

    plt.suptitle('DVS Power Breakdown (high_threshold, low_texture)\n'
                 'Static power dominates — dynamic is barely visible at full scale',
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'dvs_power_breakdown.png'), dpi=200)
    plt.close()

# =============================================================================
# MAIN — runs when script is executed directly (not when imported)
# =============================================================================

if __name__ == '__main__':
    # Output directory for all plots and CSVs
    OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dvs_results')
    os.makedirs(OUT_DIR, exist_ok=True)

    # Run the full scene sweep and save results to CSV
    df = run_all_scenes()
    df.to_csv(os.path.join(OUT_DIR, 'dvs_all_scenes_summary.csv'), index=False)

    # Print key hardware constants for quick sanity check
    print(f'P_static = {P_STATIC_MW:.2f} mW | E_per_event = {E_PER_EVENT_NJ:.3f} nJ | '
          f'Refractory cap = {REFRACTORY_CAP/1e6:.2f}M ev/s')

    # Generate all plots
    plot_power_vs_velocity(df, OUT_DIR)
    plot_power_vs_background(df, OUT_DIR)
    plot_power_vs_threshold(df, OUT_DIR)
    plot_static_vs_dynamic(df, OUT_DIR)
    plot_pixel_breakdown(df, OUT_DIR)

    # Run and plot temporal variation
    df_temporal = run_temporal_variation()
    df_temporal.to_csv(os.path.join(OUT_DIR, 'dvs_temporal_variation.csv'), index=False)
    plot_temporal_variation(df_temporal, OUT_DIR)

    print('Done. Results in:', OUT_DIR)
