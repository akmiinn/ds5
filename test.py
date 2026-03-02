"""
================================================================================
  MEDITATION STATE CLASSIFIER — TESTING / INFERENCE SCRIPT (FIXED v2)
================================================================================
  Input : Muse 2 CSV with pre-computed band powers
          (Delta_TP9, Theta_AF7, Alpha_AF8, Beta_TP10, Gamma_*, ...)
           + RAW_TP9/AF7/AF8/TP10 columns (raw µV samples)
           + Accelerometer_X/Y/Z  (for motion artefact gating)

  Output: <filename>_predictions.csv
           <filename>_timeline.png

  FIXES
  ─────
  1. Bels→linear conversion: 10^value before computing relative power
  2. All 5 std_rel_ bands computed (was missing Gamma)
  3. Regional features use linear-converted values
  4. Relative paths, CLI argument support
  5. Confidence warnings for uncertain predictions
================================================================================
"""

import os
import sys
import glob
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ─── 1. PATHS ──────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
testing_dir = SCRIPT_DIR

# Accept target CSV from command line, or auto-detect S*.csv files
if len(sys.argv) > 1:
    target_file = sys.argv[1]
else:
    # Try to find any S*.csv file in the script directory
    candidates = sorted(glob.glob(os.path.join(SCRIPT_DIR, 'S47.csv')))
    if candidates:
        target_file = candidates[0]
        print(f"[INFO] Auto-detected: {target_file}")
        print(f"[INFO] Usage: python test.py <path_to_muse_csv>")
    else:
        print("[ERROR] No CSV specified and no S*.csv found in script directory.")
        print("        Usage: python test.py <path_to_muse_csv>")
        sys.exit(1)

model_path    = os.path.join(testing_dir, 'meditation_rf_classifier.pkl')
features_path = os.path.join(testing_dir, 'model_feature_names.pkl')
scaler_path   = os.path.join(testing_dir, 'scaler.pkl')

# ─── 2. CONSTANTS ──────────────────────────────────────────────────────────────
FS          = 256
WINDOW_SEC  = 2
STRIDE_SEC  = 1

MUSE_CHANNELS = ['TP9', 'AF7', 'AF8', 'TP10']
BANDS         = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

MOTION_THRESHOLD = 0.08

LEVEL_NAMES  = {0: "Baseline", 1: "Relaxed", 2: "Focused", 3: "Deep"}
LEVEL_COLORS = {0: '#e74c3c', 1: '#f39c12', 2: '#2ecc71', 3: '#3498db'}

eps = 1e-8

# ─── 3. LOAD MODEL ARTEFACTS ───────────────────────────────────────────────────
print("=" * 70)
print("  MEDITATION CLASSIFIER — INFERENCE (FIXED v2)")
print("=" * 70)

for path, label in [(model_path, 'Model'),
                    (features_path, 'Feature names'),
                    (scaler_path,   'Scaler')]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[ERROR] {label} not found: {path}\n"
            "        Run train.py first to generate .pkl files."
        )

with open(model_path,    'rb') as f: model         = pickle.load(f)
with open(features_path, 'rb') as f: feature_names = pickle.load(f)
with open(scaler_path,   'rb') as f: scaler        = pickle.load(f)

print(f"[LOAD] Model   : {model_path}")
print(f"[LOAD] Features: {len(feature_names)} features")
print(f"[LOAD] Scaler  : {scaler_path}")

# ─── 4. LOAD MUSE 2 CSV ────────────────────────────────────────────────────────
print(f"\n[DATA] Loading: {target_file}")
if not os.path.exists(target_file):
    raise FileNotFoundError(f"[ERROR] CSV not found: {target_file}")

raw_df = pd.read_csv(target_file, low_memory=False)
raw_df.dropna(how='all', inplace=True)
raw_df.reset_index(drop=True, inplace=True)
print(f"[DATA] Shape: {raw_df.shape}")

# ─── 4a. Detect band-power columns ────────────────────────────────────────────
band_cols = {}
for band in BANDS:
    band_cols[band] = {}
    for ch in MUSE_CHANNELS:
        col = f'{band}_{ch}'
        if col in raw_df.columns:
            band_cols[band][ch] = col
        else:
            alts = [c for c in raw_df.columns
                    if band.lower() in c.lower() and ch.lower() in c.lower()]
            if alts:
                band_cols[band][ch] = alts[0]

found_bands = {b: list(band_cols[b].values()) for b in BANDS if band_cols[b]}
for b, cols in found_bands.items():
    print(f"  {b:7s}: {cols}")

if not found_bands:
    raise ValueError(
        f"[ERROR] No band-power columns found.\n        Columns: {list(raw_df.columns)}"
    )

all_bp_cols = [c for b in BANDS for c in band_cols[b].values()]
raw_df[all_bp_cols] = raw_df[all_bp_cols].apply(pd.to_numeric, errors='coerce')

# ─── 4b. RAW EEG columns ──────────────────────────────────────────────────────
raw_eeg_cols = {}
for ch in MUSE_CHANNELS:
    col = f'RAW_{ch}'
    if col in raw_df.columns:
        raw_eeg_cols[ch] = col

# ─── 4c. IMU columns ──────────────────────────────────────────────────────────
imu_cols = [c for c in ['Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z']
            if c in raw_df.columns]
if not imu_cols:
    imu_cols = [c for c in raw_df.columns
                if 'acc' in c.lower() or 'accel' in c.lower()]
has_imu = len(imu_cols) >= 1
if has_imu:
    raw_df[imu_cols] = raw_df[imu_cols].apply(pd.to_numeric, errors='coerce')
    print(f"[DATA] IMU: {imu_cols}")
else:
    print("[DATA] No IMU — motion gating disabled")

# ─── 4d. Timestamps & sample rate ─────────────────────────────────────────────
time_col = None
for c in ['TimeStamp', 'timestamps', 'Timestamp', 'time', 'Time']:
    if c in raw_df.columns:
        time_col = c
        break

if time_col:
    raw_df[time_col] = pd.to_datetime(raw_df[time_col], errors='coerce')
    raw_df.dropna(subset=[time_col], inplace=True)
    raw_df.reset_index(drop=True, inplace=True)

    if len(raw_df) > 100:
        ts_sec = raw_df[time_col].astype(np.int64) / 1e9
        dt = np.diff(ts_sec[:500])
        dt = dt[(dt > 0) & (dt < 1.0)]
        if len(dt) > 10:
            est_fs = int(round(1.0 / np.median(dt)))
            if 50 <= est_fs <= 512:
                FS = est_fs
                print(f"[DATA] Sample rate: {FS} Hz")

WINDOW_SAMP = FS * WINDOW_SEC
STRIDE_SAMP = FS * STRIDE_SEC
print(f"[DATA] {len(raw_df)} rows  (~{len(raw_df)/FS:.0f}s at {FS} Hz)")

# ─── 4e. Quality flags ────────────────────────────────────────────────────────
hsi_cols = [c for c in raw_df.columns if c.startswith('HSI_')]
headband_col = 'HeadBandOn' if 'HeadBandOn' in raw_df.columns else None

if headband_col:
    raw_df[headband_col] = pd.to_numeric(raw_df[headband_col], errors='coerce')

quality_ok = pd.Series(True, index=raw_df.index)
if headband_col:
    quality_ok &= (raw_df[headband_col] == 1)
    n_off = (~quality_ok).sum()
    if n_off:
        print(f"[QUALITY] {n_off} rows HeadBandOn=0")

if hsi_cols:
    raw_df[hsi_cols] = raw_df[hsi_cols].apply(pd.to_numeric, errors='coerce')
    hsi_bad = (raw_df[hsi_cols] > 2).any(axis=1)
    quality_ok &= ~hsi_bad
    print(f"[QUALITY] {hsi_bad.sum()} rows poor HSI")

# ─── 5. WINDOWING & FEATURE EXTRACTION ────────────────────────────────────────
print(f"\n[FEATURES] Windowing ({WINDOW_SEC}s window, {STRIDE_SEC}s stride)...")

n_rows = len(raw_df)
n_win  = max(0, (n_rows - WINDOW_SAMP) // STRIDE_SAMP + 1)
print(f"[FEATURES] Expected windows: {n_win}")

records       = []
window_times  = []
skipped_motion  = 0
skipped_quality = 0


def get_ch_rel_power(win_df, band_name, channel_name, total_pwr):
    """Convert one channel's band power from Bels→linear→relative."""
    if channel_name in band_cols[band_name]:
        vals = pd.to_numeric(win_df[band_cols[band_name][channel_name]],
                             errors='coerce').dropna()
        if len(vals) > 0:
            return float((10.0 ** vals).mean()) / total_pwr
    return np.nan


for i in range(n_win):
    start = i * STRIDE_SAMP
    end   = start + WINDOW_SAMP
    win   = raw_df.iloc[start:end]

    # Quality gate
    if quality_ok.iloc[start:end].mean() < 0.70:
        skipped_quality += 1
        continue

    # Motion gate
    if has_imu:
        imu_vals = win[imu_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
        resultant = np.sqrt((imu_vals ** 2).sum(axis=1))
        if (resultant - resultant.mean()).std() > MOTION_THRESHOLD:
            skipped_motion += 1
            continue

    # ── Convert band powers: Bels → linear → relative ──
    row = {}
    bp_linear = {}
    for band in BANDS:
        ch_vals = []
        for ch, col in band_cols[band].items():
            vals = pd.to_numeric(win[col], errors='coerce').dropna()
            if len(vals) > 0:
                ch_vals.append(float((10.0 ** vals).mean()))
        bp_linear[band] = np.array(ch_vals) if ch_vals else np.array([0.0])

    total_power = sum(bp_linear[b].mean() for b in BANDS) + eps

    # Group A: mean relative power (5)
    for band in BANDS:
        row[f'mean_rel_{band}'] = float(bp_linear[band].mean()) / total_power

    # Group B: std relative power — ALL 5 bands (5)
    for band in BANDS:
        if len(bp_linear[band]) > 1:
            row[f'std_rel_{band}'] = float((bp_linear[band] / total_power).std())
        else:
            row[f'std_rel_{band}'] = 0.0

    # Group C: spectral ratios (6)
    row['ratio_theta_alpha'] = row['mean_rel_Theta'] / (row['mean_rel_Alpha'] + eps)
    row['ratio_alpha_beta']  = row['mean_rel_Alpha'] / (row['mean_rel_Beta']  + eps)
    row['ratio_delta_alpha'] = row['mean_rel_Delta'] / (row['mean_rel_Alpha'] + eps)
    row['ratio_theta_beta']  = row['mean_rel_Theta'] / (row['mean_rel_Beta']  + eps)
    row['ratio_delta_beta']  = row['mean_rel_Delta'] / (row['mean_rel_Beta']  + eps)
    row['ratio_gamma_alpha'] = row['mean_rel_Gamma'] / (row['mean_rel_Alpha'] + eps)

    # Group D: regional features (3)
    af7_a = get_ch_rel_power(win, 'Alpha', 'AF7', total_power)
    af8_a = get_ch_rel_power(win, 'Alpha', 'AF8', total_power)
    row['frontal_alpha_asymm'] = (
        (af7_a - af8_a) if not np.isnan(af7_a) and not np.isnan(af8_a) else 0.0
    )

    ft = [get_ch_rel_power(win, 'Theta', ch, total_power) for ch in ['AF7', 'AF8']]
    ft = [v for v in ft if not np.isnan(v)]
    row['frontal_theta_mean'] = float(np.mean(ft)) if ft else row['mean_rel_Theta']

    oa = [get_ch_rel_power(win, 'Alpha', ch, total_power) for ch in ['TP9', 'TP10']]
    oa = [v for v in oa if not np.isnan(v)]
    row['occipital_alpha_mean'] = float(np.mean(oa)) if oa else row['mean_rel_Alpha']

    if time_col:
        window_times.append(raw_df[time_col].iloc[start])
    else:
        window_times.append(start / FS)

    records.append(row)

print(f"[FEATURES] Valid    : {len(records)}")
print(f"[FEATURES] Skipped  : {skipped_motion} motion, {skipped_quality} quality")

if not records:
    raise RuntimeError("[ERROR] No valid windows. Check signal quality.")

feat_df = pd.DataFrame(records)

# ─── 6. ALIGN FEATURES ───────────────────────────────────────────────────────
missing = [f for f in feature_names if f not in feat_df.columns]
if missing:
    print(f"[WARN] {len(missing)} missing features (filled 0): {missing}")
    for m in missing:
        feat_df[m] = 0.0

X_test = feat_df[feature_names].values.astype(np.float64)
bad = ~np.isfinite(X_test)
if bad.any():
    print(f"[WARN] {bad.sum()} NaN/Inf → 0")
    X_test[bad] = 0.0

# ─── 7. PREDICT ──────────────────────────────────────────────────────────────
X_scaled     = scaler.transform(X_test)
predictions  = model.predict(X_scaled)
probabilities = model.predict_proba(X_scaled)
confidence   = probabilities.max(axis=1)

print("\n[PREDICT] Distribution:")
for lvl in sorted(LEVEL_NAMES):
    mask = predictions == lvl
    n = mask.sum()
    pct = 100 * n / len(predictions)
    bar = '█' * int(pct / 2)
    print(f"  L{lvl} {LEVEL_NAMES[lvl]:10s}: {n:4d} ({pct:5.1f}%)  {bar}")

# ─── 8. RESULTS ─────────────────────────────────────────────────────────────
results = feat_df[feature_names].copy()
results.insert(0, 'window_time', window_times[:len(results)])
results['predicted_level'] = predictions
results['level_name']      = [LEVEL_NAMES.get(p, str(p)) for p in predictions]
results['confidence']      = np.round(confidence, 4)
n_classes = probabilities.shape[1]
for i, lvl in enumerate(sorted(LEVEL_NAMES)[:n_classes]):
    results[f'prob_L{lvl}'] = np.round(probabilities[:, i], 4)

# ─── 9. SUMMARY ─────────────────────────────────────────────────────────────
total_s = len(raw_df) / FS
print(f"\n[SUMMARY] Duration    : {total_s:.0f}s ({total_s/60:.1f} min)")
print(f"[SUMMARY] Confidence  : {confidence.mean():.3f} mean, "
      f"{np.median(confidence):.3f} median")
dominant = LEVEL_NAMES.get(int(np.bincount(predictions).argmax()), '?')
print(f"[SUMMARY] Dominant    : {dominant}")
for lvl in sorted(LEVEL_NAMES):
    secs = (predictions == lvl).sum() * STRIDE_SEC
    print(f"  L{lvl} {LEVEL_NAMES[lvl]:10s}: {secs:5.0f}s  ({100*secs/max(1,total_s):.1f}%)")

# Confidence analysis
low_conf  = (confidence < 0.50).sum()
med_conf  = ((confidence >= 0.50) & (confidence < 0.70)).sum()
high_conf = (confidence >= 0.70).sum()
print(f"\n[CONFIDENCE] High (≥70%): {high_conf} ({100*high_conf/len(confidence):.1f}%)")
print(f"[CONFIDENCE] Med (50-70%): {med_conf} ({100*med_conf/len(confidence):.1f}%)")
print(f"[CONFIDENCE] Low (<50%)  : {low_conf} ({100*low_conf/len(confidence):.1f}%)")

if low_conf / len(confidence) > 0.20:
    print("\n⚠️  >20% low-confidence windows. Possible causes:")
    print("   • Poor electrode contact during recording")
    print("   • Domain gap: model trained on 64-ch research EEG, you use 4-ch Muse")
    print("   • Subject's EEG pattern differs from training population (monks)")
    print("   Suggestion: focus analysis on HIGH-confidence windows only.")
elif low_conf > 0:
    print(f"\n[INFO] {low_conf} low-confidence windows — normal for consumer EEG.")

# ─── 10. SAVE ────────────────────────────────────────────────────────────────
base = os.path.splitext(os.path.basename(target_file))[0]
out_csv = os.path.join(testing_dir, f'{base}_predictions.csv')
results.to_csv(out_csv, index=False)
print(f"\n[SAVE] CSV  → {out_csv}")

# ─── 11. PLOT ────────────────────────────────────────────────────────────────
if pd.api.types.is_datetime64_any_dtype(pd.Series(window_times)):
    t0 = pd.Timestamp(window_times[0])
    times = np.array([(pd.Timestamp(t) - t0).total_seconds()
                      for t in window_times[:len(predictions)]])
else:
    times = np.array(window_times[:len(predictions)], dtype=float)

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Subplot 1: Level
ax = axes[0]
for lvl in sorted(LEVEL_NAMES):
    mask = predictions == lvl
    if mask.any():
        ax.scatter(times[mask], np.full(mask.sum(), lvl),
                   color=LEVEL_COLORS[lvl], s=12, alpha=0.8,
                   label=f"L{lvl}: {LEVEL_NAMES[lvl]}")
ax.set_ylabel("Level")
ax.set_yticks(list(LEVEL_NAMES.keys()))
ax.set_yticklabels([f"L{k}" for k in LEVEL_NAMES.keys()])
ax.set_title("Predicted Meditation Level")
ax.legend(ncol=4, fontsize=8, loc='upper right')
ax.grid(alpha=0.3)

# Subplot 2: Confidence (color-coded)
ax = axes[1]
ax.fill_between(times, confidence, alpha=0.15, color='purple')
ax.plot(times, confidence, color='purple', linewidth=0.9, alpha=0.85)
ax.axhline(0.70, linestyle='--', color='green', linewidth=0.8, label='70% (high)')
ax.axhline(0.50, linestyle='--', color='red', linewidth=0.8, label='50% (low)')
ax.set_ylabel("Confidence")
ax.set_ylim(0, 1.05)
ax.set_title("Model Confidence (windows below red line are unreliable)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Subplot 3: Band powers
ax = axes[2]
band_colors = {'Delta': '#e74c3c', 'Theta': '#f39c12',
               'Alpha': '#2ecc71', 'Beta':  '#3498db', 'Gamma': '#9b59b6'}
for band in BANDS:
    col = f'mean_rel_{band}'
    if col in results.columns:
        ax.plot(times, results[col].values,
                label=band, color=band_colors[band], linewidth=0.9, alpha=0.85)
ax.set_ylabel("Relative Power")
ax.set_xlabel("Time (seconds)")
ax.set_title("Band Power Over Time")
ax.legend(fontsize=8, ncol=5)
ax.grid(alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(testing_dir, f'{base}_timeline.png')
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"[SAVE] Plot → {plot_path}")

print("\n" + "=" * 70)
print("  INFERENCE COMPLETE")
print(f"  CSV  : {out_csv}")
print(f"  Plot : {plot_path}")
print("=" * 70)
