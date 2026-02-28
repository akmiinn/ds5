"""
================================================================================
  MEDITATION STATE CLASSIFIER — TESTING / INFERENCE SCRIPT (Fixed)
================================================================================
  Input : Muse 2 CSV with pre-computed band powers
          (Delta_TP9, Theta_AF7, Alpha_AF8, Beta_TP10, Gamma_*, ...)
           + RAW_TP9/AF7/AF8/TP10 columns (raw µV samples)
           + Accelerometer_X/Y/Z  (for motion artefact gating)

  Output: meditation_predictions.csv
           meditation_timeline.png
           (saved to testing_dir)

  Feature pipeline (matches training exactly):
  ─────────────────────────────────────────────────────────────────────────────
  The Muse 2 CSV provides one row per sample (256 Hz) with repeated band-power
  values that update roughly once per second. We:
    1. Parse timestamps → detect sample rate
    2. Gate out high-motion samples using IMU accelerometer variance
    3. Group rows into 2-second windows (50 % overlap / 1-second stride)
    4. Per window, take the MEAN of each band-power column across all samples
       in that window (band powers are already relative 0–1 values)
    5. Compute the same 19 features used in training
    6. Predict meditation level with the loaded model

  This avoids re-running Welch PSD since the Muse already provides band powers.
================================================================================
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ─── 1. PATHS ──────────────────────────────────────────────────────────────────
testing_dir = '/Users/syian/Desktop/ds5'

target_file   = '/Users/syian/Desktop/S47.csv'
model_path    = os.path.join(testing_dir, 'meditation_rf_classifier.pkl')
features_path = os.path.join(testing_dir, 'model_feature_names.pkl')
scaler_path   = os.path.join(testing_dir, 'scaler.pkl')

# ─── 2. CONSTANTS ──────────────────────────────────────────────────────────────
FS          = 256       # Muse 2 nominal sample rate (Hz); auto-detected below
WINDOW_SEC  = 2         # window length in seconds
STRIDE_SEC  = 1         # 50 % overlap

MUSE_CHANNELS = ['TP9', 'AF7', 'AF8', 'TP10']
BANDS         = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

# Motion artefact threshold: std of resultant acceleration per window
# Windows above this are skipped  (units: g, typical walking ~0.5 g)
MOTION_THRESHOLD = 0.08

LEVEL_NAMES  = {0: "Baseline", 1: "Relaxed", 2: "Focused", 3: "Deep"}
LEVEL_COLORS = {0: '#e74c3c', 1: '#f39c12', 2: '#2ecc71', 3: '#3498db'}

eps = 1e-8

# ─── 3. LOAD MODEL ARTEFACTS ───────────────────────────────────────────────────
print("=" * 70)
print("  MEDITATION CLASSIFIER — INFERENCE")
print("=" * 70)

for path, label in [(model_path, 'Model'),
                    (features_path, 'Feature names'),
                    (scaler_path,   'Scaler')]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[ERROR] {label} not found: {path}\n"
            "        Run train_meditation_classifier.py first and copy the .pkl files."
        )

with open(model_path,    'rb') as f: model         = pickle.load(f)
with open(features_path, 'rb') as f: feature_names = pickle.load(f)
with open(scaler_path,   'rb') as f: scaler        = pickle.load(f)

print(f"[LOAD] Model          : {model_path}")
print(f"[LOAD] Feature names  : {feature_names}")
print(f"[LOAD] Scaler         : {scaler_path}")

# ─── 4. LOAD MUSE 2 CSV ────────────────────────────────────────────────────────
print(f"\n[DATA] Loading: {target_file}")
if not os.path.exists(target_file):
    raise FileNotFoundError(f"[ERROR] CSV not found: {target_file}")

raw_df = pd.read_csv(target_file, low_memory=False)

# Drop fully-empty rows (e.g. the first connection-event row)
raw_df.dropna(how='all', inplace=True)
raw_df.reset_index(drop=True, inplace=True)
print(f"[DATA] Shape after load: {raw_df.shape}")

# ─── 4a. Detect band-power columns ────────────────────────────────────────────
# Expected format: Delta_TP9, Theta_AF7, Alpha_AF8, Beta_TP10, Gamma_TP9, ...
band_cols = {}   # band → {channel → col_name}
for band in BANDS:
    band_cols[band] = {}
    for ch in MUSE_CHANNELS:
        col = f'{band}_{ch}'
        if col in raw_df.columns:
            band_cols[band][ch] = col
        else:
            # Try alternate capitalisation
            alts = [c for c in raw_df.columns
                    if band.lower() in c.lower() and ch.lower() in c.lower()]
            if alts:
                band_cols[band][ch] = alts[0]

found_bands = {b: list(band_cols[b].values()) for b in BANDS if band_cols[b]}
for b, cols in found_bands.items():
    print(f"  {b:7s}: {cols}")

if not found_bands:
    raise ValueError(
        "[ERROR] No band-power columns found in the CSV.\n"
        f"        CSV columns: {list(raw_df.columns)}"
    )

# Convert band power columns to numeric
all_bp_cols = [c for b in BANDS for c in band_cols[b].values()]
raw_df[all_bp_cols] = raw_df[all_bp_cols].apply(pd.to_numeric, errors='coerce')

# ─── 4b. Detect RAW EEG columns (for artefact reference only) ─────────────────
raw_eeg_cols = {}
for ch in MUSE_CHANNELS:
    col = f'RAW_{ch}'
    if col in raw_df.columns:
        raw_eeg_cols[ch] = col

# ─── 4c. Detect IMU columns ───────────────────────────────────────────────────
imu_cols = [c for c in ['Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z']
            if c in raw_df.columns]
if not imu_cols:
    imu_cols = [c for c in raw_df.columns
                if 'acc' in c.lower() or 'accel' in c.lower()]
has_imu = len(imu_cols) >= 1
if has_imu:
    raw_df[imu_cols] = raw_df[imu_cols].apply(pd.to_numeric, errors='coerce')
    print(f"[DATA] IMU columns: {imu_cols}")
else:
    print("[DATA] No IMU columns found — motion gating disabled")

# ─── 4d. Parse timestamps & estimate sample rate ──────────────────────────────
time_col = None
for c in ['TimeStamp', 'timestamps', 'Timestamp', 'time', 'Time']:
    if c in raw_df.columns:
        time_col = c
        break

if time_col:
    # Muse timestamps may be strings like "2025-11-14 11:35:38.265"
    raw_df[time_col] = pd.to_datetime(raw_df[time_col], errors='coerce')
    raw_df.dropna(subset=[time_col], inplace=True)
    raw_df.reset_index(drop=True, inplace=True)

    if len(raw_df) > 100:
        ts_sec = raw_df[time_col].astype(np.int64) / 1e9   # nanoseconds → seconds
        dt     = np.diff(ts_sec[:500])
        dt     = dt[(dt > 0) & (dt < 1.0)]
        if len(dt) > 10:
            estimated_fs = int(round(1.0 / np.median(dt)))
            if 50 <= estimated_fs <= 512:
                FS = estimated_fs
                print(f"[DATA] Estimated sample rate: {FS} Hz")

WINDOW_SAMP = FS * WINDOW_SEC
STRIDE_SAMP = FS * STRIDE_SEC
print(f"[DATA] Total rows: {len(raw_df)}  (~{len(raw_df)/FS:.0f} s at {FS} Hz)")

# ─── 4e. Handle HeadBandOn / HSI quality flags ────────────────────────────────
hsi_cols = [c for c in raw_df.columns if c.startswith('HSI_')]
headband_col = 'HeadBandOn' if 'HeadBandOn' in raw_df.columns else None

if headband_col:
    raw_df[headband_col] = pd.to_numeric(raw_df[headband_col], errors='coerce')

quality_ok = pd.Series(True, index=raw_df.index)
if headband_col:
    quality_ok &= (raw_df[headband_col] == 1)
    n_off = (~quality_ok).sum()
    if n_off:
        print(f"[QUALITY] {n_off} rows with HeadBandOn=0 flagged as low quality")

if hsi_cols:
    raw_df[hsi_cols] = raw_df[hsi_cols].apply(pd.to_numeric, errors='coerce')
    # HSI: 1=good, 2=OK, 3=poor, 4=very poor — keep ≤ 2
    hsi_bad = (raw_df[hsi_cols] > 2).any(axis=1)
    quality_ok &= ~hsi_bad
    print(f"[QUALITY] {hsi_bad.sum()} rows with poor HSI signal flagged")

# ─── 5. WINDOWING & FEATURE EXTRACTION ────────────────────────────────────────
print(f"\n[FEATURES] Windowing ({WINDOW_SEC}s, {STRIDE_SEC}s stride)...")

n_rows   = len(raw_df)
n_win    = max(0, (n_rows - WINDOW_SAMP) // STRIDE_SAMP + 1)
print(f"[FEATURES] Expected windows: {n_win}")

records       = []
window_times  = []
skipped_motion = 0
skipped_quality = 0

for i in range(n_win):
    start = i * STRIDE_SAMP
    end   = start + WINDOW_SAMP
    win   = raw_df.iloc[start:end]

    # ── Quality gate: skip if >30% low-quality samples
    if quality_ok.iloc[start:end].mean() < 0.70:
        skipped_quality += 1
        continue

    # ── Motion gate using IMU
    if has_imu:
        imu_vals = win[imu_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
        resultant = np.sqrt((imu_vals ** 2).sum(axis=1))
        # Remove gravity component (approx 1g)
        resultant_detrended = resultant - resultant.mean()
        motion_std = resultant_detrended.std()
        if motion_std > MOTION_THRESHOLD:
            skipped_motion += 1
            continue

    # ── Band power: mean over the window for each band × channel
    row = {}
    bp  = {}   # band → array of per-channel values in this window
    for band in BANDS:
        ch_vals = []
        for ch, col in band_cols[band].items():
            vals = pd.to_numeric(win[col], errors='coerce').dropna()
            if len(vals) > 0:
                ch_vals.append(float(vals.mean()))
        bp[band] = np.array(ch_vals) if ch_vals else np.array([0.0])

    # ── Engineer features (same as training)
    # Group A: mean relative power per band
    for band in BANDS:
        row[f'mean_rel_{band}'] = float(bp[band].mean())

    # Group B: std relative power per band
    for band in BANDS[:4]:
        row[f'std_rel_{band}'] = float(bp[band].std()) if len(bp[band]) > 1 else 0.0

    # Group C: spectral ratios
    row['ratio_theta_alpha'] = row['mean_rel_Theta'] / (row['mean_rel_Alpha'] + eps)
    row['ratio_alpha_beta']  = row['mean_rel_Alpha'] / (row['mean_rel_Beta']  + eps)
    row['ratio_delta_alpha'] = row['mean_rel_Delta'] / (row['mean_rel_Alpha'] + eps)
    row['ratio_theta_beta']  = row['mean_rel_Theta'] / (row['mean_rel_Beta']  + eps)
    row['ratio_delta_beta']  = row['mean_rel_Delta'] / (row['mean_rel_Beta']  + eps)
    row['ratio_gamma_alpha'] = row['mean_rel_Gamma'] / (row['mean_rel_Alpha'] + eps)

    # Group D: spatial / regional features
    # frontal_alpha_asymm → AF7_Alpha − AF8_Alpha
    af7_alpha = float(pd.to_numeric(win.get(band_cols['Alpha'].get('AF7', ''), pd.Series([np.nan])),
                                    errors='coerce').mean()) if 'AF7' in band_cols['Alpha'] else 0.0
    af8_alpha = float(pd.to_numeric(win.get(band_cols['Alpha'].get('AF8', ''), pd.Series([np.nan])),
                                    errors='coerce').mean()) if 'AF8' in band_cols['Alpha'] else 0.0
    row['frontal_alpha_asymm'] = af7_alpha - af8_alpha

    # frontal_theta_mean → mean(AF7_Theta, AF8_Theta)
    ft_vals = []
    for ch in ['AF7', 'AF8']:
        if ch in band_cols['Theta']:
            v = pd.to_numeric(win[band_cols['Theta'][ch]], errors='coerce').mean()
            if np.isfinite(v): ft_vals.append(v)
    row['frontal_theta_mean'] = float(np.mean(ft_vals)) if ft_vals else row['mean_rel_Theta']

    # occipital_alpha_mean → mean(TP9_Alpha, TP10_Alpha)  [closest Muse has to occipital]
    oa_vals = []
    for ch in ['TP9', 'TP10']:
        if ch in band_cols['Alpha']:
            v = pd.to_numeric(win[band_cols['Alpha'][ch]], errors='coerce').mean()
            if np.isfinite(v): oa_vals.append(v)
    row['occipital_alpha_mean'] = float(np.mean(oa_vals)) if oa_vals else row['mean_rel_Alpha']

    # timestamp for this window
    if time_col:
        window_times.append(raw_df[time_col].iloc[start])
    else:
        window_times.append(start / FS)

    records.append(row)

print(f"[FEATURES] Valid windows    : {len(records)}")
print(f"[FEATURES] Skipped (motion) : {skipped_motion}")
print(f"[FEATURES] Skipped (quality): {skipped_quality}")

if not records:
    raise RuntimeError(
        "[ERROR] No valid windows produced.\n"
        "        Check signal quality, HeadBandOn flag, and motion threshold."
    )

feat_df = pd.DataFrame(records)

# ─── 6. ALIGN WITH TRAINING FEATURES ─────────────────────────────────────────
missing = [f for f in feature_names if f not in feat_df.columns]
if missing:
    print(f"[WARN] {len(missing)} features from training not found — filling with 0:")
    for m in missing:
        print(f"       {m}")
        feat_df[m] = 0.0

X_test = feat_df[feature_names].values.astype(np.float64)

# Replace any NaN/Inf with 0
bad = ~np.isfinite(X_test)
if bad.any():
    print(f"[WARN] {bad.sum()} NaN/Inf values in features — replacing with 0")
    X_test[bad] = 0.0

# ─── 7. SCALE & PREDICT ───────────────────────────────────────────────────────
X_scaled     = scaler.transform(X_test)
predictions  = model.predict(X_scaled)
probabilities = model.predict_proba(X_scaled)
confidence   = probabilities.max(axis=1)

print("\n[PREDICT] Prediction distribution:")
for lvl in sorted(LEVEL_NAMES):
    mask = predictions == lvl
    n    = mask.sum()
    pct  = 100 * n / len(predictions)
    bar  = '█' * int(pct / 2)
    print(f"  L{lvl} {LEVEL_NAMES[lvl]:10s}: {n:4d} windows ({pct:5.1f}%)  {bar}")

# ─── 8. BUILD RESULTS DATAFRAME ───────────────────────────────────────────────
results = feat_df[feature_names].copy()
results.insert(0, 'window_time', window_times[:len(results)])
results['predicted_level'] = predictions
results['level_name']      = [LEVEL_NAMES.get(p, str(p)) for p in predictions]
results['confidence']      = np.round(confidence, 4)
n_classes = probabilities.shape[1]
for i, lvl in enumerate(sorted(LEVEL_NAMES)[:n_classes]):
    results[f'prob_L{lvl}'] = np.round(probabilities[:, i], 4)

# ─── 9. SESSION SUMMARY ───────────────────────────────────────────────────────
total_s = len(raw_df) / FS
print(f"\n[SUMMARY] Session duration  : {total_s:.0f}s  ({total_s/60:.1f} min)")
print(f"[SUMMARY] Mean confidence   : {confidence.mean():.3f}")
dominant = LEVEL_NAMES.get(int(np.bincount(predictions).argmax()), '?')
print(f"[SUMMARY] Dominant state    : {dominant}")
for lvl in sorted(LEVEL_NAMES):
    secs = (predictions == lvl).sum() * STRIDE_SEC
    print(f"[SUMMARY]   L{lvl} {LEVEL_NAMES[lvl]:10s}: {secs:5.0f}s  "
          f"({100*secs/max(1,total_s):.1f}%)")

# ─── 10. SAVE CSV ─────────────────────────────────────────────────────────────
out_csv = os.path.join(testing_dir, 'meditation_predictions.csv')
results.to_csv(out_csv, index=False)
print(f"\n[SAVE] Predictions CSV  → {out_csv}")

# ─── 11. TIMELINE PLOT ────────────────────────────────────────────────────────
# Convert window times to elapsed seconds for plotting
if pd.api.types.is_datetime64_any_dtype(pd.Series(window_times)):
    t0    = pd.Timestamp(window_times[0])
    times = np.array([(pd.Timestamp(t) - t0).total_seconds()
                      for t in window_times[:len(predictions)]])
else:
    times = np.array(window_times[:len(predictions)], dtype=float)

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# -- Subplot 1: Predicted level
ax = axes[0]
for lvl in sorted(LEVEL_NAMES):
    mask = predictions == lvl
    ax.scatter(times[mask], np.full(mask.sum(), lvl),
               color=LEVEL_COLORS[lvl], s=12, alpha=0.8,
               label=f"L{lvl}: {LEVEL_NAMES[lvl]}")
ax.set_ylabel("Meditation Level")
ax.set_yticks(list(LEVEL_NAMES.keys()))
ax.set_yticklabels([f"L{k}" for k in LEVEL_NAMES.keys()])
ax.set_title("Predicted Meditation Level Over Time")
ax.legend(ncol=4, fontsize=8, loc='upper right')
ax.grid(alpha=0.3)

# -- Subplot 2: Confidence
ax = axes[1]
ax.plot(times, confidence, color='purple', linewidth=0.9, alpha=0.85)
ax.fill_between(times, confidence, alpha=0.15, color='purple')
ax.axhline(0.5, linestyle='--', color='red', linewidth=0.8, label='50%')
ax.set_ylabel("Confidence")
ax.set_ylim(0, 1.05)
ax.set_title("Model Confidence Per Window")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# -- Subplot 3: Band powers over time
ax = axes[2]
band_colors_map = {'Delta': '#e74c3c', 'Theta': '#f39c12',
                   'Alpha': '#2ecc71', 'Beta':  '#3498db', 'Gamma': '#9b59b6'}
for band in BANDS:
    col = f'mean_rel_{band}'
    if col in results.columns:
        ax.plot(times, results[col].values,
                label=band, color=band_colors_map[band],
                linewidth=0.9, alpha=0.85)
ax.set_ylabel("Mean Relative Power")
ax.set_xlabel("Time (seconds)")
ax.set_title("Mean Relative Band Power Over Time")
ax.legend(fontsize=8, ncol=5)
ax.grid(alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(testing_dir, 'meditation_timeline.png')
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"[SAVE] Timeline plot    → {plot_path}")

print("\n" + "=" * 70)
print("  INFERENCE COMPLETE")
print(f"  CSV    : {out_csv}")
print(f"  Plot   : {plot_path}")
print("=" * 70)